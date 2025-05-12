#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use rand::Rng;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        neuron::{gpu_lattices::LatticeGPU, CellGrid, Lattice, RunLattice}
    };


    neuron_builder!("
        [neurotransmitter_kinetics]
            type: BoundedNeurotransmitterKinetics
            vars: t_max = 1, clearance_constant = 0.001, conc = 0
            on_iteration:
                [if] is_spiking [then]
                    conc = t_max
                [else]
                    conc = 0
                [end]

                t = t + dt * -clearance_constant * t + conc

                t = min(max(t, 0), t_max)
        [end]

        [receptor_kinetics]
            type: BoundedReceptorKinetics
            vars: r_max = 1
            on_iteration:
                r = min(max(t, 0), r_max)
        [end]

        [receptors]
            type: DopaGluGABA
            kinetics: BoundedReceptorKinetics
            vars: inh_modifier = 1, nmda_modifier = 1
            neurotransmitter: Glutamate
            receptors: ampa_r, nmda_r
            vars: current = 0, g_ampa = 1, g_nmda = 0.6, e_ampa = 0, e_nmda = 0, mg = 0.3
            on_iteration:
                current = inh_modifier * g_ampa * ampa_r * (v - e_ampa) + (1 / (1 + exp(-0.062 * v) * mg / 3.57)) * inh_modifier * g_nmda * (nmda_r ^ nmda_modifier) * (v - e_nmda)
            neurotransmitter: GABA
            vars: current = 0, g = 1.2, e = -80
            on_iteration:
                current = g * r * (v - e)
            neurotransmitter: Dopamine
            receptors: r_d1, r_d2
            vars: s_d2 = 0, s_d1 = 0
            on_iteration:
                inh_modifier = 1 - (r_d2 * s_d2)
                nmda_modifier = 1 - (r_d1 * s_d1)
        [end]

        [neuron]
            type: IzhikevichNeuron
            kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
            receptors: DopaGluGABA
            vars: u = 30, a = 0.02, b = 0.2, c = -55, d = 8, v_th = 30, tau_m = 1
            on_spike: 
                v = c
                u += d
            spike_detection: v >= v_th
            on_iteration:
                du/dt = (a * (b * v - u)) / tau_m
                dv/dt = (0.04 * v ^ 2 + 5 * v + 140 - u + i) / c_m
        [end]"
    );

    const ITERATIONS: usize = 1000;

    #[test]
    pub fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match IzhikevichNeuron::<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>::iterate_and_spike_electrical_kernel(&context) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        }
    }

    #[test]
    pub fn test_electrochemical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match IzhikevichNeuron::<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>::iterate_and_spike_electrochemical_kernel(&context) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        }
    }

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
        rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
        x != y
    }

    fn check_entire_history(cpu_grid_history: &[Vec<Vec<f32>>], gpu_grid_history: &[Vec<Vec<f32>>]) {
        for (cpu_cell_grid, gpu_cell_grid) in cpu_grid_history.iter()
            .zip(gpu_grid_history) {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= 3., "error: {}, voltage1: {}, voltage2: {}", 
                        error,
                        voltage1,
                        voltage2,
                    );
                }
            }
        }
    }

    fn check_last_state<U: IterateAndSpikeGPU, L: CellGrid<T=U>, G: CellGrid<T=U>>(lattice: &L, gpu_lattice: &G) {
        for (row1, row2) in lattice.cell_grid().iter().zip(gpu_lattice.cell_grid().iter()) {
            for (neuron1, neuron2) in row1.iter().zip(row2.iter()) {
                let error = (neuron1.get_current_voltage() - neuron2.get_current_voltage()).abs();
                assert!(
                    error <= 3., "error: {}, neuron1: {}, neuron2: {}\n{:#?}\n{:#?}", 
                    error,
                    neuron1.get_current_voltage(),
                    neuron2.get_current_voltage(),
                    lattice.cell_grid().iter()
                        .map(|i| i.iter().map(|j| j.get_current_voltage()).collect::<Vec<f32>>())
                        .collect::<Vec<Vec<f32>>>(),
                    gpu_lattice.cell_grid().iter()
                        .map(|i| i.iter().map(|j| j.get_current_voltage()).collect::<Vec<f32>>())
                        .collect::<Vec<Vec<f32>>>(),
                );
    
                let error = (
                    neuron1.get_last_firing_time().unwrap_or(0) as isize - 
                    neuron2.get_last_firing_time().unwrap_or(0) as isize
                ).abs();
                assert!(
                    error <= 2, "error: {:#?}, neuron1: {:#?}, neuron2: {:#?}",
                    error,
                    neuron1.get_last_firing_time(),
                    neuron2.get_last_firing_time(),
                );
            }
        }
    }

    #[test]
    pub fn test_electrical_lattice_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        let base_neuron = IzhikevichNeuron::default_impl();
    
        let (num_rows, num_cols) = (2, 2);

        let mut lattice = Lattice::default_impl();
        
        lattice.populate(
            &base_neuron, 
            num_rows, 
            num_cols, 
        )?;
    
        lattice.connect(&connection_conditional, None);

        lattice.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.c..=neuron.v_th);
        });
    
        lattice.update_grid_history = true;
    
        let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
    
        lattice.run_lattice(ITERATIONS)?;
    
        gpu_lattice.run_lattice(ITERATIONS)?;
    
        check_last_state(&lattice, &gpu_lattice);
    
        check_entire_history(&lattice.grid_history.history, &gpu_lattice.grid_history.history);

        Ok(())
    }

    #[test]
    pub fn test_chemical_lattice_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            let mut base_neuron = IzhikevichNeuron::default_impl();

            base_neuron.receptors
                .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
                .expect("Valid neurotransmitter pairing");
            base_neuron.synaptic_neurotransmitters
                .insert(DopaGluGABANeurotransmitterType::Glutamate, BoundedNeurotransmitterKinetics::default());
        
            let (num_rows, num_cols) = (2, 2);

            let mut lattice = Lattice::default_impl();

            lattice.electrical_synapse = false;
            lattice.chemical_synapse = true;
            
            lattice.populate(
                &base_neuron, 
                num_rows, 
                num_cols, 
            )?;
        
            lattice.connect(&connection_conditional, None);

            lattice.apply(|neuron: &mut _| {
                let mut rng = rand::thread_rng();
                neuron.current_voltage = rng.gen_range(neuron.c..=neuron.v_th);
            });
        
            lattice.update_grid_history = true;
        
            let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
        
            lattice.run_lattice(ITERATIONS)?;
        
            gpu_lattice.run_lattice(ITERATIONS)?;

            check_last_state(&lattice, &gpu_lattice);

            check_entire_history(&lattice.grid_history.history, &gpu_lattice.grid_history.history);
        }

        Ok(())
    }

    #[test]
    fn test_d1_functionality() {
        let glu_ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let dopamine_ts = glu_ts.clone();

        let mut spike_counts: Vec<Vec<usize>> = (0..11).map(|_| (0..11).map(|_| 0).collect()).collect();
        
        for (n, glu) in glu_ts.iter().enumerate() {
            for (m, dopamine) in dopamine_ts.iter().enumerate() {
                let mut neuron = IzhikevichNeuron::default_impl();

                neuron.receptors
                    .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
                    .expect("Valid neurotransmitter pairing");
                neuron.receptors
                    .insert(
                        DopaGluGABANeurotransmitterType::Dopamine, 
                        DopaGluGABAType::Dopamine(DopamineReceptor { s_d2: 0., s_d1: 1., ..DopamineReceptor::default() })
                    )
                    .expect("Valid neurotransmitter pairing");

                let t_total = HashMap::from([
                    (DopaGluGABANeurotransmitterType::Glutamate, *glu),
                    (DopaGluGABANeurotransmitterType::Dopamine, *dopamine)
                ]);

                let mut spikes = 0;
                for _ in 0..ITERATIONS {
                    let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &t_total);
                    if is_spiking {
                        spikes += 1;
                    }
                    match neuron.receptors.get(&DopaGluGABANeurotransmitterType::Dopamine).unwrap() {
                        DopaGluGABAType::Dopamine(receptor) => assert_eq!(receptor.r_d1.get_r(), *dopamine),
                        _ => unreachable!()
                    }
                    assert_eq!(neuron.receptors.nmda_modifier, 1. - *dopamine);
                    assert_eq!(neuron.receptors.inh_modifier, 1.);
                }

                spike_counts[n][m] = spikes;
            }
        }

        for i in 1..11 {
            for j in 1..11 {
                assert!(spike_counts[i][j] >= spike_counts[i][j - 1]);
                assert!(spike_counts[j][i] >= spike_counts[j - 1][i]);
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 3..8 {
            assert!(
                spike_counts[i][0] < spike_counts[i][10], 
                "{}: {} < {}", 
                i, 
                spike_counts[i][0], 
                spike_counts[i][10]
            );
        }
    }

    #[test]
    fn test_d2_functionality() {
        let glu_ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let dopamine_ts = glu_ts.clone();

        let mut spike_counts: Vec<Vec<usize>> = (0..11).map(|_| (0..11).map(|_| 0).collect()).collect();

        for (n, glu) in glu_ts.iter().enumerate() {
            for (m, dopamine) in dopamine_ts.iter().enumerate() {
                let mut neuron = IzhikevichNeuron::default_impl();

                neuron.receptors
                    .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
                    .expect("Valid neurotransmitter pairing");
                neuron.receptors
                    .insert(
                        DopaGluGABANeurotransmitterType::Dopamine, 
                        DopaGluGABAType::Dopamine(DopamineReceptor { s_d2: 0.5, s_d1: 0., ..DopamineReceptor::default() })
                    )
                    .expect("Valid neurotransmitter pairing");

                let t_total = HashMap::from([
                    (DopaGluGABANeurotransmitterType::Glutamate, *glu),
                    (DopaGluGABANeurotransmitterType::Dopamine, *dopamine)
                ]);

                let mut spikes = 0;
                for _ in 0..ITERATIONS {
                    let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &t_total);
                    if is_spiking {
                        spikes += 1;
                    }
                    match neuron.receptors.get(&DopaGluGABANeurotransmitterType::Dopamine).unwrap() {
                        DopaGluGABAType::Dopamine(receptor) => assert_eq!(receptor.r_d2.get_r(), *dopamine),
                        _ => unreachable!()
                    }
                    assert_eq!(neuron.receptors.nmda_modifier, 1.);
                    assert_eq!(neuron.receptors.inh_modifier, 1. - (0.5 * *dopamine));
                }

                spike_counts[n][m] = spikes;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 1..11 {
            for j in 1..11 {
                assert!(spike_counts[i][j] <= spike_counts[i][j - 1]);
            }
        }
    }
}