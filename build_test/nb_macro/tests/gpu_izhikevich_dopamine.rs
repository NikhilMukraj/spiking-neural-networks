mod izhikevich_dopamine;

#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use std::{collections::HashMap, ptr};
    use crate::izhikevich_dopamine::{
        BoundedNeurotransmitterKinetics, BoundedReceptorKinetics, DopaGluGABANeurotransmitterType, DopaGluGABAType, DopamineReceptor, GABAReceptor, GlutamateReceptor, IzhikevichNeuron
    };
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel, memory::{Buffer, CL_MEM_READ_WRITE}, types::{cl_float, CL_NON_BLOCKING}
    };
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use spiking_neural_networks::{
        error::{GPUError, SpikingNeuralNetworksError}, neuron::{
            gpu_lattices::{LatticeGPU, LatticeNetworkGPU}, iterate_and_spike::{
                BufferGPU, IterateAndSpike, IterateAndSpikeGPU, NeurotransmitterConcentrations, NeurotransmitterTypeGPU, Receptors, Timestep
            }, CellGrid, GridVoltageHistory, Lattice, LatticeNetwork, RunLattice, RunNetwork,
        }
    };


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
        for (n, (cpu_cell_grid, gpu_cell_grid)) in cpu_grid_history.iter()
            .zip(gpu_grid_history)
            .enumerate() {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= 3., "{} | error: {}, voltage1: {}, voltage2: {}", 
                        n,
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

    unsafe fn create_and_write_buffer<T>(
        context: &Context,
        queue: &CommandQueue,
        size: usize,
        init_value: T,
    ) -> Result<Buffer<cl_float>, GPUError>
    where
        T: Clone + Into<f32>,
    {
        let mut buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let initial_data = vec![init_value.into(); size];
        let write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &initial_data, &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };
    
        write_event.wait().map_err(|_| GPUError::WaitError)?;
    
        Ok(buffer)
    }

    type FullNeuronType = IzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>;
    type GetGPUAttribute = dyn Fn(&HashMap<String, BufferGPU>, &CommandQueue, &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError>;
    
    fn iterate_neuron(
        input: f32,
        t: (f32, f32, f32),
        s_ds: (f32, f32),
        cpu_get_attribute: &dyn Fn(&FullNeuronType, &mut Vec<f32>),
        gpu_get_attribute: &GetGPUAttribute,
        chemical: bool,
        receptors_on: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron: FullNeuronType = IzhikevichNeuron::default();
        neuron.set_dt(0.1);

        if receptors_on {
            neuron.receptors.insert(
                DopaGluGABANeurotransmitterType::Glutamate,
                DopaGluGABAType::Glutamate(GlutamateReceptor::default())
            ).unwrap();
             neuron.receptors.insert(
                DopaGluGABANeurotransmitterType::GABA,
                DopaGluGABAType::GABA(GABAReceptor::default())
            ).unwrap();
            neuron.receptors.insert(
                DopaGluGABANeurotransmitterType::Dopamine,
                DopaGluGABAType::Dopamine(DopamineReceptor { s_d1: s_ds.0, s_d2: s_ds.1, ..DopamineReceptor::default() })
            ).unwrap();
        }

        let mut cpu_neuron = neuron.clone();

        let mut cpu_tracker = vec![];

        let mut neurotransmitter_input: NeurotransmitterConcentrations<DopaGluGABANeurotransmitterType> = HashMap::new();
        neurotransmitter_input.insert(
            DopaGluGABANeurotransmitterType::Glutamate, t.0
        );
        neurotransmitter_input.insert(
            DopaGluGABANeurotransmitterType::GABA, t.1
        );
        neurotransmitter_input.insert(
            DopaGluGABANeurotransmitterType::Dopamine, t.2
        );

        for _ in 0..iterations {
            if !chemical {
                cpu_neuron.iterate_and_spike(
                    input
                );
            } else {
                cpu_neuron.iterate_with_neurotransmitter_and_spike(
                    input, 
                    &neurotransmitter_input,
                );
            }
            cpu_get_attribute(&cpu_neuron, &mut cpu_tracker);
        }

        let cell_grid = vec![vec![neuron]];

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
        };

        let queue =  match CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
            };

        let iterate_kernel = if !chemical {
            IzhikevichNeuron::<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>::iterate_and_spike_electrical_kernel(&context)?
        } else {
            IzhikevichNeuron::<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>::iterate_and_spike_electrochemical_kernel(&context)?
        };

        let gpu_cell_grid = if !chemical {
            IzhikevichNeuron::convert_to_gpu(&cell_grid, &context, &queue)?
        } else {
            IzhikevichNeuron::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?
        };

        let sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, input)?
        };

        let mut t_sums_buffer = unsafe {
            Buffer::<cl_float>::create(
                &context, 
                CL_MEM_READ_WRITE, 
                DopaGluGABANeurotransmitterType::number_of_types(), 
                ptr::null_mut()
            )
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let t_buffer_write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut t_sums_buffer, CL_NON_BLOCKING, 0, &[t.0, t.1, t.2], &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };
    
        t_buffer_write_event.wait().map_err(|_| GPUError::WaitError)?;
    
        let index_to_position_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, 0.0)?
        };

        let mut gpu_tracker = vec![];

        for _ in 0..iterations {
            let iterate_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                for i in iterate_kernel.argument_names.iter() {
                    if i == "inputs" {
                        kernel_execution.set_arg(&sums_buffer);
                    } else if i == "index_to_position" {
                        kernel_execution.set_arg(&index_to_position_buffer);
                    } else if i == "number_of_types" {
                        kernel_execution.set_arg(&DopaGluGABANeurotransmitterType::number_of_types());
                    } else if i == "t" {
                        kernel_execution.set_arg(&t_sums_buffer); 
                    } else {
                        match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                            BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        };
                    }
                }

                match kernel_execution.set_global_work_size(1)
                    .enqueue_nd_range(&queue) {
                        Ok(value) => value,
                        Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::QueueFailure)),
                    }
            };

            match iterate_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::WaitError)),
            };

            gpu_get_attribute(&gpu_cell_grid, &queue, &mut gpu_tracker)?;
        }

        Ok((cpu_tracker, gpu_tracker))
    }

    fn get_voltage(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        tracker.push(neuron.current_voltage);
    }

    fn get_gpu_voltage(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("current_voltage").unwrap() {
            BufferGPU::Float(buffer) => {
                let mut read_vector = vec![0.];

                let read_event = unsafe {
                    match queue.enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, &mut read_vector, &[]) {
                        Ok(value) => value,
                        Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::BufferReadError)),
                    }
                };
    
                match read_event.wait() {
                    Ok(value) => value,
                    Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::WaitError)),
                };

                gpu_tracker.push(read_vector[0]);
            },
            _ => unreachable!(),
        }

        Ok(())
    }

    #[test]
    pub fn test_single_neuron_electrical() -> Result<(), SpikingNeuralNetworksError> {
        let is = [0., 10., 20., 30., 40., 50.];

        for i in is {
            let (cpu_voltages, gpu_voltages) = iterate_neuron(
                i,
                (0., 0., 0.), 
                (0., 0.),
                &get_voltage, 
                &get_gpu_voltage,
                false,
                false,
            )?;

            for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
                let error = (cpu_voltage - gpu_voltage).abs();
                assert!(error < 3., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
            }
        }

        Ok(())
    }

    // fn get_is_spiking(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
    //     if neuron.is_spiking {
    //         tracker.push(1.);
    //     } else {
    //         tracker.push(0.);
    //     }
    // }

    // fn gpu_get_is_spiking(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
    //     match gpu_cell_grid.get("is_spiking").unwrap() {
    //         BufferGPU::UInt(buffer) => {
    //             let mut read_vector = vec![0];

    //             let read_event = unsafe {
    //                 match queue.enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, &mut read_vector, &[]) {
    //                     Ok(value) => value,
    //                     Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::BufferReadError)),
    //                 }
    //             };
    
    //             match read_event.wait() {
    //                 Ok(value) => value,
    //                 Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::WaitError)),
    //             };

    //             gpu_tracker.push(read_vector[0] as f32);
    //         },
    //         _ => unreachable!(),
    //     }

    //     Ok(())
    // }

    #[test]
    pub fn test_single_neuron_glu() -> Result<(), SpikingNeuralNetworksError> {
        let ts = [0., 0.2, 0.4, 0.6, 0.8, 1.];

        for t in ts {
            let (cpu_voltages, gpu_voltages) = iterate_neuron(
                0.,
                (t, 0., 0.), 
                (0., 0.),
                &get_voltage, 
                &get_gpu_voltage,
                true,
                true,
            )?;

            for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
                let error = (cpu_voltage - gpu_voltage).abs();
                assert!(error < 3., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_single_neuron_glu_d1() -> Result<(), SpikingNeuralNetworksError> {
        let ts = [0., 0.2, 0.4, 0.6, 0.8, 1.];

        for glu in ts.iter() {
            for dopa in ts.iter() {
                let (cpu_voltages, gpu_voltages) = iterate_neuron(
                    0.,
                    (*glu, 0., *dopa), 
                    (1., 0.),
                    &get_voltage, 
                    &get_gpu_voltage,
                    true,
                    true,
                )?;

                for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
                    let error = (cpu_voltage - gpu_voltage).abs();
                    assert!(error < 3., "(glu: {}, dopa: {}) | error: {} ({} - {})", glu, dopa, error, cpu_voltage, gpu_voltage);
                }
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_single_neuron_glu_d2() -> Result<(), SpikingNeuralNetworksError> {
        let ts = [0., 0.2, 0.4, 0.6, 0.8, 1.];

        for glu in ts.iter() {
            for dopa in ts.iter() {
                let (cpu_voltages, gpu_voltages) = iterate_neuron(
                    0.,
                    (*glu, 0., *dopa), 
                    (0., 1.),
                    &get_voltage, 
                    &get_gpu_voltage,
                    true,
                    true,
                )?;

                for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
                    let error = (cpu_voltage - gpu_voltage).abs();
                    assert!(error < 3., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::needless_range_loop)]
    fn generate_weights(seed: u64, rows: usize, cols: usize) -> Box<dyn Fn((usize, usize), (usize, usize)) -> f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut weights = vec![vec![0.; rows]; cols];
        for x in 0..rows {
            for y in 0..cols {
                if x != y {
                    weights[x][y] = rng.gen_range(0.0..1.0);
                }
            }
        }

        Box::new(move |x: (usize, usize), _| weights[x.0][x.1])
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::needless_range_loop)]
    fn generate_voltages(seed: u64, rows: usize, cols: usize, min: f32, max: f32) -> Box<dyn Fn((usize, usize), &mut IzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut voltages = vec![vec![0.; rows]; cols];
        for x in 0..rows {
            for y in 0..cols {
                voltages[x][y] = rng.gen_range(min..max);
            }
        }

        Box::new(move |x: (usize, usize), neuron: &mut IzhikevichNeuron<_, _>| {
            neuron.current_voltage = voltages[x.0][x.1];
        })
    }

    fn test_isolated_lattices_accuracy(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let mut base_neuron = IzhikevichNeuron::default_impl();
        base_neuron.synaptic_neurotransmitters.insert(
            DopaGluGABANeurotransmitterType::Glutamate, 
            BoundedNeurotransmitterKinetics::default()
        );
        base_neuron.receptors.insert(
            DopaGluGABANeurotransmitterType::Glutamate, 
            DopaGluGABAType::Glutamate(GlutamateReceptor::default()),
        )?;
    
        let mut lattice1: Lattice<IzhikevichNeuron<_, _>, _, _, _, DopaGluGABANeurotransmitterType> = Lattice::default();
        lattice1.populate(
            &base_neuron, 
            2, 
            2, 
        ).unwrap();
        lattice1.apply_given_position(generate_voltages(2, 2, 2, -55., 20.));
        lattice1.connect(&(|x, y| x != y), Some(&generate_weights(2, 2, 2)));
        lattice1.update_grid_history = true;

        let mut lattice2: Lattice<IzhikevichNeuron<_, _>, _, _, _, DopaGluGABANeurotransmitterType> = Lattice::default();

        lattice2.set_id(1);
        lattice2.populate(
            &base_neuron, 
            3, 
            3, 
        ).unwrap();
        lattice2.apply_given_position(generate_voltages(3, 3, 3, -55., 20.));
        lattice2.connect(&(|x, y| x != y), Some(&generate_weights(3, 3, 3)));
        lattice2.update_grid_history = true;

        let mut network: LatticeNetwork<IzhikevichNeuron<_, _>, _, GridVoltageHistory, _, _, _, _, DopaGluGABANeurotransmitterType> = LatticeNetwork::default_impl();

        network.parallel = true;
        network.add_lattice(lattice1).unwrap();
        network.add_lattice(lattice2).unwrap();

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;
        
        network.run_lattices(1000)?;
        gpu_network.run_lattices(1000)?;

        for i in network.get_all_ids() {
            let cpu_grid_history = &network.get_lattice(&i).unwrap().grid_history.history;
            let gpu_grid_history = &gpu_network.get_lattice(&i).unwrap().grid_history.history;
    
            check_entire_history(cpu_grid_history, gpu_grid_history);
        }

        Ok(())
    }

    #[test]
    pub fn test_isolated_network_electrical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        test_isolated_lattices_accuracy(true, false)?;

        Ok(())
    }

    #[test]
    pub fn test_isolated_network_chemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        test_isolated_lattices_accuracy(false, true)?;

        Ok(())
    }

    #[test]
    pub fn test_isolated_network_electrochemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        test_isolated_lattices_accuracy(true, true)?;

        Ok(())
    }
}
