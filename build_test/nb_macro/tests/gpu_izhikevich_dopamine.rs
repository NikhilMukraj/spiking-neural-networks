mod izhikevich_dopamine;

#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use crate::izhikevich_dopamine::{
        BoundedNeurotransmitterKinetics, BoundedReceptorKinetics,
        IzhikevichNeuron, DopaGluGABANeurotransmitterType, DopaGluGABAType, GlutamateReceptor,
    };
    use opencl3::{
        context::Context,
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    use rand::Rng;
    use spiking_neural_networks::{
        error::{SpikingNeuralNetworksError, GPUError}, 
        neuron::{
            iterate_and_spike::{IterateAndSpikeGPU, Receptors}, 
            gpu_lattices::LatticeGPU, CellGrid, Lattice, RunLattice
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
}