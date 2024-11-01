#[cfg(test)]
mod tests {
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::neuron::iterate_and_spike::{
        LigandGatedChannel, LigandGatedChannels, IonotropicNeurotransmitterType,
        ApproximateReceptor,
        AMPADefault, NMDADefault, GABAaDefault, GABAbDefault,
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;

    #[test]
    pub fn test_empty_ligand_gates_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        type GridType = Vec<Vec<LigandGatedChannels<ApproximateReceptor>>>;
        let ligand_gates_grid: GridType = vec![];

        let gpu_conversion = LigandGatedChannels::convert_to_gpu(
            &ligand_gates_grid,
            &context,
            &queue,
            0,
            0,
        )?;

        let mut cpu_conversion = ligand_gates_grid.clone();
        LigandGatedChannels::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            0,
            0,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(ligand_gates_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual, 
                    expected,
                );
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_ligand_gates_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut ligand_gates1 = LigandGatedChannels::<ApproximateReceptor>::default();
        ligand_gates1.insert(
            IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default()
        )?;
        ligand_gates1.insert(
            IonotropicNeurotransmitterType::NMDA, LigandGatedChannel::nmda_default()
        )?;
        let mut ligand_gates2 = LigandGatedChannels::default();
        ligand_gates2.insert(
            IonotropicNeurotransmitterType::NMDA, LigandGatedChannel::nmda_default()
        )?;
        let mut ligand_gates3 = LigandGatedChannels::default();
        ligand_gates3.insert(
            IonotropicNeurotransmitterType::GABAa, LigandGatedChannel::gabaa_default()
        )?;
        ligand_gates3.insert(
            IonotropicNeurotransmitterType::GABAb, LigandGatedChannel::gabab_default()
        )?;
        let ligand_gates4 = LigandGatedChannels::default();

        let ligand_gates_grid = vec![
            vec![ligand_gates1, ligand_gates2, ligand_gates3, ligand_gates4]
        ];

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = LigandGatedChannels::convert_to_gpu(
            &ligand_gates_grid,
            &context,
            &queue,
            1,
            4,
        )?;

        let mut cpu_conversion = ligand_gates_grid.clone();
        LigandGatedChannels::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            1,
            4,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(ligand_gates_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual, 
                    expected,
                );
            }
        }

        Ok(())
    }
}
