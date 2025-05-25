#[cfg(test)]
mod tests {
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::neuron::iterate_and_spike::{
        AMPAReceptor, ApproximateReceptor, GABAReceptor, Ionotropic, IonotropicNeurotransmitterType, 
        IonotropicType, NMDAReceptor, Receptors, ReceptorsGPU
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
                0,
            )
            .expect("CommandQueue::create_default failed");

        type GridType = Vec<Vec<Ionotropic<ApproximateReceptor>>>;
        let ligand_gates_grid: GridType = vec![];

        let gpu_conversion = Ionotropic::convert_to_gpu(
            &ligand_gates_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion = ligand_gates_grid.clone();
        Ionotropic::convert_to_cpu(
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
        let mut ligand_gates1 = Ionotropic::<ApproximateReceptor>::default();
        ligand_gates1.insert(
            IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default())
        )?;
        ligand_gates1.insert(
            IonotropicNeurotransmitterType::NMDA, 
            IonotropicType::NMDA(NMDAReceptor {
                r: ApproximateReceptor { r: 1. },
                ..NMDAReceptor::default()
            })
        )?;
        let mut ligand_gates2 = Ionotropic::default();
        ligand_gates2.insert(
            IonotropicNeurotransmitterType::NMDA, IonotropicType::NMDA(NMDAReceptor::default())
        )?;
        let mut ligand_gates3 = Ionotropic::default();
        ligand_gates3.insert(
            IonotropicNeurotransmitterType::GABA, IonotropicType::GABA(GABAReceptor::default())
        )?;
        ligand_gates3.insert(
            IonotropicNeurotransmitterType::GABA, 
            IonotropicType::GABA(GABAReceptor {
                current: 5.,
                ..GABAReceptor::default()
            })
        )?;
        let ligand_gates4 = Ionotropic::default();

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
                0,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = Ionotropic::convert_to_gpu(
            &ligand_gates_grid,
            &context,
            &queue,
        )?;

        // let mut cpu_conversion = ligand_gates_grid.clone();
        let mut cpu_conversion = vec![
            vec![
                Ionotropic::default(), Ionotropic::default(),
                Ionotropic::default(), Ionotropic::default(),
            ]
        ];
        Ionotropic::convert_to_cpu(
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
