#[cfg(test)]
mod tests {
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::neuron::iterate_and_spike::{
        Neurotransmitters, IonotropicReceptorNeurotransmitterType,
        ApproximateNeurotransmitter,
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;

    
    type GridType = Vec<Vec<Neurotransmitters<IonotropicReceptorNeurotransmitterType, ApproximateNeurotransmitter>>>;

    #[test]
    pub fn test_empty_neurotransmitter_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let neurotransmitters_grid: GridType = vec![];

        let gpu_conversion = Neurotransmitters::convert_to_gpu(
            &neurotransmitters_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion = neurotransmitters_grid.clone();
        Neurotransmitters::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            0,
            0,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(neurotransmitters_grid.iter()) {
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
    pub fn test_neurotransmitter_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut neurotransmitters1 = Neurotransmitters::default();
        neurotransmitters1.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, ApproximateNeurotransmitter {
                t_max: 0.5,
                ..ApproximateNeurotransmitter::default()
            }
        );
        neurotransmitters1.insert(
            IonotropicReceptorNeurotransmitterType::NMDA, ApproximateNeurotransmitter::default()
        );
        let mut neurotransmitters2 = Neurotransmitters::default();
        neurotransmitters2.insert(
            IonotropicReceptorNeurotransmitterType::NMDA, ApproximateNeurotransmitter {
                clearance_constant: 0.02,
                ..ApproximateNeurotransmitter::default()
            }
        );
        let mut neurotransmitters3 = Neurotransmitters::default();
        neurotransmitters3.insert(
            IonotropicReceptorNeurotransmitterType::GABAa, ApproximateNeurotransmitter::default()
        );
        neurotransmitters3.insert(
            IonotropicReceptorNeurotransmitterType::GABAb, ApproximateNeurotransmitter::default()
        );
        let neurotransmitters4 = Neurotransmitters::default();

        let neurotransmitters_grid = vec![
            vec![neurotransmitters1, neurotransmitters2, neurotransmitters3, neurotransmitters4]
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

        let gpu_conversion = Neurotransmitters::convert_to_gpu(
            &neurotransmitters_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion: GridType = vec![
            vec![
                Neurotransmitters::default(), Neurotransmitters::default(), 
                Neurotransmitters::default(), Neurotransmitters::default()
            ]
        ];
        Neurotransmitters::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            1,
            4,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(neurotransmitters_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual, 
                    expected,
                );
            }
        }

        Ok(())
    }

    // test 2x3 grid
}
