#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::CL_QUEUE_PROFILING_ENABLE, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}};
    use rand::{seq::SliceRandom, Rng};
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, neuron::{iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType}, spike_train::DeltaDiracRefractoriness}};


    neuron_builder!(
        "[spike_train]
            type: RateSpikeTrain
            vars: step = 0., rate = 0.
            on_iteration:
                step += dt
                [if] rate != 0. && step >= rate [then]
                    step = 0
                    current_voltage = v_th
                    is_spiking = true
                [else]
                    current_voltage = v_resting
                    is_spiking = false
                [end]
        [end]"
    );

    #[test]
    fn test_electrical_kernel_compiles() {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_function = RateSpikeTrain::<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>::
            spike_train_electrical_kernel(&context);

        assert!(kernel_function.is_ok());
    }

    #[test]
    fn test_electrochemical_kernel_compiles() {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_function = RateSpikeTrain::<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>::
            spike_train_electrochemical_kernel(&context);

        assert!(kernel_function.is_ok());
    }

    type GridType = Vec<Vec<RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>>>;

    #[test]
    pub fn test_empty_grid_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![];

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

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = RateSpikeTrain::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        let gpu_conversion = RateSpikeTrain::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_grid_of_empty_grids_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![vec![], vec![]];

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

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = RateSpikeTrain::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        let gpu_conversion = RateSpikeTrain::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_spike_train_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
                i.is_spiking = rand::thread_rng().gen::<bool>();
                i.step = rand::thread_rng().gen_range(0.0..100.);
                i.neural_refractoriness.k = rand::thread_rng().gen_range(0.0..100.);
            }
        }

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

        let gpu_conversion = RateSpikeTrain::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.is_spiking, 
                    expected.is_spiking,
                );
                assert_eq!(
                    actual.step, 
                    expected.step,
                );
                assert_eq!(
                    actual.neural_refractoriness, 
                    expected.neural_refractoriness,
                );
            }
        }

        Ok(())
    }
    
    #[test]
    pub fn test_spike_train_conversion_non_square() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
                i.is_spiking = rand::thread_rng().gen::<bool>();
                i.step = rand::thread_rng().gen_range(0.0..100.);
                i.neural_refractoriness.k = rand::thread_rng().gen_range(0.0..100.);
            }
        }

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

        let gpu_conversion = RateSpikeTrain::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.is_spiking, 
                    expected.is_spiking,
                );
                assert_eq!(
                    actual.step, 
                    expected.step,
                );
                assert_eq!(
                    actual.neural_refractoriness, 
                    expected.neural_refractoriness,
                );
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_electrochemical_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
                i.is_spiking = rand::thread_rng().gen::<bool>();
                i.step = rand::thread_rng().gen_range(0.0..100.);
                i.neural_refractoriness.k = rand::thread_rng().gen_range(0.0..100.);
                if rand::thread_rng().gen_range(0.0..1.) < 0.5 {
                    i.synaptic_neurotransmitters.insert(
                        *IonotropicNeurotransmitterType::get_all_types().iter().cloned().collect::<Vec<_>>().choose(&mut rand::thread_rng()).unwrap(),
                        ApproximateNeurotransmitter { t: rand::thread_rng().gen_range(0.0..1.), ..Default::default() }
                    )
                }
            }
        }

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

        let gpu_conversion = RateSpikeTrain::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.is_spiking, 
                    expected.is_spiking,
                );
                assert_eq!(
                    actual.step, 
                    expected.step,
                );
                assert_eq!(
                    actual.neural_refractoriness, 
                    expected.neural_refractoriness,
                );
                assert_eq!(
                    actual.synaptic_neurotransmitters, 
                    expected.synaptic_neurotransmitters,
                );
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_electrochemical_conversion_non_square() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
            vec![RateSpikeTrain::default(), RateSpikeTrain::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
                i.is_spiking = rand::thread_rng().gen::<bool>();
                i.step = rand::thread_rng().gen_range(0.0..100.);
                i.neural_refractoriness.k = rand::thread_rng().gen_range(0.0..100.);
                if rand::thread_rng().gen_range(0.0..1.) < 0.5 {
                    i.synaptic_neurotransmitters.insert(
                        *IonotropicNeurotransmitterType::get_all_types().iter().cloned().collect::<Vec<_>>().choose(&mut rand::thread_rng()).unwrap(),
                        ApproximateNeurotransmitter { t: rand::thread_rng().gen_range(0.0..1.), ..Default::default() }
                    )
                }
            }
        }

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

        let gpu_conversion = RateSpikeTrain::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        RateSpikeTrain::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.is_spiking, 
                    expected.is_spiking,
                );
                assert_eq!(
                    actual.step, 
                    expected.step,
                );
                assert_eq!(
                    actual.neural_refractoriness, 
                    expected.neural_refractoriness,
                );
                assert_eq!(
                    actual.synaptic_neurotransmitters, 
                    expected.synaptic_neurotransmitters,
                );
            }
        }

        Ok(())
    }

    // check kernel works as expected
}