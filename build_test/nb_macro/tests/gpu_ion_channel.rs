#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::Kernel, program::Program};
    use spiking_neural_networks::error::SpikingNeuralNetworksError; 

    
    neuron_builder!(r#"
        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1,
            on_iteration:
                current = g * (v - e)
        [end]
    "#);

    #[test]
    fn test_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let program_source = TestLeak::get_update_function().1;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_name = String::from("update_TestLeak_ion_channel");

        let program = match Program::create_and_build_from_source(&context, &program_source, "") {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::ProgramCompileFailure)),
        };

        match Kernel::create(&program, &kernel_name) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::from(GPUError::KernelCompileFailure)),
        }
    }

    #[test]
    fn test_get_all_attrs() {
        let vec_of_attrs = vec![
            (String::from("ion_channel$current"), AvailableBufferType::Float),
            (String::from("ion_channel$e"), AvailableBufferType::Float),
            (String::from("ion_channel$g"), AvailableBufferType::Float),
        ];

        assert_eq!(
            vec_of_attrs,
            TestLeak::get_attribute_names_as_vector(),
        );

        assert_eq!(
            HashSet::from_iter(vec_of_attrs),
            TestLeak::get_all_attributes(),
        );
    }

    #[test]
    fn test_get_and_set_attr() {
        let mut ion_channel = TestLeak::default();

        assert_eq!(
            Some(BufferType::Float(1.)),
            ion_channel.get_attribute("ion_channel$g"),
        );
        assert_eq!(
            Some(BufferType::Float(0.)),
            ion_channel.get_attribute("ion_channel$e"),
        );
        assert_eq!(
            Some(BufferType::Float(0.)),
            ion_channel.get_attribute("ion_channel$current"),
        );
        assert_eq!(
            None,
            ion_channel.get_attribute("ion_channel$a"),
        );
        assert_eq!(
            None,
            ion_channel.get_attribute("current"),
        );
        assert_eq!(
            None,
            ion_channel.get_attribute("e"),
        );
        assert_eq!(
            None,
            ion_channel.get_attribute("g"),
        );

        ion_channel.current = 100.;
        ion_channel.g = 2.;

        assert_eq!(
            Some(BufferType::Float(2.)),
            ion_channel.get_attribute("ion_channel$g"),
        );
        assert_eq!(
            Some(BufferType::Float(100.)),
            ion_channel.get_attribute("ion_channel$current"),
        );

        assert!(ion_channel.set_attribute("ion_channel$current", BufferType::UInt(0)).is_err());

        assert_eq!(
            Some(BufferType::Float(100.)),
            ion_channel.get_attribute("ion_channel$current"),
        );

        assert!(ion_channel.set_attribute("ion_channel$current", BufferType::OptionalUInt(0)).is_err());

        assert_eq!(
            Some(BufferType::Float(100.)),
            ion_channel.get_attribute("ion_channel$current"),
        );

        assert!(ion_channel.set_attribute("ion_channel$current", BufferType::Float(0.)).is_ok());

        assert_eq!(
            Some(BufferType::Float(0.)),
            ion_channel.get_attribute("ion_channel$current"),
        );

        assert!(ion_channel.set_attribute("ion_channel$c", BufferType::Float(0.)).is_err());
    }

    #[test]
    fn test_ion_channels_empty_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let ion_channel_grid: Vec<Vec<TestLeak>> = vec![];

        let gpu_conversion = TestLeak::convert_to_gpu(
            &ion_channel_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion = ion_channel_grid.clone();
        TestLeak::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(ion_channel_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual, 
                    expected,
                );
            }
        }

        assert_eq!(cpu_conversion.len(), ion_channel_grid.len());

        Ok(())
    }

    #[test]
    fn test_ion_channel_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let ion_channels = vec![
            vec![TestLeak::default(), TestLeak { current: 100., ..TestLeak::default() }],
            vec![TestLeak { g: -1., ..TestLeak::default() }, TestLeak { e: -80., ..TestLeak::default() }],
        ];

        let gpu_conversion = TestLeak::convert_to_gpu(&ion_channels, &context, &queue)?;

        let mut cpu_conversion = vec![
            vec![TestLeak::default(), TestLeak::default()],
            vec![TestLeak::default(), TestLeak::default()],
        ];

        TestLeak::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(ion_channels.iter()) {
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
    fn test_ion_channel_non_square_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let ion_channels = vec![
            vec![TestLeak::default(), TestLeak { current: 100., ..TestLeak::default() }],
            vec![TestLeak { g: -1., ..TestLeak::default() }, TestLeak { e: -80., ..TestLeak::default() }],
            vec![TestLeak::default(), TestLeak::default()],
        ];

        let gpu_conversion = TestLeak::convert_to_gpu(&ion_channels, &context, &queue)?;

        let mut cpu_conversion = vec![
            vec![TestLeak::default(), TestLeak::default()],
            vec![TestLeak::default(), TestLeak::default()],
            vec![TestLeak::default(), TestLeak::default()],
        ];

        TestLeak::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(ion_channels.iter()) {
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