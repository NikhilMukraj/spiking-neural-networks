#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}};
    use spiking_neural_networks::error::SpikingNeuralNetworksError;


    neuron_builder!(r#"
        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1,
            on_iteration:
                current = g * (v - e)
        [end]

        [neuron]
            type: LIF
            ion_channels: l = TestLeak
            vars: v_reset = -75, v_th = -55
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                l.update_current(v)
                dv/dt = l.current + i
        [end]
    "#);

    type GridType = Vec<Vec<LIF<ApproximateNeurotransmitter, ApproximateReceptor>>>;

    #[test]
    fn test_neuron_conversion_empty() -> Result<(), SpikingNeuralNetworksError> {
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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = LIF::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_to_cpu(
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
    fn test_neuron_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![
            vec![LIF::default(), LIF { current_voltage: -100., l: TestLeak { e: -80., ..Default::default() }, ..Default::default() }],
            vec![LIF { v_th: -40., l: TestLeak { g: 2., ..Default::default() }, ..Default::default() }, LIF::default()],
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

        let mut cpu_conversion: GridType = vec![
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
        ];

        let gpu_conversion = LIF::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (cpu_row, ref_row) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (i, i_ref) in cpu_row.iter().zip(ref_row.iter()) {
                assert_eq!(i.current_voltage, i_ref.current_voltage);
                assert_eq!(i.v_th, i_ref.v_th);
                assert_eq!(i.l, i_ref.l);
            }
        }

        Ok(())
    }   

    #[test]
    fn test_neuron_conversion_non_square() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![
            vec![LIF::default(), LIF { current_voltage: -100., l: TestLeak { e: -80., ..Default::default() }, ..Default::default() }],
            vec![LIF { v_th: -40., l: TestLeak { g: 2., ..Default::default() }, ..Default::default() }, LIF::default()],
            vec![LIF::default(), LIF::default()],
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

        let mut cpu_conversion: GridType = vec![
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
        ];

        let gpu_conversion = LIF::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (cpu_row, ref_row) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (i, i_ref) in cpu_row.iter().zip(ref_row.iter()) {
                assert_eq!(i.current_voltage, i_ref.current_voltage);
                assert_eq!(i.v_th, i_ref.v_th);
                assert_eq!(i.l, i_ref.l);
            }
        }

        Ok(())
    }   

    #[test]
    fn test_neuron_conversion_electrochemical_empty() -> Result<(), SpikingNeuralNetworksError> {
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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = LIF::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_electrochemical_to_cpu(
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
    fn test_neuron_conversion_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![
            vec![LIF::default(), LIF { current_voltage: -100., l: TestLeak { e: -80., ..Default::default() }, ..Default::default() }],
            vec![LIF { v_th: -40., l: TestLeak { g: 2., ..Default::default() }, ..Default::default() }, LIF::default()],
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

        let mut cpu_conversion: GridType = vec![
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
        ];

        let gpu_conversion = LIF::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (cpu_row, ref_row) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (i, i_ref) in cpu_row.iter().zip(ref_row.iter()) {
                assert_eq!(i.current_voltage, i_ref.current_voltage);
                assert_eq!(i.v_th, i_ref.v_th);
                assert_eq!(i.l, i_ref.l);
            }
        }

        Ok(())
    }   

    #[test]
    fn test_neuron_conversion_electrochemical_non_square() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![
            vec![LIF::default(), LIF { current_voltage: -100., l: TestLeak { e: -80., ..Default::default() }, ..Default::default() }],
            vec![LIF { v_th: -40., l: TestLeak { g: 2., ..Default::default() }, ..Default::default() }, LIF::default()],
            vec![LIF::default(), LIF::default()],
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

        let mut cpu_conversion: GridType = vec![
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
            vec![LIF::default(), LIF::default()],
        ];

        let gpu_conversion = LIF::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        LIF::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (cpu_row, ref_row) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (i, i_ref) in cpu_row.iter().zip(ref_row.iter()) {
                assert_eq!(i.current_voltage, i_ref.current_voltage);
                assert_eq!(i.v_th, i_ref.v_th);
                assert_eq!(i.l, i_ref.l);
            }
        }

        Ok(())
    }   
}