#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use rand::Rng;
    use opencl3::{
        command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;

    
    neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
    [end]
    "#);

    type GridType = Vec<Vec<BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor>>>;

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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = BasicIntegrateAndFire::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        // let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
        //     &cell_grid,
        //     &context,
        //     &queue,
        // )?;

        // BasicIntegrateAndFire::convert_electrochemical_to_cpu(
        //     &mut cpu_conversion,
        //     &gpu_conversion,
        //     0,
        //     0,
        //     &queue,
        // )?;

        // assert_eq!(cpu_conversion.len(), 0);

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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = BasicIntegrateAndFire::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        // let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
        //     &cell_grid,
        //     &context,
        //     &queue,
        // )?;

        // BasicIntegrateAndFire::convert_electrochemical_to_cpu(
        //     &mut cpu_conversion,
        //     &gpu_conversion,
        //     0,
        //     0,
        //     &queue,
        // )?;

        // assert_eq!(cpu_conversion.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_neuron_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![BasicIntegrateAndFire::default(), BasicIntegrateAndFire::default()],
            vec![BasicIntegrateAndFire::default(), BasicIntegrateAndFire::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
                i.is_spiking = rand::thread_rng().gen::<bool>();
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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = BasicIntegrateAndFire::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_to_cpu(
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
            }
        }

        Ok(())
    }

    // #[test]
    // pub fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
    //     let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
    //         .expect("Could not get GPU devices")
    //         .first()
    //         .expect("No GPU found");
    //     let device = Device::new(device_id);

    //     let context = Context::from_device(&device).expect("Context::from_device failed");

    //     let electrical_kernel = BasicIntegrateAndFire::iterate_and_spike_electrical_kernel(&context);

    //     Ok(())
    // }
}
