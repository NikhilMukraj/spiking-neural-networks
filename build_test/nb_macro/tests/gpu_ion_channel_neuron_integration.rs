#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::CL_QUEUE_PROFILING_ENABLE, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel};
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, neuron::iterate_and_spike::{DefaultReceptorsType, XReceptor}};


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

    #[test]
    fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
            let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        assert!(LIF::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context).is_ok());

        Ok(())
    }

    // #[test]
    // fn test_electrochemical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {}

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

    type FullNeuronType = LIF<ApproximateNeurotransmitter, ApproximateReceptor>;
    type GetGPUAttribute = dyn Fn(&HashMap<String, BufferGPU>, &CommandQueue, &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError>;

    fn iterate_neuron(
        input: f32,
        t: f32,
        cpu_get_attribute: &dyn Fn(&FullNeuronType, &mut Vec<f32>),
        gpu_get_attribute: &GetGPUAttribute,
        chemical: bool,
        receptors_on: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron: FullNeuronType = LIF::default();
        neuron.set_dt(0.1);

        neuron.synaptic_neurotransmitters.insert(
            DefaultReceptorsNeurotransmitterType::X, 
            ApproximateNeurotransmitter::default(),
        );
        if receptors_on {
            neuron.receptors.insert(
                DefaultReceptorsNeurotransmitterType::X,
                DefaultReceptorsType::X(XReceptor::default())
            ).unwrap();
        }

        let mut cpu_neuron = neuron.clone();

        let mut cpu_tracker = vec![];

        let mut neurotransmitter_input: NeurotransmitterConcentrations<DefaultReceptorsNeurotransmitterType> = HashMap::new();
        neurotransmitter_input.insert(DefaultReceptorsNeurotransmitterType::X, t);

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
            LIF::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context)?
        } else {
            LIF::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrochemical_kernel(&context)?
        };

        let gpu_cell_grid = if !chemical {
            LIF::convert_to_gpu(&cell_grid, &context, &queue)?
        } else {
            LIF::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?
        };

        let sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, input)?
        };

        let t_sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, t)?
        };
    
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
                        kernel_execution.set_arg(&DefaultReceptorsNeurotransmitterType::number_of_types());
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
    fn test_gpu_voltages_electrical() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(
            5.,
            0.,
            &get_voltage, 
            &get_gpu_voltage,
            false,
            false,
        )?;

        for (i, j) in cpu_voltages.iter().zip(gpu_voltages.iter()) {
            if !i.is_finite() || !j.is_finite() {
                continue;
            }
            assert!((i - j).abs() < 2., "({} - {}).abs() < 2.", i, j);
        }
       
        Ok(())
    }

    fn get_is_spiking(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        if neuron.is_spiking {
            tracker.push(1.);
        } else {
            tracker.push(0.);
        }
    }

    fn gpu_get_is_spiking(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("is_spiking").unwrap() {
            BufferGPU::UInt(buffer) => {
                let mut read_vector = vec![0];

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

                gpu_tracker.push(read_vector[0] as f32);
            },
            _ => unreachable!(),
        }

        Ok(())
    }

    #[test]
    pub fn test_gpu_is_spiking_electrical() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_spikings, gpu_spikings) = iterate_neuron(
            5.,
            0.,
            &get_is_spiking, 
            &gpu_get_is_spiking,
            false,
            false,
        )?;

        let cpu_sum = cpu_spikings.iter().sum::<f32>();
        let gpu_sum = gpu_spikings.iter().sum::<f32>();
        let error = (cpu_sum - gpu_sum).abs();

        assert!(error < 2., "error: {} ({} - {})", error, cpu_sum, gpu_sum);

        Ok(())
    }

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
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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