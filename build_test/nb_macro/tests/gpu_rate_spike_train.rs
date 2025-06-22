#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::CL_QUEUE_PROFILING_ENABLE, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel};
    use rand::{seq::SliceRandom, Rng};
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, neuron::{iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType, IonotropicReceptorNeurotransmitterType}, spike_train::DeltaDiracRefractoriness}};


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

    type SpikeTrainType = RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;
    type GridType = Vec<Vec<SpikeTrainType>>;

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

    type GetGPUAttribute = dyn Fn(&HashMap<String, BufferGPU>, &CommandQueue, &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError>;

    fn iterate_neuron(
        rate: f32,
        cpu_get_attribute: &dyn Fn(&SpikeTrainType, &mut Vec<f32>),
        gpu_get_attribute: &GetGPUAttribute,
        electrical: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        let iterations = 1000;

        let mut neuron = SpikeTrainType { rate, ..Default::default() };

        if !electrical {
            neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        }

        let mut cpu_neuron = neuron.clone();

        let mut cpu_tracker = vec![];

        for _ in 0..iterations {
            let _ = cpu_neuron.iterate();

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

        let iterate_kernel = if electrical {
            SpikeTrainType::spike_train_electrical_kernel(&context)?
        } else {
            SpikeTrainType::spike_train_electrochemical_kernel(&context)?
        };

        let gpu_cell_grid = SpikeTrainType::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?;

        let index_to_position_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, 0.0)?
        };

        let mut gpu_tracker = vec![];

        for _ in 0..iterations {
            let iterate_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                let mut counter = 0;

                for i in iterate_kernel.argument_names.iter() {
                    if i == "number_of_types" {
                        kernel_execution.set_arg(&IonotropicReceptorNeurotransmitterType::number_of_types());
                    } else if i == "index_to_position" {
                        kernel_execution.set_arg(&index_to_position_buffer);
                    } else if i == "neuro_flags" {
                        match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            _ => unreachable!("Could not retrieve neurotransmitter flags"),
                        };
                    } else {
                        match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                            BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        };
                    }
                    counter += 1;
                }

                assert_eq!(
                    counter,
                    kernel_execution.num_args, 
                    "counter: {} != num_args: {}",
                    counter,
                    kernel_execution.num_args,
                );

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

    fn get_cpu_voltage(neuron: &SpikeTrainType, tracker: &mut Vec<f32>) {
        tracker.push(neuron.current_voltage);
    }

    fn get_gpu_is_spiking(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("is_spiking").unwrap() {
            BufferGPU::UInt(buffer) => {
                let mut read_vector: Vec<u32> = vec![0];

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

    fn get_cpu_is_spiking(neuron: &SpikeTrainType, tracker: &mut Vec<f32>) {
        if neuron.is_spiking {
            tracker.push(1.);
        } else {
            tracker.push(0.);
        }
    }

    #[test]
    fn test_voltages() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(
            100., 
            &get_cpu_voltage, 
            &get_gpu_voltage, 
            true,
        )?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages.iter()) {
            assert_eq!(cpu_voltage, gpu_voltage);
        }

        Ok(())
    }

    #[test]
    fn test_is_spikings() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_spikes, gpu_spikes) = iterate_neuron(
            100., 
            &get_cpu_is_spiking, 
            &get_gpu_is_spiking, 
            true,
        )?;

        for (cpu_spike, gpu_spike) in cpu_spikes.iter().zip(gpu_spikes.iter()) {
            assert_eq!(cpu_spike, gpu_spike);
        }

        Ok(())
    }

    #[test]
    fn test_voltages_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(
            100., 
            &get_cpu_voltage, 
            &get_gpu_voltage, 
            false,
        )?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages.iter()) {
            assert_eq!(cpu_voltage, gpu_voltage);
        }

        Ok(())
    }

    #[test]
    fn test_is_spikings_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_spikes, gpu_spikes) = iterate_neuron(
            100., 
            &get_cpu_is_spiking, 
            &get_gpu_is_spiking, 
            false,
        )?;

        for (cpu_spike, gpu_spike) in cpu_spikes.iter().zip(gpu_spikes.iter()) {
            assert_eq!(cpu_spike, gpu_spike);
        }

        Ok(())
    }

    fn get_cpu_ampa_neurotransmitter(neuron: &SpikeTrainType, tracker: &mut Vec<f32>) {
        let ampa_neurotransmitter = neuron.synaptic_neurotransmitters.get(&IonotropicNeurotransmitterType::AMPA)
            .expect("Could not get neurotransmitter")
            .t;

        tracker.push(ampa_neurotransmitter);
    }

    fn get_gpu_ampa_neurotransmitter(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("neurotransmitters$t").unwrap() {
            BufferGPU::Float(buffer) => {
                let mut read_vector = vec![0., 0., 0.,];

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
    fn test_neurotransmitters() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_ts, gpu_ts) = iterate_neuron(
            100., 
            &get_cpu_ampa_neurotransmitter, 
            &get_gpu_ampa_neurotransmitter, 
            false,
        )?;

        for (cpu_t, gpu_t) in cpu_ts.iter().zip(gpu_ts.iter()) {
            assert_eq!(cpu_t, gpu_t);
        }

        Ok(())
    }
}
