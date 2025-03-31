#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use rand::Rng;
    use opencl3::{
        command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel,
    };
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, neuron::iterate_and_spike::{DefaultReceptorsType, NeurotransmitterTypeGPU, XReceptor}};

    
    neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = -(v - e) + i
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

        let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_electrochemical_to_cpu(
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

        let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_electrochemical_to_cpu(
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

    #[test]
    pub fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        }
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

    type FullNeuronType = BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor>;
    type GetGPUAttribute = dyn Fn(&HashMap<String, BufferGPU>, &CommandQueue, &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError>;

    fn iterate_neuron(
        input: f32,
        cpu_get_attribute: &dyn Fn(&FullNeuronType, &mut Vec<f32>),
        gpu_get_attribute: &GetGPUAttribute,
        chemical: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron: FullNeuronType = BasicIntegrateAndFire::default();
        neuron.set_dt(0.1);

        neuron.synaptic_neurotransmitters.insert(
            DefaultReceptorsNeurotransmitterType::X, 
            ApproximateNeurotransmitter::default(),
        );

        let mut cpu_neuron = neuron.clone();

        let mut cpu_tracker = vec![];

        for _ in 0..iterations {
            cpu_neuron.iterate_and_spike(
                input
            );
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
                CL_QUEUE_SIZE,
            ) {
                Ok(value) => value,
                Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
            };

        let iterate_kernel = if !chemical {
            BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context)?
        } else {
            BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrochemical_kernel(&context)?
        };

        let gpu_cell_grid = if !chemical {
            BasicIntegrateAndFire::convert_to_gpu(&cell_grid, &context, &queue)?
        } else {
            BasicIntegrateAndFire::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?
        };

        let sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, input)?
        };

        let t_sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, input)?
        };
    
        let index_to_position_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, 0.0)?
        };

        let receptor_flags_buffer = unsafe {
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
                    } else if i == "receptors$flags" {
                        kernel_execution.set_arg(&receptor_flags_buffer);
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
    pub fn test_single_neuron_is_spiking() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_spikings, gpu_spikings) = iterate_neuron(
            5.,
            &get_is_spiking, 
            &gpu_get_is_spiking,
            false,
        )?;

        let cpu_sum = cpu_spikings.iter().sum::<f32>();
        let gpu_sum = gpu_spikings.iter().sum::<f32>();
        let error = (cpu_sum - gpu_sum).abs();

        assert!(error < 2., "error: {} ({} - {})", error, cpu_sum, gpu_sum);

        Ok(())
    }

    #[test]
    fn test_single_neuron_voltage() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(
            5.,
            &get_voltage, 
            &get_gpu_voltage,
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

    #[test]
    fn test_electrochemical_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let cell_grid: GridType = vec![vec![
            BasicIntegrateAndFire { current_voltage: -100., ..BasicIntegrateAndFire::default() }
        ]];

        let mut cpu_conversion: GridType = vec![vec![BasicIntegrateAndFire::default()]];

        let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            1,
            1,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 1);
        assert_eq!(cpu_conversion[0].len(), 1);
        assert!(cpu_conversion[0][0].receptors.is_empty());
        assert_eq!(cpu_conversion[0][0].current_voltage, -100.);

        let mut cell_grid: GridType = vec![vec![
            BasicIntegrateAndFire { current_voltage: -100., ..BasicIntegrateAndFire::default() }
        ]];
        cell_grid[0][0].receptors
            .insert(
                DefaultReceptorsNeurotransmitterType::X, 
                DefaultReceptorsType::X(XReceptor{ g: 100., ..XReceptor::default() })
            )?;
        let neuro = ApproximateNeurotransmitter { t_max: 0.5, t: 0.1, clearance_constant: 0.002 };
        cell_grid[0][0].synaptic_neurotransmitters
            .insert(
                DefaultReceptorsNeurotransmitterType::X,
                neuro
            );

        let mut cpu_conversion: GridType = vec![vec![BasicIntegrateAndFire::default()]];

        let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            1,
            1,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 1);
        assert_eq!(cpu_conversion[0].len(), 1);
        assert_eq!(cpu_conversion[0][0].receptors.len(), 1);
        match cpu_conversion[0][0].receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
            Some(DefaultReceptorsType::X(receptor)) => assert_eq!(receptor.g, 100.),
            _ => panic!("Unexpected conversion") 
        }
        assert_eq!(cpu_conversion[0][0].current_voltage, -100.);
        assert_eq!(
            *cpu_conversion[0][0].synaptic_neurotransmitters.get(&DefaultReceptorsNeurotransmitterType::X).unwrap(), 
            neuro
        );

        Ok(())
    }

    #[test]
    fn test_electrochemical_conversion_non_square() -> Result<(), SpikingNeuralNetworksError> {
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

        let mut cell_grid: GridType = vec![vec![
            BasicIntegrateAndFire { current_voltage: -100., ..BasicIntegrateAndFire::default() },
            BasicIntegrateAndFire { gap_conductance: 100., ..BasicIntegrateAndFire::default() },
        ]];
        cell_grid[0][1].receptors
            .insert(
                DefaultReceptorsNeurotransmitterType::X, 
                DefaultReceptorsType::X(XReceptor{ g: 100., ..XReceptor::default() })
            )?;
        let neuro = ApproximateNeurotransmitter { t_max: 0.5, t: 0.1, clearance_constant: 0.002 };
        cell_grid[0][1].synaptic_neurotransmitters
            .insert(
                DefaultReceptorsNeurotransmitterType::X,
                neuro
            );

        let mut cpu_conversion: GridType = vec![vec![BasicIntegrateAndFire::default(), BasicIntegrateAndFire::default()]];

        let gpu_conversion = BasicIntegrateAndFire::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        BasicIntegrateAndFire::convert_electrochemical_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            1,
            2,
            &queue,
        )?;
    
        assert_eq!(cpu_conversion.len(), 1);
        assert_eq!(cpu_conversion[0].len(), 2);
        assert_eq!(cpu_conversion[0][1].receptors.len(), 1);
        match cpu_conversion[0][1].receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
            Some(DefaultReceptorsType::X(receptor)) => assert_eq!(receptor.g, 100.),
            _ => panic!("Unexpected conversion") 
        }
        assert_eq!(cpu_conversion[0][0].current_voltage, -100.);
        assert_eq!(cpu_conversion[0][1].gap_conductance, 100.);
        assert_eq!(
            *cpu_conversion[0][1].synaptic_neurotransmitters.get(&DefaultReceptorsNeurotransmitterType::X).unwrap(), 
            neuro
        );

        Ok(())
    }

    #[test]
    pub fn test_electrochemical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrochemical_kernel(&context) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        }
    }

    // test if neurotransmitters update as expected

    fn get_neurotransmitter(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        let ampa_neurotransmitter = neuron.synaptic_neurotransmitters.get(&DefaultReceptorsNeurotransmitterType::X)
            .expect("Could not get neurotransmitter")
            .t;

        tracker.push(ampa_neurotransmitter);
    }

    fn gpu_get_neurotransmitter(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("neurotransmitters$t").unwrap() {
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
    fn test_neurotransmitter_update() {
        let (cpu_ts, gpu_ts) = iterate_neuron( 
            5.,
            &get_neurotransmitter, 
            &gpu_get_neurotransmitter,
            true,
        ).unwrap();

        assert!(gpu_ts.iter().sum::<f32>() > 0.);

        for (n, (cpu_t, gpu_t)) in cpu_ts.iter().zip(gpu_ts).enumerate() {
            let error = (cpu_t - gpu_t).abs();
            assert!(error < 0.1, "timestep: {} | error: {} < ({} - {})", n, error, cpu_t, gpu_t);
        }
    }

    // neuron will have to build receptor kinetics calls and receptor update calls
    // from the given traits
}
