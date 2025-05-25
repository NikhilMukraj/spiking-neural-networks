#[cfg(feature = "gpu")]
#[cfg(test)]
pub mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::{CL_QUEUE_PROFILING_ENABLE}, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel};
    use spiking_neural_networks::error::SpikingNeuralNetworksError; 


    neuron_builder!(r#"
        [neuron]
            type: BasicIntegrateAndFire
            vars: e = 0, v_reset = -75, v_th = -55, flag1 = 0, flag2 = 0, flag3 = 0
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                dv/dt = -(v - e) + i

                [if] i > 0 [then]
                    flag1 = 1
                [end]

                [if] v_reset > 0 [then]
                    flag2 = 1
                [else]
                    flag2 = 2
                [end]

                [if] e < 0 [then]
                    flag3 = 1
                [elseif] e > 0 [then]
                    flag3 = 2
                [else]
                    flag3 = 3
                [end]
        [end]"#
    );

    #[test]
    fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
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
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron: FullNeuronType = BasicIntegrateAndFire::default();
        neuron.set_dt(0.1);

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
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
            };

        let iterate_kernel = BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context)?;

        let gpu_cell_grid = BasicIntegrateAndFire::convert_to_gpu(&cell_grid, &context, &queue)?;

        let sums_buffer = unsafe {
            create_and_write_buffer(&context, &queue, 1, input)?
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

    fn get_flag1(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        tracker.push(neuron.flag1);
    }

    fn get_gpu_flag1(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("flag1").unwrap() {
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

    fn get_flag2(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        tracker.push(neuron.flag2);
    }

    fn get_gpu_flag2(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("flag2").unwrap() {
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

    fn get_flag3(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        tracker.push(neuron.flag3);
    }

    fn get_gpu_flag3(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("flag3").unwrap() {
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
    fn test_single_neuron_flag1() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_flag1, gpu_flag1) = iterate_neuron(
            5.,
            &get_flag1, 
            &get_gpu_flag1,
        )?;

        for (i, j) in cpu_flag1.iter().zip(gpu_flag1.iter()) {
            assert_eq!(*i, 1.);
            assert_eq!(i, j);
        }
       
        Ok(())
    }

    #[test]
    fn test_single_neuron_flag2() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_flag2, gpu_flag2) = iterate_neuron(
            5.,
            &get_flag2, 
            &get_gpu_flag2,
        )?;

        for (i, j) in cpu_flag2.iter().zip(gpu_flag2.iter()) {
            assert_eq!(*i, 2.);
            assert_eq!(i, j);
        }
       
        Ok(())
    }

    #[test]
    fn test_single_neuron_flag3() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_flag2, gpu_flag2) = iterate_neuron(
            5.,
            &get_flag3, 
            &get_gpu_flag3,
        )?;

        for (i, j) in cpu_flag2.iter().zip(gpu_flag2.iter()) {
            assert_eq!(*i, 3.);
            assert_eq!(i, j);
        }
       
        Ok(())
    }
}