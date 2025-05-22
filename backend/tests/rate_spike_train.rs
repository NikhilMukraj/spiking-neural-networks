#[cfg(test)]
mod test {
    use std::{collections::HashMap, ptr};
    use opencl3::{
        command_queue::{
            CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE
        }, 
        context::Context, 
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
        kernel::ExecuteKernel, 
        memory::{Buffer, CL_MEM_READ_WRITE}, 
        types::{cl_float, CL_NON_BLOCKING}
    };
    use spiking_neural_networks::{
        error::{GPUError, SpikingNeuralNetworksError}, 
        neuron::{
            iterate_and_spike::{
                ApproximateNeurotransmitter, BufferGPU, IonotropicNeurotransmitterType, 
                IonotropicReceptorNeurotransmitterType, NeurotransmitterTypeGPU
            }, 
        spike_train::{DeltaDiracRefractoriness, RateSpikeTrain, SpikeTrain, SpikeTrainGPU}
    }};


    const ITERATIONS: usize = 10_000;

    #[test]
    fn test_expected_rate() {
        let rates = [0, 100, 200, 300, 400, 500];

        for rate in rates {
            let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain {
                rate: rate as f32,
                ..Default::default()
            };

            let mut spikes = 0;
            for _ in 0..ITERATIONS {
                let is_spiking = spike_train.iterate();
                if is_spiking {
                    spikes += 1;
                }
            }

            if rate == 0 {
                assert_eq!(spikes, 0);
            } else {
                assert!((spikes as f32 - (ITERATIONS as f32 / (rate as f32 / 0.1))).abs() <= 1.);
            }
        }
    }

    #[test]
    fn test_spacing() {
        let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain {
            rate: 100.,
            dt: 1.,
            ..Default::default()
        };

        for i in 0..1001 {
            let is_spiking = spike_train.iterate();

            if i == 0 || (i + 1) % 100 != 0 {
                assert!(!is_spiking);
            } else {
                assert!(is_spiking);
            }
        }
    }

    type SpikeTrainType = RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;

    #[test]
    pub fn test_program_source() {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_function = SpikeTrainType::spike_train_electrical_kernel(&context);

        assert!(kernel_function.is_ok());

        let kernel_function = SpikeTrainType::spike_train_electrochemical_kernel(&context);

        assert!(kernel_function.is_ok());
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

        let neuron = SpikeTrainType { rate, ..Default::default() };
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
                CL_QUEUE_SIZE,
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
                    } else if i == "skip_index" { 
                        kernel_execution.set_arg(&0);
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
    fn test_electrical_iteration_voltage() -> Result<(), SpikingNeuralNetworksError> {
        let rates = [0., 100., 200., 300., 400.,];

        for rate in rates {
            let (cpu_voltages, gpu_voltages) = iterate_neuron(
                rate, 
                &get_cpu_voltage, 
                &get_gpu_voltage, 
                true,
            )?;

            for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages.iter()) {
                assert_eq!(cpu_voltage, gpu_voltage);
                assert!(*cpu_voltage == 30. || *cpu_voltage == 0.);
                assert!(*gpu_voltage == 30. || *gpu_voltage == 0.);
            }
        }

        Ok(())
    }

    #[test]
    fn test_electrical_iteration_is_spiking() -> Result<(), SpikingNeuralNetworksError> {
        let rates = [0., 100., 200., 300., 400.,];

        for rate in rates {
            let (cpu_is_spikings, gpu_is_spikings) = iterate_neuron(
                rate, 
                &get_cpu_is_spiking, 
                &get_gpu_is_spiking, 
                true,
            )?;

            for (cpu_is_spiking, gpu_is_spiking) in cpu_is_spikings.iter().zip(gpu_is_spikings.iter()) {
                assert_eq!(cpu_is_spiking, gpu_is_spiking);
            }
        }

        Ok(())
    }
}