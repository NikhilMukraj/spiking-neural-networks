// check that program compiles
// check that program emits spikes
// check that program correctly modifies neurotransmitters

mod tests {
    use spiking_neural_networks::{
        neuron::{
            iterate_and_spike::{
                ApproximateNeurotransmitter, IonotropicNeurotransmitterType, BufferGPU, Timestep,
                NMDADefault, AMPADefault, GABAaDefault, NeurotransmitterTypeGPU,
            }, 
            spike_train::{DeltaDiracRefractoriness, PoissonNeuron, SpikeTrainGPU},
        },
        error::{SpikingNeuralNetworksError, GPUError},
    };
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, 
        context::Context, 
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
        kernel::ExecuteKernel, 
        memory::{Buffer, CL_MEM_READ_WRITE}, 
        types::{cl_float, CL_NON_BLOCKING}
    };
    use std::{ptr, collections::HashMap};

    type SpikeTrainType = PoissonNeuron::<
        IonotropicNeurotransmitterType, 
        ApproximateNeurotransmitter, 
        DeltaDiracRefractoriness
    >;

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
        gpu_get_attribute: &GetGPUAttribute,
        electrical: bool,
    ) -> Result<Vec<f32>, SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron = PoissonNeuron::default_impl();
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default());
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::nmda_default());
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::GABAa, ApproximateNeurotransmitter::gabaa_default());

        neuron.set_dt(1.);

        neuron.chance_of_firing = 0.05;

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
                        kernel_execution.set_arg(&IonotropicNeurotransmitterType::number_of_types());
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

        Ok(gpu_tracker)
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

    // check if spikes are being emitted
    // check that neurotransmitter is being updated appropriately

    #[test]
    pub fn test_voltage() -> Result<(), SpikingNeuralNetworksError> {
        let voltages = iterate_neuron(&get_gpu_voltage, true)?;

        let mut unique_values = vec![];

        for i in voltages {
            if !unique_values.contains(&i) {
                unique_values.push(i);
            }
        }

        assert_eq!(2, unique_values.len());

        Ok(())
    }

    #[test]
    pub fn test_is_spiking() -> Result<(), SpikingNeuralNetworksError> {
        let is_spikings = iterate_neuron(&get_gpu_is_spiking, true)?;

        let mut unique_values = vec![];

        for i in is_spikings {
            if !unique_values.contains(&i) {
                unique_values.push(i);
            }
        }
        
        assert_eq!(2, unique_values.len());

        Ok(())
    }

    #[test]
    pub fn test_voltage_when_chemical() -> Result<(), SpikingNeuralNetworksError> {
        let voltages = iterate_neuron(&get_gpu_voltage, false)?;

        let mut unique_values = vec![];

        for i in voltages {
            if !unique_values.contains(&i) {
                unique_values.push(i);
            }
        }

        assert_eq!(2, unique_values.len());

        Ok(())
    }

    #[test]
    pub fn test_is_spiking_when_chemical() -> Result<(), SpikingNeuralNetworksError> {
        let is_spikings = iterate_neuron(&get_gpu_is_spiking, false)?;

        let mut unique_values = vec![];

        for i in is_spikings {
            if !unique_values.contains(&i) {
                unique_values.push(i);
            }
        }
        
        assert_eq!(2, unique_values.len());

        Ok(())
    }

    fn gpu_get_ampa_neurotransmitter(gpu_cell_grid: &HashMap<String, BufferGPU>, queue: &CommandQueue, gpu_tracker: &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError> {
        match gpu_cell_grid.get("neurotransmitters$t").unwrap() {
            BufferGPU::Float(buffer) => {
                let mut read_vector = vec![0., 0., 0., 0.];

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
    pub fn test_ampa() -> Result<(), SpikingNeuralNetworksError> {
        let ampas = iterate_neuron(&gpu_get_ampa_neurotransmitter, false)?;

        for i in ampas {
            assert!(i <= 1.);
            assert!(i >= 0.);
        }

        Ok(())
    }    

    // test refractoriness functionality
}
