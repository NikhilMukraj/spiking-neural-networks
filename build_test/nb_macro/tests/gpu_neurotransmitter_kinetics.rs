#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use std::{collections::HashMap, ptr};
    use nb_macro::neuron_builder; 
    use spiking_neural_networks::{
        error::{GPUError, SpikingNeuralNetworksError},
        neuron::{
            iterate_and_spike::{
                BufferGPU, DefaultReceptorsNeurotransmitterType, NeurotransmitterTypeGPU, Neurotransmitters
            }, 
            spike_train::{DeltaDiracRefractoriness, PoissonNeuron, SpikeTrainGPU}
        },
    };
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_READ_WRITE}, program::Program, types::{cl_float, CL_NON_BLOCKING}
    };
    

    neuron_builder!(r#"
    [neurotransmitter_kinetics]
        type: BasicNeurotransmitterKinetics
        vars: t_max = 1, c = 0.001, conc = 0
        on_iteration:
            [if] is_spiking [then]
                conc = t_max
            [else]
                conc = 0
            [end]

            t = t + dt * -c * t + conc

            t = min(max(t, 0), t_max)
    [end]
    "#);

    type GridType = Vec<Vec<Neurotransmitters<DefaultReceptorsNeurotransmitterType, BasicNeurotransmitterKinetics>>>;

    // get neurotransmitters program source from neurotransmitters struct and then
    // check that program source compiles
    // compile Neurotransmitters::<T>::get_neurotransmitter_update_kernel_code() from string

    #[test]
    pub fn test_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let program_source = format!(
            "{}\n{}",
            BasicNeurotransmitterKinetics::get_update_function().1,
            Neurotransmitters::<DefaultReceptorsNeurotransmitterType, BasicNeurotransmitterKinetics>::get_neurotransmitter_update_kernel_code()
        );

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_name = String::from("neurotransmitters_update");

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
    pub fn test_empty_neurotransmitter_conversion() -> Result<(), SpikingNeuralNetworksError> {
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

        let neurotransmitters_grid: GridType = vec![];

        let gpu_conversion = Neurotransmitters::convert_to_gpu(
            &neurotransmitters_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion = neurotransmitters_grid.clone();
        Neurotransmitters::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            0,
            0,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(neurotransmitters_grid.iter()) {
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
    pub fn test_neurotransmitter_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut neurotransmitters1 = Neurotransmitters::default();
        neurotransmitters1.insert(
            DefaultReceptorsNeurotransmitterType::X, BasicNeurotransmitterKinetics {
                t_max: 0.5,
                ..BasicNeurotransmitterKinetics::default()
            }
        );
        let mut neurotransmitters2 = Neurotransmitters::default();
        neurotransmitters2.insert(
            DefaultReceptorsNeurotransmitterType::X, BasicNeurotransmitterKinetics {
                c: 0.02,
                ..BasicNeurotransmitterKinetics::default()
            }
        );
        let mut neurotransmitters3 = Neurotransmitters::default();
        neurotransmitters3.insert(
            DefaultReceptorsNeurotransmitterType::X, BasicNeurotransmitterKinetics {
                t: 0.02,
                ..BasicNeurotransmitterKinetics::default()
            }
        );
        neurotransmitters3.insert(
            DefaultReceptorsNeurotransmitterType::X, BasicNeurotransmitterKinetics::default()
        );
        let neurotransmitters4 = Neurotransmitters::default();

        let neurotransmitters_grid = vec![
            vec![neurotransmitters1, neurotransmitters2, neurotransmitters3, neurotransmitters4]
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

        let gpu_conversion = Neurotransmitters::convert_to_gpu(
            &neurotransmitters_grid,
            &context,
            &queue,
        )?;

        let mut cpu_conversion: GridType = vec![
            vec![
                Neurotransmitters::default(), Neurotransmitters::default(), 
                Neurotransmitters::default(), Neurotransmitters::default()
            ]
        ];
        Neurotransmitters::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            &queue,
            1,
            4,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(neurotransmitters_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual, 
                    expected,
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
    type SpikeTrainType = PoissonNeuron<DefaultReceptorsNeurotransmitterType, BasicNeurotransmitterKinetics, DeltaDiracRefractoriness>;

    fn iterate_neuron(
        gpu_get_attribute: &GetGPUAttribute,
    ) -> Result<Vec<f32>, SpikingNeuralNetworksError> {
        let iterations = 1000;
        
        let mut neuron = PoissonNeuron::default();
        neuron.synaptic_neurotransmitters.insert(DefaultReceptorsNeurotransmitterType::X, BasicNeurotransmitterKinetics::default());

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
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
            };

        let iterate_kernel = SpikeTrainType::spike_train_electrochemical_kernel(&context)?;

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
                        kernel_execution.set_arg(&DefaultReceptorsNeurotransmitterType::number_of_types());
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

        Ok(gpu_tracker)
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
    pub fn test_range() -> Result<(), SpikingNeuralNetworksError> {
        let ts = iterate_neuron(&gpu_get_neurotransmitter)?;

        for i in ts {
            assert!(i <= 1.);
            assert!(i >= 0.);
        }

        Ok(())
    } 

    #[test]
    pub fn test_decay() -> Result<(), SpikingNeuralNetworksError> {
        let ts = iterate_neuron(&gpu_get_neurotransmitter)?;

        assert!(ts.iter().sum::<f32>() > 0.);

        for i in 0..(ts.len() - 1) {
            if ts[i + 1] != 1.0 {
                let decayed = ts[i] + -0.001 * ts[i];
                assert!(
                    (ts[i + 1] - decayed).abs() < 0.01, 
                    "{} to {}, expected: {}", 
                    ts[i], ts[i + 1], decayed
                );
            }
        }

        Ok(())
    }
}
