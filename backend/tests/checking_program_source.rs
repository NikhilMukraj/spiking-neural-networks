mod tests {
    use std::{collections::HashMap, ptr};

    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel, memory::{Buffer, CL_MEM_READ_WRITE}, types::{cl_float, CL_NON_BLOCKING}
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{error::{GPUError, SpikingNeuralNetworksError}, neuron::{
        integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
        iterate_and_spike::{
            AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, BufferGPU, GABAaDefault, IonotropicNeurotransmitterType, IterateAndSpike, IterateAndSpikeGPU, LigandGatedChannel, NMDADefault, NeurotransmitterConcentrations, NeurotransmitterTypeGPU, Timestep
        }
    }};

    #[test]
    pub fn test_program_source() {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_function = QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::
            iterate_and_spike_electrical_kernel(&context);

        assert!(kernel_function.is_ok());

        let kernel_function = QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::
            iterate_and_spike_electrochemical_kernel(&context);

        assert!(kernel_function.is_ok());
    }

    fn create_and_write_buffer<T>(
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

    type FullNeuronType = QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
    type GetGPUAttribute = dyn Fn(&HashMap<String, BufferGPU>, &CommandQueue, &mut Vec<f32>) -> Result<(), SpikingNeuralNetworksError>;

    fn iterate_neuron(
        t_values_tuple: &(f32, f32, f32),
        cpu_get_attribute: &dyn Fn(&FullNeuronType, &mut Vec<f32>),
        gpu_get_attribute: &GetGPUAttribute,
    ) -> Result<(Vec<f32>, Vec<f32>), SpikingNeuralNetworksError> {
        // initialize 1x1 grid
        // give constant ampa input, then constant nmda, gaba, etc
        // check against cpu equavilent

        let t_values = vec![t_values_tuple.0, t_values_tuple.1, t_values_tuple.2, 0.];

        let iterations = 1000;
        
        let mut neuron = QuadraticIntegrateAndFireNeuron::default_impl();
        neuron.ligand_gates.insert(IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default())?;
        neuron.ligand_gates.insert(IonotropicNeurotransmitterType::NMDA, LigandGatedChannel::nmda_default())?;
        neuron.ligand_gates.insert(IonotropicNeurotransmitterType::GABAa, LigandGatedChannel::gabaa_default())?;
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default());
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::nmda_default());
        neuron.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::GABAa, ApproximateNeurotransmitter::gabaa_default());

        neuron.set_dt(1.);

        let mut cpu_neuron = neuron.clone();

        let mut neurotransmitter_conc = NeurotransmitterConcentrations::new();
        neurotransmitter_conc.insert(IonotropicNeurotransmitterType::AMPA, t_values_tuple.0);
        neurotransmitter_conc.insert(IonotropicNeurotransmitterType::NMDA, t_values_tuple.1);
        neurotransmitter_conc.insert(IonotropicNeurotransmitterType::GABAa, t_values_tuple.2);
        
        let mut cpu_tracker = vec![];

        for _ in 0..iterations {
            cpu_neuron.iterate_with_neurotransmitter_and_spike(
                0., 
                &neurotransmitter_conc
            );
            cpu_get_attribute(&cpu_neuron, &mut cpu_tracker);
        }

        // create 1 length grid for voltage input, init to 0
        // create N::number_of_types() length grid, init first index to 1
        // should expect only ampa to activate

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

        let iterate_kernel = QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrochemical_kernel(&context)?;

        let gpu_cell_grid = QuadraticIntegrateAndFireNeuron::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?;

        let sums_buffer = create_and_write_buffer(&context, &queue, 1, 0.0)?;
    
        let mut t_buffer = unsafe {
            Buffer::<cl_float>::create(
                &context, 
                CL_MEM_READ_WRITE, 
                IonotropicNeurotransmitterType::number_of_types(), 
                ptr::null_mut()
            )
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let t_buffer_write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut t_buffer, CL_NON_BLOCKING, 0, &t_values, &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };
    
        t_buffer_write_event.wait().map_err(|_| GPUError::WaitError)?;

        let index_to_position_buffer = create_and_write_buffer(&context, &queue, 1, 0.0)?;

        let mut gpu_tracker = vec![];

        for _ in 0..iterations {
            let iterate_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                for i in iterate_kernel.argument_names.iter() {
                    if i == "inputs" {
                        kernel_execution.set_arg(&sums_buffer);
                    } else if i == "t" {
                        kernel_execution.set_arg(&t_buffer);
                    } else if i == "index_to_position" {
                        kernel_execution.set_arg(&index_to_position_buffer);
                    } else if i == "number_of_types" {
                        kernel_execution.set_arg(&IonotropicNeurotransmitterType::number_of_types());
                    } else if i == "neuro_flags" {
                        match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            _ => unreachable!("Could not retrieve neurotransmitter flags"),
                        };
                    } else if i == "lg_flags" {
                        match &gpu_cell_grid.get("ligand_gates$flags").expect("Could not retrieve receptor flags") {
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            _ => unreachable!("Could not retrieve receptor flags"),
                        };
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
    pub fn test_single_quadratic_neuron_ampa() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(&(1., 0., 0.), &get_voltage, &get_gpu_voltage)?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
            let error = (cpu_voltage - gpu_voltage).abs();
            assert!(error < 5., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
        }

        Ok(())
    }

    #[test]
    pub fn test_single_quadratic_neuron_nmda() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(&(0., 1., 0.), &get_voltage, &get_gpu_voltage)?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
            let error = (cpu_voltage - gpu_voltage).abs();
            assert!(error < 5., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
        }

        Ok(())
    }

    #[test]
    pub fn test_single_quadratic_neuron_gabaa() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(&(0., 0., 1.), &get_voltage, &get_gpu_voltage)?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
            let error = (cpu_voltage - gpu_voltage).abs();
            assert!(error < 5., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
        }

        Ok(())
    }

    #[test]
    pub fn test_single_quadratic_neuron_ampa_nmda() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(&(1., 1., 0.), &get_voltage, &get_gpu_voltage)?;

        for (cpu_voltage, gpu_voltage) in cpu_voltages.iter().zip(gpu_voltages) {
            let error = (cpu_voltage - gpu_voltage).abs();
            assert!(error < 5., "error: {} ({} - {})", error, cpu_voltage, gpu_voltage);
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
    pub fn test_single_quadratic_neuron_is_spiking() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_spikings, gpu_spikings) = iterate_neuron(
            &(1., 0., 0.), 
            &get_is_spiking, 
            &gpu_get_is_spiking,
        )?;

        let cpu_sum = cpu_spikings.iter().sum::<f32>();
        let gpu_sum = gpu_spikings.iter().sum::<f32>();
        let error = (cpu_sum - gpu_sum).abs();

        assert!(error < 2., "error: {} ({} - {})", error, cpu_sum, gpu_sum);

        Ok(())
    }

    fn get_ampa_neurotransmitter(neuron: &FullNeuronType, tracker: &mut Vec<f32>) {
        let ampa_neurotransmitter = neuron.synaptic_neurotransmitters.get(&IonotropicNeurotransmitterType::AMPA)
            .expect("Could not get neurotransmitter")
            .t;

        tracker.push(ampa_neurotransmitter);
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
    pub fn test_single_quadratic_neuron_neurotransmitters() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_ampas, gpu_ampas) = iterate_neuron(
            &(1., 0., 0.), 
            &get_ampa_neurotransmitter, 
            &gpu_get_ampa_neurotransmitter
        )?;

        let cpu_sum = cpu_ampas.iter().sum::<f32>();
        let gpu_sum = gpu_ampas.iter().sum::<f32>();
        let error = (cpu_sum - gpu_sum).abs();

        assert!(error < 5., "error: {} ({} - {})", error, cpu_sum, gpu_sum);

        for (n, (cpu_ampa, gpu_ampa)) in cpu_ampas.iter().zip(gpu_ampas).enumerate() {
            let error = (cpu_ampa - gpu_ampa).abs();
            assert!(error < 0.1, "timestep: {} | error: {} ({} - {})", n, error, cpu_ampa, gpu_ampa);
        }

        Ok(())
    }
}
