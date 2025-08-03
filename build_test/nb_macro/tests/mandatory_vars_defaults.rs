#[cfg(test)]
mod tests {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::CL_QUEUE_PROFILING_ENABLE, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::ExecuteKernel};
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, neuron::{gpu_lattices::LatticeNetworkGPU, iterate_and_spike::{DefaultReceptorsType, IonotropicNeurotransmitterType, XReceptor}, Lattice, LatticeNetwork, RunNetwork, SpikeTrainLattice}};


    neuron_builder!(r#"
        [neuron]
            type: BasicIntegrateAndFire
            kinetics: TestNeurotransmitterKinetics, TestReceptorKinetics
            vars: e = 0, v_reset = -75, v_th = -55, dt = 100
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                dv/dt = (v - e) + i
        [end]

        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1, current = 10
            on_iteration:
                current = g * (v - e)
        [end]

        [neuron]
            type: IonChannelNeuron
            ion_channels: l = TestLeak
            vars: v_reset = -75, v_th = -55
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                l.update_current(v)
                dv/dt = l.current + i
        [end]

        [spike_train]
            type: RateSpikeTrain
            vars: step = 0, rate = 0, v_resting = 24
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
        [end]

        [neurotransmitter_kinetics]
            type: TestNeurotransmitterKinetics
            vars: t = 0.5, t_max = 1, c = 0.001, conc = 0
            on_iteration:
                [if] is_spiking [then]
                    conc = t_max
                [else]
                    conc = 0
                [end]

                t = t + dt * -c * t + conc

                t = min(max(t, 0), t_max)
        [end]

        [receptor_kinetics]
            type: TestReceptorKinetics
            vars: r = 0.5, r_max = 1
            on_iteration:
                r = min(max(t, 0), r_max)
        [end]

        [neural_refractoriness]
            type: TestRefractoriness
            vars: decay = 5000
            effect: (v_th - v_resting) * exp((-1 / (decay / dt)) * (time_difference ^ 2)) + v_resting
        [end]
    "#);

    #[test]
    fn test_custom_dt() {
        let lif = BasicIntegrateAndFire::default_impl();

        assert_eq!(lif.dt, 100.);
    }

    #[test]
    fn test_custom_current() {
        let ion_channel = TestLeak::default();

        assert_eq!(ion_channel.current, 10.);
    }

    #[test]
    fn test_custom_v_resting() {
        let spike_train = RateSpikeTrain::default_impl();

        assert_eq!(spike_train.v_resting, 24.);
    }

    #[test]
    fn test_custom_neurotransmitter_kinetics() {
        let kinetics = TestNeurotransmitterKinetics::default();

        assert_eq!(kinetics.t, 0.5);
    }

    #[test]
    fn test_custom_receptor_kinetics() {
        let kinetics = TestReceptorKinetics::default();

        assert_eq!(kinetics.r, 0.5);
    }

     #[test]
    fn test_custom_neural_refractoriness() {
        let refractoriness = TestRefractoriness::default();

        assert_eq!(refractoriness.decay, 5000.);
    }

    // test if gpu kernel and conversion works as expected

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

    type FullNeuronType = BasicIntegrateAndFire<TestNeurotransmitterKinetics, TestReceptorKinetics>;
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
        
        let mut neuron: FullNeuronType = BasicIntegrateAndFire::default();
        neuron.set_dt(0.1);

        neuron.synaptic_neurotransmitters.insert(
            DefaultReceptorsNeurotransmitterType::X, 
            TestNeurotransmitterKinetics::default(),
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
            BasicIntegrateAndFire::<TestNeurotransmitterKinetics, TestReceptorKinetics>::iterate_and_spike_electrical_kernel(&context)?
        } else {
            BasicIntegrateAndFire::<TestNeurotransmitterKinetics, TestReceptorKinetics>::iterate_and_spike_electrochemical_kernel(&context)?
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
    pub fn test_single_neuron_is_spiking() -> Result<(), SpikingNeuralNetworksError> {
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

    #[test]
    fn test_single_neuron_voltage() -> Result<(), SpikingNeuralNetworksError> {
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

     #[test]
    fn test_single_electrochemical_neuron_voltage() -> Result<(), SpikingNeuralNetworksError> {
        let (cpu_voltages, gpu_voltages) = iterate_neuron(
            0.,
            1.,
            &get_voltage, 
            &get_gpu_voltage,
            true,
            true,
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
    fn test_spike_train_lattice() -> Result<(), SpikingNeuralNetworksError> {
        let spike_train: RateSpikeTrain<DefaultReceptorsNeurotransmitterType, TestNeurotransmitterKinetics, TestRefractoriness> = RateSpikeTrain::default();

        let mut spike_train_lattice = SpikeTrainLattice::default_impl();
        spike_train_lattice.populate(&spike_train, 1, 1)?;
        spike_train_lattice.update_grid_history = true;

        let base_neuron = BasicIntegrateAndFire::default_impl();

        let mut lattice = Lattice::default_impl();
        lattice.set_id(1);
        lattice.populate(&base_neuron, 1, 1)?;
        lattice.update_grid_history = true;

        let lattices = vec![lattice];
        let spike_train_lattices = vec![spike_train_lattice];

        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(100)?;
        gpu_network.run_lattices(100)?;

        for (n, (cpu_grid, gpu_grid)) in network.get_lattice(&1).unwrap().grid_history.history.iter()
            .zip(gpu_network.get_lattice(&1).unwrap().grid_history.history.iter())
            .enumerate() {
            for (cpu_row, gpu_row) in cpu_grid.iter().zip(gpu_grid.iter()) {
                for (i, j) in cpu_row.iter().zip(gpu_row.iter()) {
                    if !i.is_finite() || !j.is_finite() {
                        continue;
                    }
                    assert!((i - j).abs() < 3., "{}: |{} - {}| = {}", n, i, j, (i - j).abs());
                }
            }
        }

        Ok(())
    }

    // #[test]
    // fn test_ion_channel_neuron() -> Result<(), SpikingNeuralNetworksError> {
        // let base_neuron = IonChannelNeuron::default_impl();
        // let mut lattice = Lattice::default_impl();
        // lattice.populate(&base_neuron, 3, 3)?;
        // lattice.update_grid_history = true;
        // let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
        // lattice.run_lattice(1000)?;
        // gpu_lattice.run_lattice(1000)?;
    // }
}
