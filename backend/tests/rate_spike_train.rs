#[cfg(test)]
mod test {
    use opencl3::{
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}
    };
    use spiking_neural_networks::neuron::{
        iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType}, 
        spike_train::{DeltaDiracRefractoriness, RateSpikeTrain, SpikeTrain, SpikeTrainGPU}
    };


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

    // fn iterate_neuron(
    //     rate: f32,
    //     mutneuron: SpikeTrainType,
    //     gpu_get_attribute: &GetGPUAttribute,
    //     electrical: bool,
    // ) -> Result<Vec<f32>, SpikingNeuralNetworksError> {
    //     let iterations = 1000;
        
    //     let cell_grid = vec![vec![neuron]];

    //     let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
    //         .expect("Could not get GPU devices")
    //         .first()
    //         .expect("No GPU found");
    //     let device = Device::new(device_id);

    //     let context = match Context::from_device(&device) {
    //         Ok(value) => value,
    //         Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
    //     };

    //     let queue =  match CommandQueue::create_default_with_properties(
    //             &context, 
    //             CL_QUEUE_PROFILING_ENABLE,
    //             CL_QUEUE_SIZE,
    //         ) {
    //             Ok(value) => value,
    //             Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::GetDeviceFailure)),
    //         };

    //     let iterate_kernel = if electrical {
    //         SpikeTrainType::spike_train_electrical_kernel(&context)?
    //     } else {
    //         SpikeTrainType::spike_train_electrochemical_kernel(&context)?
    //     };

    //     let gpu_cell_grid = SpikeTrainType::convert_electrochemical_to_gpu(&cell_grid, &context, &queue)?;

    //     let index_to_position_buffer = unsafe {
    //         create_and_write_buffer(&context, &queue, 1, 0.0)?
    //     };

    //     let mut gpu_tracker = vec![];

    //     for _ in 0..iterations {
    //         let iterate_event = unsafe {
    //             let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

    //             let mut counter = 0;

    //             for i in iterate_kernel.argument_names.iter() {
    //                 if i == "number_of_types" {
    //                     kernel_execution.set_arg(&IonotropicReceptorNeurotransmitterType::number_of_types());
    //                 } else if i == "index_to_position" {
    //                     kernel_execution.set_arg(&index_to_position_buffer);
    //                 } else if i == "skip_index" { 
    //                     kernel_execution.set_arg(&0);
    //                 } else if i == "neuro_flags" {
    //                     match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
    //                         BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
    //                         _ => unreachable!("Could not retrieve neurotransmitter flags"),
    //                     };
    //                 } else {
    //                     match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
    //                         BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
    //                         BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
    //                         BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
    //                     };
    //                 }
    //                 counter += 1;
    //             }

    //             assert_eq!(
    //                 counter,
    //                 kernel_execution.num_args, 
    //                 "counter: {} != num_args: {}",
    //                 counter,
    //                 kernel_execution.num_args,
    //             );

    //             match kernel_execution.set_global_work_size(1)
    //                 .enqueue_nd_range(&queue) {
    //                     Ok(value) => value,
    //                     Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::QueueFailure)),
    //                 }
    //         };

    //         match iterate_event.wait() {
    //             Ok(_) => {},
    //             Err(_) => return Err(SpikingNeuralNetworksError::from(GPUError::WaitError)),
    //         };

    //         gpu_get_attribute(&gpu_cell_grid, &queue, &mut gpu_tracker)?;
    //     }

    //     Ok(gpu_tracker)
    // }
}