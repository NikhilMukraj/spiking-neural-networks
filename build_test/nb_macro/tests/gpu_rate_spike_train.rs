#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use spiking_neural_networks::neuron::{iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType}, spike_train::DeltaDiracRefractoriness};


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
}