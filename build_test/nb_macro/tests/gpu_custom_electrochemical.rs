#[allow(clippy::assign_op_pattern)]
#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use spiking_neural_networks::error::SpikingNeuralNetworksError;


    neuron_builder!(r#"
    [neuron]
        type: ElectroChemicalIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, modifier = 1
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = -(v - e) + i
        on_electrochemical_iteration:
            receptors.update_receptor_kinetics(t, dt)
            receptors.set_receptor_currents(v, dt)
            dv/dt = -(v - e) + i
            v = (modifier * -receptors.get_receptor_currents(dt, c_m)) + v
            synaptic_neurotransmitters.apply_t_changes()
    [end]
    "#);

    #[test]
    pub fn test_electrical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match ElectroChemicalIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrical_kernel(&context) {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        }
    }

    // #[test]
    // pub fn test_electrochemical_kernel_compiles() -> Result<(), SpikingNeuralNetworksError> {
    //     let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
    //         .expect("Could not get GPU devices")
    //         .first()
    //         .expect("No GPU found");
    //     let device = Device::new(device_id);

    //     let context = Context::from_device(&device).expect("Context::from_device failed");

    //     match BasicIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor>::iterate_and_spike_electrochemical_kernel(&context) {
    //         Ok(_) => Ok(()),
    //         Err(_) => Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
    //     }
    // }
}