#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, program::Program};
    use spiking_neural_networks::error::{GPUError, SpikingNeuralNetworksError};


    neuron_builder!(r#"
    [receptor_kinetics]
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]
    "#);

    #[test]
    pub fn test_compiles() -> Result<(), SpikingNeuralNetworksError> {
        let program_source = BoundedReceptorKinetics::get_update_function().1;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match Program::create_and_build_from_source(&context, &program_source, "") {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::from(GPUError::ProgramCompileFailure)),
        }
    }
}
