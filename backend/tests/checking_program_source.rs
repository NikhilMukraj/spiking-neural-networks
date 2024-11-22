mod tests {
    use opencl3::{
        // command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{
        // error::SpikingNeuralNetworksError, 
        neuron::{
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor, IterateAndSpikeGPU,
            }
        }
    };

    #[test]
    pub fn test_program_source() {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let kernel_function = QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::
            iterate_and_spike_electrochemical_kernel(&context);

        assert!(kernel_function.is_ok());

        println!("{}", kernel_function.unwrap().program_source);
    }
}
