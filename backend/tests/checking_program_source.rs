mod tests {
    use opencl3::{
        // command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::neuron::{
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType, IterateAndSpike, IterateAndSpikeGPU, LigandGatedChannel, NeurotransmitterConcentrations, Timestep
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
    }

    // #[test]
    // pub fn test_single_quadratic_neuron() {
    //     // initialize 1x1 grid
    //     // give constant ampa input, then constant nmda, gaba, etc
    //     // check against cpu equavilent

    //     // let iterations = 1000;
        
    //     // let neuron = QuadraticIntegrateAndFireNeuron::default_impl();
    //     // neuron.ligand_gates.insert(IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default());
    //     // neuron.ligand_gates.insert(IonotropicNeurotransmitterType::NMDA, LigandGatedChannel::nmda_default());
    //     // neuron.ligand_gates.insert(IonotropicNeurotransmitterType::GABAa, LigandGatedChannel::gabaa_default());

    //     // neuron.set_dt(1.);

    //     // let cpu_neuron = neuron.clone();

    //     // let cell_grid = vec![vec![neuron]];

    //     // let mut ampa_conc = NeurotransmitterConcentrations::new();
    //     // ampa_conc.insert(IonotropicNeurotransmitterType::AMPA, 1.0);

    //     // for _ in 0..iterations {
    //     //     cpu_neuron.iterate_with_neurotransmitter_and_spike(
    //     //         0, 
    //     //         &ampa_conc
    //     //     );
    //     // }

    //     // create 1 length grid for voltage input, init to 0
    //     // create N::number_of_types() length grid, init first index to 1
    //     // should expect only ampa to activate
    // }
}
