extern crate test;


#[cfg(test)]
mod tests {
    use rand::Rng;
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context,
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError,
        neuron::iterate_and_spike::{
            Neurotransmitters, IonotropicNeurotransmitterType,
            ApproximateNeurotransmitterKinetics,
            AMPADefault, NMDADefault, GABAaDefault, GABAbDefault,
        },
    };

    #[test]
    pub fn test_neurotransmitter_conversion() {
        let mut neurotransmitters1 = Neurotransmitters { neurotransmitters: HashMap::new() };
        neurotransmitters1.neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitterKinetics::AMPADefault
        );
        neurotransmitters1.neurotransmitters.insert(
            IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitterKinetics::NMDADefault
        );
        let mut neurotransmitters2 = Neurotransmitters { neurotransmitters: HashMap::new() };
        neurotransmitters2.neurotransmitters.insert(
            IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitterKinetics::NMDADefault
        );
        let mut neurotransmitters3 = Neurotransmitters { neurotransmitters: HashMap::new() };
        neurotransmitters2.neurotransmitters.insert(
            IonotropicNeurotransmitterType::GABAa, ApproximateNeurotransmitterKinetics::GABAaDefault
        );
        neurotransmitters2.neurotransmitters.insert(
            IonotropicNeurotransmitterType::GABAb, ApproximateNeurotransmitterKinetics::GABAbDefault
        );
        let mut neurotransmitters4 = Neurotransmitters { neurotransmitters: HashMap::new() };

        let neurotransmitters_grid = vec![
            vec![neurotransmitters, neurotransmitters2, neurotransmitters3, neurotransmitters4]
        ];

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = Neurotransmitters::convert_to_gpu(
            neurotransmitter_grid,
            &context,
            &queue,
            1,
            4,
        );

        let cpu_conversion = Neurotransmitters::convert_to_cpu(
            neurotransmitters_grid.clone(),
            &queue,
            1,
            4,
        );

        assert_eq!(cpu_conversion, neurotransmitters);
    }
}
