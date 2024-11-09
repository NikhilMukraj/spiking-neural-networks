

#[cfg(test)]
mod tests {
    extern crate spiking_neural_networks;
    use spiking_neural_networks::graph::AdjacencyMatrix;
    use spiking_neural_networks::neuron::hodgkin_huxley::HodgkinHuxleyNeuron;
    use spiking_neural_networks::neuron::iterate_and_spike::{
        AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, 
        IonotropicNeurotransmitterType, IterateAndSpike, LigandGatedChannel, 
        LigandGatedChannels, NeurotransmitterKinetics, Neurotransmitters
    };
    use spiking_neural_networks::neuron::plasticity::STDP;
    use spiking_neural_networks::neuron::spike_train::NeuralRefractoriness;
    use spiking_neural_networks::neuron::{
        integrate_and_fire::IzhikevichNeuron,
        spike_train::PoissonNeuron,
        SpikeTrainLattice, Lattice, LatticeNetwork, SpikeHistory
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;
    

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        x == y
    }

    type SpikeTrainType<K, U> = PoissonNeuron<IonotropicNeurotransmitterType, K, U>;

    fn get_history_from_example<T, K, U>(
        neuron: &mut T,
        spike_train: &mut SpikeTrainType<K, U>,
        size: (usize, usize),
        iterations: usize,
        dt: f32,
        weight: f32,
        synapses: (bool, bool)
    ) -> Result<SpikeHistory, SpikingNeuralNetworksError>
    where
        T: IterateAndSpike<N = IonotropicNeurotransmitterType>,
        K: NeurotransmitterKinetics,
        U: NeuralRefractoriness,
    {
        let mut spike_train_lattice = SpikeTrainLattice::<IonotropicNeurotransmitterType, SpikeTrainType<K, U>, _>::default_impl();
        spike_train_lattice.set_id(0);
        spike_train_lattice.populate(spike_train, size.0, size.1);
        spike_train_lattice.update_grid_history = true;
    
        let mut lattice = Lattice::<T, _, _, _, IonotropicNeurotransmitterType>::default();
        lattice.set_id(1);
        lattice.populate(neuron, size.0, size.1);
        lattice.update_grid_history = true;
    
        let lattices: Vec<
            Lattice<_, AdjacencyMatrix<_, _>, SpikeHistory, STDP, IonotropicNeurotransmitterType>
        > = vec![lattice];
        let spike_train_lattices = vec![spike_train_lattice];
        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
    
        let weight_fn: &dyn Fn((usize, usize), (usize, usize)) -> f32 = &|_, _| weight;
        network.connect(0, 1, &connection_conditional, Some(weight_fn))?;
        network.parallel = true;
        network.electrical_synapse = synapses.0;
        network.chemical_synapse = synapses.1;
        network.set_dt(dt);

        network.run_lattices(iterations)?;
    
        network.get_mut_spike_train_lattice(&0).unwrap().apply(|neuron| {
            neuron.chance_of_firing = (dt / 1.) * 0.01;
        });
        
        network.run_lattices(iterations)?;

        Ok(network.get_lattice(&1).unwrap().grid_history.clone())
    }
    
    fn counts_spikes_in_range(
        data: &[Vec<Vec<bool>>], 
        start: usize, 
        end: usize
    ) -> usize {
        let bounded_end = end.min(data.len());
    
        data[start..bounded_end].iter()
            .flat_map(|inner_vec| inner_vec.iter())
            .flat_map(|innermost_vec| innermost_vec.iter())
            .filter(|&&value| value)
            .count()
    }    

    #[test]
    pub fn test_electrical_synapse_input() -> Result<(), SpikingNeuralNetworksError> {
        let mut neurotransmitters = Neurotransmitters::default();
        neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );
        let mut ligand_gates = LigandGatedChannels::default();
        ligand_gates.insert(
            IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default()
        )?;

        let mut izhikevich_neuron = IzhikevichNeuron::default_impl();
        izhikevich_neuron.gap_conductance = 10.;
        izhikevich_neuron.synaptic_neurotransmitters = neurotransmitters.clone();
        izhikevich_neuron.ligand_gates = ligand_gates;
        let mut poisson_neuron = PoissonNeuron::default_impl();
        poisson_neuron.synaptic_neurotransmitters = neurotransmitters;

        let iterations = 2500;
        let history = get_history_from_example(
            &mut izhikevich_neuron, &mut poisson_neuron, (1, 1,), iterations, 1.0, 1.0, (true, false)
        )?;

        // check that before 2500 it is <=1, then after it is >= 1

        assert!(counts_spikes_in_range(&history.history, 0, iterations) <= 1);
        let after_activation_spikes = counts_spikes_in_range(
            &history.history, iterations, iterations + iterations
        );
        assert!(after_activation_spikes > 2, "number of spikes: {}", after_activation_spikes);

        Ok(())
    }

    #[test]
    pub fn test_chemical_synapse_input() -> Result<(), SpikingNeuralNetworksError> {
        let mut neurotransmitters = Neurotransmitters::default();
        neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );
        let mut ligand_gates = LigandGatedChannels::default();
        ligand_gates.insert(
            IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default()
        )?;
        
        let mut izhikevich_neuron = IzhikevichNeuron::default_impl();
        izhikevich_neuron.gap_conductance = 10.;
        izhikevich_neuron.synaptic_neurotransmitters = neurotransmitters.clone();
        izhikevich_neuron.ligand_gates = ligand_gates;
        let mut poisson_neuron = PoissonNeuron::default_impl();
        poisson_neuron.synaptic_neurotransmitters = neurotransmitters;

        let iterations = 2500;
        let history = get_history_from_example(
            &mut izhikevich_neuron, &mut poisson_neuron, (1, 1), iterations, 1.0, 1.0, (false, true)
        )?;

        // check that before 2500 it is <=1, then after it is >= 1

        assert!(counts_spikes_in_range(&history.history, 0, iterations) <= 1);
        let after_activation_spikes = counts_spikes_in_range(
            &history.history, iterations, iterations + iterations
        );
        assert!(after_activation_spikes > 2, "number of spikes: {}", after_activation_spikes);

        Ok(())
    }

    #[test]
    pub fn test_hodgkin_huxley_chemical_synapse_input() -> Result<(), SpikingNeuralNetworksError> {
        let mut neurotransmitters = Neurotransmitters::default();
        neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );
        let mut spike_train_neurotransmitters = Neurotransmitters::default();
        spike_train_neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );
        let mut ligand_gates = LigandGatedChannels::<ApproximateReceptor>::default();
        ligand_gates.insert(
            IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default()
        )?;
        
        let mut hodgkin_huxley_neuron = HodgkinHuxleyNeuron {
            synaptic_neurotransmitters: neurotransmitters,
            ligand_gates,
            ..HodgkinHuxleyNeuron::default()
        };
        let mut poisson_neuron = PoissonNeuron::default_impl();
        poisson_neuron.synaptic_neurotransmitters = spike_train_neurotransmitters;

        let iterations = 100000;
        let history = get_history_from_example(
            &mut hodgkin_huxley_neuron, &mut poisson_neuron, (1, 1), iterations, 0.01, 1.0, (false, true)
        )?;

        // check that before 2500 it is <=1, then after it is >= 1

        assert!(counts_spikes_in_range(&history.history, 0, iterations) <= 1);
        let after_activation_spikes = counts_spikes_in_range(
            &history.history, iterations, iterations + iterations
        );
        assert!(after_activation_spikes > 2, "number of spikes: {}", after_activation_spikes);

        // similar results could be achieved by increasing c_m rather than iterations

        Ok(())
    }
}
