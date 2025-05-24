#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::{AdjacencyMatrix, Graph, GraphPosition}, 
        neuron::{
            gpu_lattices::LatticeNetworkGPU, 
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType
            }, 
            plasticity::STDP, 
            spike_train::{
                DeltaDiracRefractoriness, RateSpikeTrain
            }, 
            GridVoltageHistory, Lattice, LatticeNetwork, SpikeTrainGridHistory, SpikeTrainLattice
        }
    };


    type SpikeTrainType = RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;
    type SpikeTrainLatticeType = SpikeTrainLattice<IonotropicNeurotransmitterType, SpikeTrainType, SpikeTrainGridHistory>;
    type NeuronType = QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
    type LatticeType = Lattice<NeuronType, AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, STDP, IonotropicNeurotransmitterType>;
    type NetworkType = LatticeNetwork<
        NeuronType,
        AdjacencyMatrix<(usize, usize), f32>,
        GridVoltageHistory,
        SpikeTrainType,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition, f32>,
        STDP,
        IonotropicNeurotransmitterType,
    >;

    #[test]
    fn test_connection_function() -> Result<(), SpikingNeuralNetworksError> {
        let spike_train: SpikeTrainType = RateSpikeTrain::default(); 

        let mut spike_train_lattice: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice.populate(&spike_train, 3, 3)?;
        spike_train_lattice.update_grid_history = true;
        let spike_train_lattices: Vec<SpikeTrainLatticeType> = vec![spike_train_lattice];

        let base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        let mut lattice: LatticeType = Lattice::default();
        lattice.set_id(1);
        lattice.populate(&base_neuron, 3, 3)?;
        lattice.update_grid_history = true;
        // lattice.connect(&connection_conditional, None);
        let lattices: Vec<LatticeType> = vec![lattice];

        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;
        network.connect(0, 1, &(|_, _| true), Some(&(|_, _| 2.)))?; 
        gpu_network.connect(0, 1, &(|_, _| true), Some(&(|_, _| 2.)))?; 

        let network_connecting_graph = network.get_connecting_graph();
        let gpu_network_connecting_graph = gpu_network.get_connecting_graph();

        let positions = network_connecting_graph.index_to_position.values().collect::<HashSet<_>>();
        let gpu_positions = gpu_network_connecting_graph.index_to_position.values().collect::<HashSet<_>>();

        assert_eq!(positions, gpu_positions);
    
        for presynaptic in positions.iter() {
            for postsynaptic in positions.iter() {
                assert_eq!(
                    network_connecting_graph.lookup_weight(*presynaptic, *postsynaptic),
                    gpu_network_connecting_graph.lookup_weight(*presynaptic, *postsynaptic),
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_connection_function_with_multiple_spike_train_lattices() -> Result<(), SpikingNeuralNetworksError> {
        let spike_train: SpikeTrainType = RateSpikeTrain::default(); 

        let mut spike_train_lattice1: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice1.populate(&spike_train, 3, 3)?;
        let mut spike_train_lattice2 = spike_train_lattice1.clone();
        spike_train_lattice2.set_id(1);
        let spike_train_lattices: Vec<SpikeTrainLatticeType> = vec![spike_train_lattice1, spike_train_lattice2];

        let base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        let mut lattice: LatticeType = Lattice::default();
        lattice.set_id(2);
        lattice.populate(&base_neuron, 3, 3)?;
        lattice.update_grid_history = true;
        // lattice.connect(&connection_conditional, None);
        let lattices: Vec<LatticeType> = vec![lattice];

        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;
        network.connect(0, 2, &(|_, _| true), Some(&(|_, _| 2.)))?;
        network.connect(1, 2, &(|_, _| true), Some(&(|_, _| 0.5)))?; 
        gpu_network.connect(0, 2, &(|_, _| true), Some(&(|_, _| 2.)))?; 
        gpu_network.connect(1, 2, &(|_, _| true), Some(&(|_, _| 0.5)))?; 

        let network_connecting_graph = network.get_connecting_graph();
        let gpu_network_connecting_graph = gpu_network.get_connecting_graph();

        let positions = network_connecting_graph.index_to_position.values().collect::<HashSet<_>>();
        let gpu_positions = gpu_network_connecting_graph.index_to_position.values().collect::<HashSet<_>>();

        assert_eq!(positions, gpu_positions);
    
        for presynaptic in positions.iter() {
            for postsynaptic in positions.iter() {
                assert_eq!(
                    network_connecting_graph.lookup_weight(*presynaptic, *postsynaptic),
                    gpu_network_connecting_graph.lookup_weight(*presynaptic, *postsynaptic),
                );
            }
        }

        Ok(())
    }
}
