#[cfg(test)]
mod tests {
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::{AdjacencyMatrix, GraphPosition}, 
        neuron::{
            gpu_lattices::{LatticeGPU, LatticeNetworkGPU}, 
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor,
            }, 
            spike_train::{PoissonNeuron, DeltaDiracRefractoriness},
            plasticity::STDP, 
            Lattice, GridVoltageHistory, LatticeNetwork, SpikeTrainGridHistory
        }
    };

    #[test]
    pub fn test_lattice_compiles() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let mut lattice: Lattice<
            _, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            STDP, 
            _,
        > = Lattice::default_impl();

        lattice.populate(
            &QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::default_impl(), 
            1, 
            1,
        )?;

        match LatticeGPU::from_lattice(lattice) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    #[test]
    pub fn test_network_compiles() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let mut network: LatticeNetwork<
            _, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            PoissonNeuron<_, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
            SpikeTrainGridHistory, 
            AdjacencyMatrix<GraphPosition, f32>, 
            STDP, 
            _,
        > = LatticeNetwork::default_impl();

        #[allow(clippy::type_complexity)]
        let mut lattice: Lattice<
            _, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            STDP, 
            _,
        > = Lattice::default_impl();

        lattice.populate(
            &QuadraticIntegrateAndFireNeuron::<ApproximateNeurotransmitter, ApproximateReceptor>::default_impl(), 
            1, 
            1,
        )?;

        network.add_lattice(lattice)?;

        match LatticeNetworkGPU::from_network(network) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}