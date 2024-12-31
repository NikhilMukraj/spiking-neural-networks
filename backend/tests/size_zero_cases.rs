#[cfg(test)]
mod tests {
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::{AdjacencyMatrix, GraphPosition}, 
        neuron::{
            gpu_lattices::{LatticeGPU, LatticeNetworkGPU}, 
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType
            }, 
            spike_train::{
                PoissonNeuron, DeltaDiracRefractoriness,
            },
            plasticity::STDP,
            GridVoltageHistory, SpikeTrainGridHistory, Lattice, LatticeNetwork, RunLattice, RunNetwork
        }
    };

    #[test]
    pub fn test_run_lattice_size_zero_cpu() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let mut lattice: Lattice<
            QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            STDP, 
            IonotropicNeurotransmitterType,
        > = Lattice::default_impl();

        lattice.run_lattice(1000)?;
        
        Ok(())
    }

    #[test]
    pub fn test_run_lattice_size_zero_gpu() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let lattice: Lattice<
            QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            STDP, 
            IonotropicNeurotransmitterType,
        > = Lattice::default_impl();

        let mut gpu_lattice = LatticeGPU::from_lattice(lattice)?;

        gpu_lattice.run_lattice(1000)?;
        
        Ok(())
    }

    #[test]
    pub fn test_run_network_size_zero_cpu() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let mut network: LatticeNetwork<
            QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
            SpikeTrainGridHistory, 
            AdjacencyMatrix<GraphPosition, f32>, 
            STDP, 
            IonotropicNeurotransmitterType,
        > = LatticeNetwork::default_impl();

        network.run_lattices(1000)?;

        Ok(())
    }

    #[test]
    pub fn test_run_network_size_zero_gpu() -> Result<(), SpikingNeuralNetworksError> {
        #[allow(clippy::type_complexity)]
        let network: LatticeNetwork<
            QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
            SpikeTrainGridHistory, 
            AdjacencyMatrix<GraphPosition, f32>, 
            STDP, 
            IonotropicNeurotransmitterType,
        > = LatticeNetwork::default_impl();

        let mut gpu_network = LatticeNetworkGPU::from_network(network)?;

        gpu_network.run_lattices(1000)?;

        Ok(())
    }
}
