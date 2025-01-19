#[cfg(test)]
mod tests {
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::{AdjacencyMatrix, GraphPosition}, 
        neuron::{
            integrate_and_fire::IzhikevichNeuron, 
            iterate_and_spike::{
                ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType
            }, 
            plasticity::STDP, 
            spike_train::PoissonNeuron, 
            GridVoltageHistory, Lattice, LatticeNetwork, SpikeTrainGrid, 
            SpikeTrainGridHistory, SpikeTrainLattice
        }
    };

    fn generate_grid<T: Clone>(base_neuron: &T, rows: usize, cols: usize) -> Vec<Vec<T>> {
        (0..rows).map(|_| 
            (0..cols).map(|_| base_neuron.clone())
                .collect()
        ).collect()
    }

    #[test]
    pub fn test_cell_grid_modification() -> Result<(), SpikingNeuralNetworksError> {
        let base_neuron = IzhikevichNeuron::default_impl();

        let mut lattice = Lattice::default_impl();

        lattice.populate(&base_neuron, 3, 3)?;

        let new_cell_grid: Vec<Vec<IzhikevichNeuron<_, _>>> = generate_grid(&base_neuron, 4, 4);

        assert!(lattice.set_cell_grid(new_cell_grid).is_err());
        assert!(lattice.cell_grid().len() == 3);
        assert!(lattice.cell_grid().iter().all(|i| i.len() == 3));

        let new_cell_grid: Vec<Vec<IzhikevichNeuron<_, _>>> = generate_grid(&base_neuron, 2, 2);

        assert!(lattice.set_cell_grid(new_cell_grid).is_err());
        assert!(lattice.cell_grid().len() == 3);
        assert!(lattice.cell_grid().iter().all(|i| i.len() == 3));

        let new_cell_grid: Vec<Vec<IzhikevichNeuron<_, _>>> = generate_grid(&base_neuron, 6, 2);

        assert!(lattice.set_cell_grid(new_cell_grid).is_err());
        assert!(lattice.cell_grid().len() == 3);
        assert!(lattice.cell_grid().iter().all(|i| i.len() == 3));

        let new_cell_grid: Vec<Vec<IzhikevichNeuron<_, _>>> = generate_grid(&base_neuron, 3, 3);

        assert!(lattice.set_cell_grid(new_cell_grid).is_ok());
        assert!(lattice.cell_grid().len() == 3);
        assert!(lattice.cell_grid().iter().all(|i| i.len() == 3));

        Ok(())
    }

    #[test]
    pub fn test_spike_train_grid_modification() -> Result<(), SpikingNeuralNetworksError> {
        let base_spike_train = PoissonNeuron::default_impl();

        let mut lattice = SpikeTrainLattice::default_impl();

        lattice.populate(&base_spike_train, 3, 3)?;

        let new_spike_train_grid: Vec<Vec<_>> = generate_grid(&base_spike_train, 4, 4);

        assert!(lattice.set_spike_train_grid(new_spike_train_grid).is_err());
        assert!(lattice.spike_train_grid().len() == 3);
        assert!(lattice.spike_train_grid().iter().all(|i| i.len() == 3));

        let new_spike_train_grid: Vec<Vec<_>> = generate_grid(&base_spike_train, 2, 2);

        assert!(lattice.set_spike_train_grid(new_spike_train_grid).is_err());
        assert!(lattice.spike_train_grid().len() == 3);
        assert!(lattice.spike_train_grid().iter().all(|i| i.len() == 3));

        let new_spike_train_grid: Vec<Vec<_>> = generate_grid(&base_spike_train, 6, 2);

        assert!(lattice.set_spike_train_grid(new_spike_train_grid).is_err());
        assert!(lattice.spike_train_grid().len() == 3);
        assert!(lattice.spike_train_grid().iter().all(|i| i.len() == 3));

        let new_spike_train_grid: Vec<Vec<_>> = generate_grid(&base_spike_train, 3, 3);

        assert!(lattice.set_spike_train_grid(new_spike_train_grid).is_ok());
        assert!(lattice.spike_train_grid().len() == 3);
        assert!(lattice.spike_train_grid().iter().all(|i| i.len() == 3));

        Ok(())
    }

    #[test]
    pub fn test_modification_of_cell_grid_in_network() -> Result<(), SpikingNeuralNetworksError> {
        let base_neuron = IzhikevichNeuron::default_impl();

        let mut lattice = Lattice::default_impl();

        lattice.populate(&base_neuron, 3, 3)?;

        let mut network = LatticeNetwork::default_impl();
        network.add_lattice(lattice)?;

        assert!(network.get_mut_lattice(&0).unwrap().populate(&base_neuron, 3, 3).is_ok());
        assert!(network.get_mut_lattice(&0).unwrap().populate(&base_neuron, 4, 3).is_err());
        assert!(network.get_mut_lattice(&0).unwrap().populate(&base_neuron, 3, 4).is_err());
        assert!(network.get_mut_lattice(&0).unwrap().populate(&base_neuron, 2, 4).is_err());

        assert!(network.get_lattice(&0).unwrap().clone().populate(&base_neuron, 2, 2).is_ok());

        Ok(())
    }

    #[test]
    pub fn test_modification_of_spike_train_grid_in_network() -> Result<(), SpikingNeuralNetworksError> {
        let base_spike_train = PoissonNeuron::default_impl();

        let mut lattice = SpikeTrainLattice::default_impl();

        lattice.populate(&base_spike_train, 3, 3)?;

        #[allow(clippy::type_complexity)]
        let mut network: LatticeNetwork<
            IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            _, 
            SpikeTrainGridHistory, 
            AdjacencyMatrix<GraphPosition, f32>, 
            STDP, 
            IonotropicNeurotransmitterType,
        > = LatticeNetwork::default_impl();
        network.add_spike_train_lattice(lattice)?;

        assert!(network.get_mut_spike_train_lattice(&0).unwrap().populate(&base_spike_train, 3, 3).is_ok());
        assert!(network.get_mut_spike_train_lattice(&0).unwrap().populate(&base_spike_train, 4, 3).is_err());
        assert!(network.get_mut_spike_train_lattice(&0).unwrap().populate(&base_spike_train, 3, 4).is_err());
        assert!(network.get_mut_spike_train_lattice(&0).unwrap().populate(&base_spike_train, 2, 4).is_err());

        assert!(network.get_spike_train_lattice(&0).unwrap().clone().populate(&base_spike_train, 2, 2).is_ok());

        Ok(())
    }
}