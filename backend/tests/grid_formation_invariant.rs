#[cfg(test)]
mod tests {
    use spiking_neural_networks::neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        spike_train::PoissonNeuron, 
        Lattice, SpikeTrainGrid, SpikeTrainLattice,
    };

    fn generate_grid<T: Clone>(base_neuron: &T, rows: usize, cols: usize) -> Vec<Vec<T>> {
        (0..rows).map(|_| 
            (0..cols).map(|_| base_neuron.clone())
                .collect()
        ).collect()
    }

    #[test]
    pub fn test_cell_grid_modification() {
        let base_neuron = IzhikevichNeuron::default_impl();

        let mut lattice = Lattice::default_impl();

        lattice.populate(&base_neuron, 3, 3);

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
    }

    #[test]
    pub fn test_spike_train_grid_modification() {
        let base_spike_train = PoissonNeuron::default_impl();

        let mut lattice = SpikeTrainLattice::default_impl();

        lattice.populate(&base_spike_train, 3, 3);

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
    }
}