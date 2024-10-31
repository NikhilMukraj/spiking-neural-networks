

#[cfg(test)]
mod tests {
    extern crate spiking_neural_networks;
    use spiking_neural_networks::graph::AdjacencyMatrix;
    use spiking_neural_networks::neuron::iterate_and_spike::IonotropicNeurotransmitterType;
    use spiking_neural_networks::neuron::plasticity::STDP;
    use spiking_neural_networks::neuron::{
        integrate_and_fire::IzhikevichNeuron,
        spike_train::PoissonNeuron,
        SpikeTrainLattice, Lattice, LatticeNetwork, SpikeHistory
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;
    

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        x == y
    }

    fn get_history_from_example(
        num_rows: usize,
        num_cols: usize,
        iterations: usize,
        electrical_synapse: bool,
        chemical_synapse: bool,
    ) -> Result<SpikeHistory, SpikingNeuralNetworksError> {
        let mut izhikevich_neuron = IzhikevichNeuron::default_impl();
        izhikevich_neuron.gap_conductance = 10.;
        let mut poisson_neuron = PoissonNeuron::default_impl();
        poisson_neuron.chance_of_firing = 0.;
    
        let mut spike_train_lattice = SpikeTrainLattice::default_impl();
        spike_train_lattice.set_id(0);
        spike_train_lattice.populate(&poisson_neuron, num_rows, num_cols);
        spike_train_lattice.update_grid_history = true;
    
        let mut lattice = Lattice::default();
        lattice.set_id(1);
        lattice.populate(&izhikevich_neuron, num_rows, num_cols);
        lattice.update_grid_history = true;
    
        let lattices: Vec<
            Lattice<_, AdjacencyMatrix<_, _>, SpikeHistory, STDP, IonotropicNeurotransmitterType>
        > = vec![lattice];
        let spike_train_lattices = vec![spike_train_lattice];
        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
    
        network.connect(0, 1, &connection_conditional, None)?;
        network.parallel = true;
        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        network.run_lattices(iterations)?;
    
        network.get_mut_spike_train_lattice(&0).unwrap().apply(|neuron| {
            neuron.chance_of_firing = 0.01;
        });
        
        network.run_lattices(iterations)?;

        Ok(network.get_lattice(&1).unwrap().grid_history.clone())
    }
    
    // fn count_nonzero_in_range(
    //     data: &Vec<Vec<Vec<usize>>>, 
    //     start: usize, 
    //     end: usize
    // ) -> usize {
    //     let bounded_end = end.min(data.len());
    
    //     data[start..bounded_end].iter()
    //         .flat_map(|inner_vec| inner_vec.iter())
    //         .flat_map(|innermost_vec| innermost_vec.iter())
    //         .filter(|&&value| value != 0)
    //         .count()
    // }    

    #[test]
    pub fn test_electrical_synapse_input() -> Result<(), SpikingNeuralNetworksError> {
        println!("{:#?}", get_history_from_example(3, 3, 2500, true, false));

        // check that before 2500 it is <=1, then after it is >= 1

        Ok(())
    }

    #[test]
    pub fn test_chemical_synapse_input() -> Result<(), SpikingNeuralNetworksError> {
        println!("{:#?}", get_history_from_example(3, 3, 2500, false, true));

        // check that before 2500 it is <=1, then after it is >= 1

        Ok(())
    }
}
