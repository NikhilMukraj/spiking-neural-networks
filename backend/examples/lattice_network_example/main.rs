use std::result::Result;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        Lattice, LatticeNetwork, SpikeTrainLattice,
        integrate_and_fire::IzhikevichNeuron, 
        spike_train::PoissonNeuron,
    }
};


fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    x == y
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    // lattice network test
    // set up 3x3 poisson grid and 3x3 izhikevich grid
    // poisson should be presynaptic and connect to one other izhikevich neuron
    // izhikevich neuron should have no postsynaptic connections
    // test by first setting poisson firing rate to 0, neurons should not spike often
    // then set poisson firing rate to something higher, neurons should then spike more
    // if timestep == iterations / 2 change firing rate
    // izhikevich_lattice.update_history = true;

    let (num_rows, num_cols) = (3, 3);

    let izhikevich_neuron = IzhikevichNeuron::default_impl();
    let poisson_neuron = PoissonNeuron::default_impl();

    let mut spike_train_lattice = SpikeTrainLattice::default_impl();
    spike_train_lattice.populate(&poisson_neuron, num_rows, num_cols);

    let mut lattice = Lattice::default_impl();
    lattice.populate(&izhikevich_neuron, num_rows, num_cols);
    lattice.set_id(1);

    let mut network = LatticeNetwork::generate_network(vec![lattice], vec![spike_train_lattice])?;

    network.connect(0, 1, connection_conditional, None)?;

    Ok(())
}
