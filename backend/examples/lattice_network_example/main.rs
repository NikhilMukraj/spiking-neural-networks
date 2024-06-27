// extern crate spiking_neural_networks;
// use spiking_neural_networks::{graph::Graph, neuron::{
//     integrate_and_fire::IzhikevichNeuron, Lattice
// }};


// fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
//     x == y
// }

fn main() {
    // lattice network test
    // set up 3x3 poisson grid and 3x3 izhikevich grid
    // poisson should be presynaptic and connect to one other izhikevich neuron
    // izhikevich neuron should have no postsynaptic connections
    // test by first setting poisson firing rate to 0, neurons should not spike often
    // then set poisson firing rate to something higher, neurons should then spike more
    // if timestep == iterations / 2 change firing rate
    // izhikevich_lattice.update_history = true;
}
