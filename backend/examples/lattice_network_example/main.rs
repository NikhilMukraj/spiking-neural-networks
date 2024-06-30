use std::{
    result::Result,
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        Lattice, LatticeNetwork, SpikeTrainLattice,
        integrate_and_fire::IzhikevichNeuron, 
        spike_train::PoissonNeuron,
    },
};


fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    x == y
}

/// Lattice network test
/// - Set up 3x3 poisson grid and 3x3 izhikevich grid
/// - Poisson should be presynaptic and connect to one other izhikevich neuron
/// - Izhikevich neuron should have no postsynaptic connections
/// - First set Poisson firing rate to 0, neurons should not spike often
/// - Then set poisson firing rate to something higher, neurons should then spike more
/// - Write history to working directory
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let (num_rows, num_cols) = (3, 3);

    let mut izhikevich_neuron = IzhikevichNeuron::default_impl();
    izhikevich_neuron.gap_conductance = 10.;
    let mut poisson_neuron = PoissonNeuron::default_impl();
    poisson_neuron.chance_of_firing = 0.;

    let mut spike_train_lattice = SpikeTrainLattice::default_impl();
    spike_train_lattice.set_id(0);
    spike_train_lattice.populate(&poisson_neuron, num_rows, num_cols);
    spike_train_lattice.update_grid_history = true;

    let mut lattice = Lattice::default_impl();
    lattice.set_id(1);
    lattice.populate(&izhikevich_neuron, num_rows, num_cols);
    lattice.update_grid_history = true;

    let mut network = LatticeNetwork::generate_network(vec![lattice], vec![spike_train_lattice])?;

    network.connect(0, 1, connection_conditional, None)?;

    let iterations = 2500;

    network.run_lattices(iterations)?;

    network.get_mut_spike_train_lattice(&0).unwrap().cell_grid
        .iter_mut()
        .for_each(|i| {
            i.iter_mut()
                .for_each(|j| {
                    j.chance_of_firing = 0.01
            })
        });
    
    network.run_lattices(iterations)?;

    let mut voltage_file = BufWriter::new(File::create("lattice_network_history.txt")
        .expect("Could not create file"));

    for grid in &network.get_lattice(&1).unwrap().grid_history.history {
        for row in grid {
            for value in row {
                write!(voltage_file, "{} ", value)
                    .expect("Could not write to file");
            }
            writeln!(voltage_file)
                .expect("Could not write to file");
        }
        writeln!(voltage_file, "-----")
            .expect("Could not write to file"); 
    }

    let mut spike_train_file = BufWriter::new(File::create("spike_train_history.txt")
        .expect("Could not create file"));

    for grid in &network.get_spike_train_lattice(&0).unwrap().grid_history.history {
        for row in grid {
            for value in row {
                write!(spike_train_file, "{} ", value)
                    .expect("Could not write to file");
            }
            writeln!(spike_train_file)
                .expect("Could not write to file");
        }
        writeln!(spike_train_file, "-----")
            .expect("Could not write to file"); 
    }

    Ok(())
}
