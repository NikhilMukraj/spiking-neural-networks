extern crate spiking_neural_networks;
use rand::Rng;
use std::{fs::File, io::{BufWriter, Write}};
use spiking_neural_networks::{
    neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        plasticity::STDP,
        Lattice, LatticeNetwork, AverageVoltageHistory
    },
    graph::AdjacencyMatrix,
    error::SpikingNeuralNetworksError, 
};


/// Creates two pools of neurons, one inhibitory and one excitatory, and connects them,
/// writes the average voltage history over time of each pool to .csv files to the
/// current working directory
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = IzhikevichNeuron::default_impl();

    let mut inh_lattice: Lattice<_, AdjacencyMatrix<_, _>, AverageVoltageHistory, STDP, _> = Lattice::default();
    inh_lattice.populate(&base_neuron, 5, 5);
    inh_lattice.connect(&|x, y| x != y, Some(&|_, _| -1.));
    inh_lattice.apply(|n| {
        let mut rng = rand::thread_rng();
        n.current_voltage = rng.gen_range(n.v_init..=n.v_th);
    });
    inh_lattice.update_grid_history = true;

    let mut exc_lattice: Lattice<_, AdjacencyMatrix<_, _>, AverageVoltageHistory, STDP, _> = Lattice::default();
    exc_lattice.set_id(1);
    exc_lattice.populate(&base_neuron, 10, 10);
    exc_lattice.connect(&|x, y| x != y, Some(&|_, _| 1.));
    exc_lattice.apply(|n| {
        let mut rng = rand::thread_rng();
        n.current_voltage = rng.gen_range(n.v_init..=n.v_th);
    });
    exc_lattice.update_grid_history = true;

    let mut network = LatticeNetwork::default_impl();
    network.parallel = true;
    network.add_lattice(inh_lattice)?;
    network.add_lattice(exc_lattice)?;

    network.connect(0, 1, &|_, _| true, Some(&|_, _| -1.))?;
    network.connect(1, 0, &|_, _| true, None)?;

    network.run_lattices(5_000)?;

    let mut inh_voltage_file = BufWriter::new(File::create("inh_lattice_history.csv")
        .expect("Could not create file"));

    writeln!(inh_voltage_file, "voltages").expect("Could not write to file");
    for average in &network.get_lattice(&0).expect("Could not get lattice").grid_history.history {
        writeln!(inh_voltage_file, "{}", average).expect("Could not write to file");
    }

    let mut exc_voltage_file = BufWriter::new(File::create("exc_lattice_history.csv")
        .expect("Could not create file"));

    writeln!(exc_voltage_file, "voltages").expect("Could not write to file");
    for average in &network.get_lattice(&1).expect("Could not get lattice").grid_history.history {
        writeln!(exc_voltage_file, "{}", average).expect("Could not write to file");
    }

    Ok(())
}
