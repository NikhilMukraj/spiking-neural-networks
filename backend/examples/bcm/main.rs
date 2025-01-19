use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    fs::File,
};
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    neuron::{
        integrate_and_fire::BCMIzhikevichNeuron, iterate_and_spike::{
            ApproximateNeurotransmitter, GaussianParameters, IterateAndSpike, NeurotransmitterType
        }, 
        plasticity::{BCMActivity, BCM}, 
        spike_train::{BCMPoissonNeuron, DeltaDiracRefractoriness}, 
        GridVoltageHistory, Lattice, LatticeNetwork, RunNetwork, SpikeTrainGridHistory, SpikeTrainLattice
    },
    graph::AdjacencyMatrix, 
    error::SpikingNeuralNetworksError,
};


/// Generates keys in an ordered manner to ensure columns in file are ordered
fn generate_keys(n: usize) -> Vec<String> {
    let mut keys_vector: Vec<String> = vec![];

    for i in 0..n {
        keys_vector.push(format!("presynaptic_voltage_{}", i))
    }
    keys_vector.push(String::from("postsynaptic_voltage"));

    keys_vector
}

type VoltageHashMap = HashMap<String, Vec<f32>>;
type Matrix = Vec<Vec<Vec<Option<f32>>>>;

/// Tests BCM dynamics over time given a set of input firing rates to a postsynaptic neuron
/// and updates the weights between the spike trains and given postsynaptic neuron, returns
/// the voltage and weight history over tim
pub fn test_bcm<N, T>(
    firing_rates: &[f32],
    postsynaptic_neuron: &T,
    iterations: usize,
    bcm_params: &BCM,
    weight_params: &GaussianParameters,
    electrical_synapse: bool,
    chemical_synapse: bool,
) -> Result<(VoltageHashMap, Matrix), SpikingNeuralNetworksError>
where
    N: NeurotransmitterType,
    T: IterateAndSpike<N=N> + BCMActivity,
{
    type SpikeTrainType<N> = BCMPoissonNeuron<N, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;

    // setup presynaptic inputs
    let mut spike_train_lattice: SpikeTrainLattice<
        N, 
        SpikeTrainType<N>, 
        SpikeTrainGridHistory,
    > = SpikeTrainLattice::default();
    let preset_spike_train = BCMPoissonNeuron::default();
    spike_train_lattice.populate(&preset_spike_train, firing_rates.len(), 1)?;
    spike_train_lattice.apply_given_position(
        |pos: (usize, usize), spike_train: &mut SpikeTrainType<N>| { 
            spike_train.chance_of_firing = firing_rates[pos.0]; 
        }
    );
    spike_train_lattice.update_grid_history = true;
    spike_train_lattice.set_id(0);

    // setup postsynaptic neuron
    let mut lattice: Lattice<T, AdjacencyMatrix<_, _>, GridVoltageHistory, BCM, N> = Lattice::default();
    lattice.populate(&postsynaptic_neuron.clone(), 1, 1)?;
    lattice.plasticity = *bcm_params;
    lattice.do_plasticity = true;
    lattice.update_grid_history = true;
    lattice.set_id(1);

    let lattices = vec![lattice];
    let spike_train_lattices = vec![spike_train_lattice];
    let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
    network.connect(0, 1, &(|_, _| true), Some(&(|_, _| weight_params.get_random_number())))?;
    network.update_connecting_graph_history = true;
    network.electrical_synapse = electrical_synapse;
    network.chemical_synapse = chemical_synapse;

    // run plasticity test
    network.run_lattices(iterations)?;

    // track postsynaptic voltage over time
    // track spike trains over time
    // track weights over time

    let mut output_hashmap: HashMap<String, Vec<f32>> = HashMap::new();
    output_hashmap.insert(
        String::from("postsynaptic_voltage"),
        network.get_lattice(&1).unwrap().grid_history
            .history
            .iter()
            .map(|i| i[0][0])
            .collect(),
    );
    let spike_train_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
    for i in 0..firing_rates.len() {
        output_hashmap
            .entry(format!("presynaptic_voltage_{}", i))
            .or_default()
            .extend(spike_train_history.iter().map(|step| step[i][0]).collect::<Vec<f32>>());
    }

    Ok((output_hashmap, network.get_connecting_graph().history.clone()))
}

// - Generates a set of presynaptic spike trains and a postsynaptic neuron (Izhikevich)
// - Couples presynaptic and postsynaptic spike trains that fire regularly
// - Updates weights based on spike time dependent plasticity when spiking occurs
// - Writes the history of the simulation to working directory
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let izhikevich_neuron = BCMIzhikevichNeuron {
        c_m: 50.,
        gap_conductance: 5.,
        ..BCMIzhikevichNeuron::default_impl()
    };

    let firing_times = vec![0.0025, 0.00125];

    let iterations = 10000;

    let weight_params = GaussianParameters {
        mean: 1.5,
        std: 0.1,
        min: 1.,
        max: 2.,
    };

    let bcm_params = BCM::default();

    let (output_hashmap, weight_history) = test_bcm(
        &firing_times, 
        &izhikevich_neuron,
        iterations, 
        &bcm_params,
        &weight_params, 
        true,
        false,
    )?;

    // could have option to directly plot in terminal

    let keys_vector = generate_keys(firing_times.len());
    let mut voltages_file = BufWriter::new(File::create("voltages.csv").expect("Could not create file"));

    for (n, i) in keys_vector.iter().enumerate() {
        if n != keys_vector.len() - 1 {
            write!(voltages_file, "{},", i).expect("Could not write to file");
        } else {
            writeln!(voltages_file, "{}", i).expect("Could not write to file");
        }
    }
    for i in 0..iterations {
        for (n, key) in keys_vector.iter().enumerate() {
            if n != keys_vector.len() - 1 {
                write!(voltages_file, "{},", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
                    .expect("Could not write to file");
            } else {
                writeln!(voltages_file, "{}", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
                    .expect("Could not write to file");
            }
        }
    }

    let mut weights_file = BufWriter::new(File::create("weights.txt").expect("Could not create file"));

    for matrix in weight_history.iter() {
        for row in matrix {
            for value in row {
                match value {
                    Some(weight) => write!(weights_file, "{},", weight).expect("Could not write to file"),
                    None => write!(weights_file, "0,").expect("Could not write to file"),
                }
            }
            
            writeln!(weights_file).expect("Could not write to file");
        }

        writeln!(weights_file, "-----").expect("Could not write to file");
    }

    Ok(())
}
