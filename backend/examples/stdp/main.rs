use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    fs::File,
};
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    neuron::{
        iterate_and_spike::{ApproximateNeurotransmitter, NeurotransmitterType}, 
        spike_train::{DeltaDiracRefractoriness, PresetSpikeTrain}, 
        Lattice, LatticeNetwork, SpikeTrainGridHistory, SpikeTrainLattice
    },
    error::SpikingNeuralNetworksError, 
};

use crate::spiking_neural_networks::{
    neuron::{
        integrate_and_fire::IzhikevichNeuron,
        iterate_and_spike::{
            IterateAndSpike, GaussianParameters, NeurotransmitterConcentrations,
            weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
        },
        plasticity::{Plasticity, STDP},
        gap_junction,
    },
    distribution::limited_distr,
};


/// Generates keys in an ordered manner to ensure columns in file are ordered
fn generate_keys(n: usize) -> Vec<String> {
    let mut keys_vector: Vec<String> = vec![];

    for i in 0..n {
        keys_vector.push(format!("presynaptic_voltage_{}", i))
    }
    keys_vector.push(String::from("postsynaptic_voltage"));
    for i in 0..n {
        keys_vector.push(format!("weight_{}", i));
    }

    keys_vector
}

fn test_isolated_stdp<T: IterateAndSpike>(
    presynaptic_neurons: &mut Vec<T>,
    postsynaptic_neuron: &mut T,
    stdp_params: &STDP,
    iterations: usize,
    input_current: f32,
    input_current_deviation: f32,
    weight_params: &GaussianParameters,
    electrical_synapse: bool,
    chemical_synapse: bool,
) -> HashMap<String, Vec<f32>> {
    let n = presynaptic_neurons.len();

    let input_currents: Vec<f32> = (0..n).map(|_| 
            input_current * limited_distr(1.0, input_current_deviation, 0., 2.)
        )
        .collect();

    let mut weights: Vec<f32> = (0..n).map(|_| weight_params.get_random_number())
        .collect();

    let mut output_hashmap: HashMap<String, Vec<f32>> = HashMap::new();
    let keys_vector = generate_keys(n);
    for i in keys_vector.iter() {
        output_hashmap.insert(String::from(i), vec![]);
    }

    for timestep in 0..iterations {
        let calculated_current: f32 = if electrical_synapse { 
            (0..n).map(
                    |i| {
                        let output = weights[i] * gap_junction(
                            &presynaptic_neurons[i], 
                            &*postsynaptic_neuron
                        );

                        output / (n as f32)
                    }
                ) 
                .collect::<Vec<f32>>()
                .iter()
                .sum()
            } else {
                0.
            };
            
        let presynaptic_neurotransmitters: NeurotransmitterConcentrations<T::N> = if chemical_synapse {
            let neurotransmitters_vec = (0..n) 
                .map(|i| {
                    let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
                    weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

                    presynaptic_neurotransmitter
                }
            ).collect::<Vec<NeurotransmitterConcentrations<T::N>>>();

            let mut neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);

            weight_neurotransmitter_concentration(&mut neurotransmitters, (1 / n) as f32); 

            neurotransmitters
        } else {
            HashMap::new()
        };
        
        let presynaptic_inputs: Vec<f32> = (0..n)
            .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
            .collect();
        let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
            .map(|(presynaptic_neuron, input_value)| {
                presynaptic_neuron.iterate_and_spike(*input_value)
            })
            .collect();
        let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
            calculated_current,
            &presynaptic_neurotransmitters,
        );

        for (n, i) in is_spikings.iter().enumerate() {
            if *i {
                presynaptic_neurons[n].set_last_firing_time(Some(timestep));
                <STDP as Plasticity<T, T, f32>>::update_weight(
                    stdp_params, 
                    &mut weights[n],
                    &presynaptic_neurons[n], 
                    postsynaptic_neuron
                );
            }
        }

        if is_spiking {
            postsynaptic_neuron.set_last_firing_time(Some(timestep));
            for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
                <STDP as Plasticity<T, T, f32>>::update_weight(
                    stdp_params, 
                    &mut weights[n_neuron],
                    i, 
                    postsynaptic_neuron
                );
            }
        }

        for (index, i) in presynaptic_neurons.iter().enumerate() {
            output_hashmap.get_mut(&format!("presynaptic_voltage_{}", index))
                .expect("Could not find hashmap value")
                .push(i.get_current_voltage());
        }
        output_hashmap.get_mut("postsynaptic_voltage").expect("Could not find hashmap value")
            .push(postsynaptic_neuron.get_current_voltage());
        for (index, i) in weights.iter().enumerate() {
            output_hashmap.get_mut(&format!("weight_{}", index))
                .expect("Could not find hashmap value")
                .push(*i);
        }
    }

    output_hashmap
}

pub fn test_stdp<N, T>(
    firing_rates: &[f32],
    postsynaptic_neuron: &T,
    iterations: usize,
    stdp_params: &STDP,
    weight_params: &GaussianParameters,
    electrical_synapse: bool,
    chemical_synapse: bool,
) -> Result<(), SpikingNeuralNetworksError>
where
    N: NeurotransmitterType,
    T: IterateAndSpike<N=N>,
{
    type SpikeTrainType<N> = PresetSpikeTrain<N, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;

    let mut spike_train_lattice: SpikeTrainLattice<
        N, 
        SpikeTrainType<N>, 
        SpikeTrainGridHistory,
    > = SpikeTrainLattice::default();
    let preset_spike_train = PresetSpikeTrain::default();
    spike_train_lattice.populate(&preset_spike_train, firing_rates.len(), 1);
    spike_train_lattice.apply_given_position(
        &(|pos: (usize, usize), spike_train: &mut SpikeTrainType<N>| { 
            spike_train.firing_times = vec![firing_rates[pos.0]]; 
        })
    );
    spike_train_lattice.update_grid_history = true;
    spike_train_lattice.set_id(0);

    let mut lattice = Lattice::default_impl();
    lattice.populate(&postsynaptic_neuron.clone(), 1, 1);
    lattice.plasticity = *stdp_params;
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
            .or_insert_with(Vec::new)
            .extend(spike_train_history.iter().map(|step| step[0][i]).collect::<Vec<f32>>());
    }

    // return this hashmap above
    // and write the weights to a seperate file

    // let keys = generate_keys(firing_rates.len());

    // let mut file = BufWriter::new(File::create("stdp.csv").expect("Could not create file"));

    // for (n, i) in keys.iter().enumerate() {
    //     if n != keys.len() - 1 {
    //         write!(file, "{},", i).expect("Could not write to file");
    //     } else {
    //         writeln!(file, "{}", i).expect("Could not write to file");
    //     }
    // }
    // for i in 0..iterations {
    //     for (n, key) in keys_vector.iter().enumerate() {
    //         if n != keys_vector.len() - 1 {
    //             write!(file, "{},", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
    //                 .expect("Could not write to file");
    //         } else {
    //             writeln!(file, "{}", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
    //                 .expect("Could not write to file");
    //         }
    //     }
    // }

    Ok(())
}

// - Generates a set of presynaptic neurons and a postsynaptic neuron (Izhikevich)
// - Couples presynaptic and postsynaptic neurons
// - Sets input to presynaptic neurons as a static current and input to postsynaptic neuron
// as a weighted input from the presynaptic neurons
// - Updates weights based on spike time dependent plasticity when spiking occurs
// - Writes the history of the simulation to working directory
fn main() {
    let mut izhikevich_neuron = IzhikevichNeuron {
        c_m: 50.,
        gap_conductance: 1.,
        ..IzhikevichNeuron::default_impl()
    };

    let mut presynaptic_neurons = vec![izhikevich_neuron.clone(), izhikevich_neuron.clone()];

    let iterations = 10000;

    let weight_params = GaussianParameters {
        mean: 1.5,
        std: 0.1,
        min: 1.,
        max: 2.,
    };

    let stdp_params = STDP::default();

    let output_hashmap = test_isolated_stdp(
        &mut presynaptic_neurons, 
        &mut izhikevich_neuron,
        &stdp_params,
        iterations, 
        30., 
        0.1, 
        &weight_params, 
        true,
        false,
    );

    // could have option to directly plot in terminal

    let keys_vector = generate_keys(presynaptic_neurons.len());
    let mut file = BufWriter::new(File::create("stdp.csv").expect("Could not create file"));

    for (n, i) in keys_vector.iter().enumerate() {
        if n != keys_vector.len() - 1 {
            write!(file, "{},", i).expect("Could not write to file");
        } else {
            writeln!(file, "{}", i).expect("Could not write to file");
        }
    }
    for i in 0..iterations {
        for (n, key) in keys_vector.iter().enumerate() {
            if n != keys_vector.len() - 1 {
                write!(file, "{},", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
                    .expect("Could not write to file");
            } else {
                writeln!(file, "{}", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
                    .expect("Could not write to file");
            }
        }
    }
}
