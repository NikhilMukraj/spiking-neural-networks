use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    fs::File,
};
extern crate spiking_neural_networks;
use crate::spiking_neural_networks::{
    neuron::{
        integrate_and_fire::IzhikevichNeuron,
        iterate_and_spike::{
            IterateAndSpike, GaussianParameters, NeurotransmitterConcentrations,
            weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
        },
        update_weight_stdp, signed_gap_junction,
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

/// Updates each presynaptic neuron's weights given the timestep
/// and whether the neuron is spiking along with the state of the
/// postsynaptic neuron
fn update_isolated_presynaptic_neuron_weights<T: IterateAndSpike>(
    neurons: &mut Vec<T>,
    neuron: &T,
    weights: &mut Vec<f32>,
    delta_ws: &mut Vec<f32>,
    timestep: usize,
    is_spikings: Vec<bool>,
) {
    for (n, i) in is_spikings.iter().enumerate() {
        if *i {
            neurons[n].set_last_firing_time(Some(timestep));
            delta_ws[n] = update_weight_stdp(&neurons[n], &*neuron);
            weights[n] += delta_ws[n];
        }
    }
}

fn test_isolated_stdp<T: IterateAndSpike>(
    presynaptic_neurons: &mut Vec<T>,
    postsynaptic_neuron: &mut T,
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

    let mut delta_ws: Vec<f32> = (0..n)
        .map(|_| 0.0)
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
                        let output = weights[i] * signed_gap_junction(
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
            
        let presynaptic_neurotransmitters: NeurotransmitterConcentrations = if chemical_synapse {
            let neurotransmitters_vec = (0..n) 
                .map(|i| {
                    let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
                    weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

                    presynaptic_neurotransmitter
                }
            ).collect::<Vec<NeurotransmitterConcentrations>>();

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

        update_isolated_presynaptic_neuron_weights(
            presynaptic_neurons, 
            &postsynaptic_neuron,
            &mut weights, 
            &mut delta_ws, 
            timestep, 
            is_spikings,
        );

        if is_spiking {
            postsynaptic_neuron.set_last_firing_time(Some(timestep));
            for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
                delta_ws[n_neuron] = update_weight_stdp(i, postsynaptic_neuron);
                weights[n_neuron] += delta_ws[n_neuron];
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

    let output_hashmap = test_isolated_stdp(
        &mut presynaptic_neurons, 
        &mut izhikevich_neuron,
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
