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
            ApproximateNeurotransmitter, ApproximateReceptor,
            weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
        },
        update_weight_stdp, signed_gap_junction,
    },
    distribution::limited_distr,
};


fn update_isolated_presynaptic_neuron_weights<T: IterateAndSpike>(
    neurons: &mut Vec<T>,
    neuron: &T,
    weights: &mut Vec<f64>,
    delta_ws: &mut Vec<f64>,
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
    input_current: f64,
    input_current_deviation: f64,
    weight_params: &GaussianParameters,
    do_receptor_kinetics: bool,
) -> HashMap<String, Vec<f64>> {
    let n = presynaptic_neurons.len();

    let input_currents: Vec<f64> = (0..n).map(|_| 
            input_current * limited_distr(1.0, input_current_deviation, 0., 2.)
        )
        .collect();

    let mut weights: Vec<f64> = (0..n).map(
        |_| limited_distr(
            weight_params.mean, 
            weight_params.std, 
            weight_params.min, 
            weight_params.max,
        )
    ).collect();

    let mut delta_ws: Vec<f64> = (0..n)
        .map(|_| 0.0)
        .collect();

    let mut output_hashmap: HashMap<String, Vec<f64>> = HashMap::new();
    for i in 0..n {
        output_hashmap.insert(format!("presynaptic_neuron_{}", i), vec![]);
    }
    output_hashmap.insert(String::from("postsynaptic_voltage"), vec![]);
    for i in 0..n {
        output_hashmap.insert(format!("weight_{}", i), vec![]);
    }

    for timestep in 0..iterations {
        let calculated_voltage: f64 = (0..n)
            .map(
                |i| {
                    let output = weights[i] * signed_gap_junction(&presynaptic_neurons[i], &*postsynaptic_neuron);

                    output / (n as f64)
                }
            ) 
            .collect::<Vec<f64>>()
            .iter()
            .sum();
        let presynaptic_neurotransmitters: Option<NeurotransmitterConcentrations> = match do_receptor_kinetics {
            true => Some({
                let neurotransmitters_vec = (0..n) 
                    .map(|i| {
                        let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
                        weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

                        presynaptic_neurotransmitter
                    }
                ).collect::<Vec<NeurotransmitterConcentrations>>();

                let mut neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);

                weight_neurotransmitter_concentration(&mut neurotransmitters, (1 / n) as f64); 

                neurotransmitters
            }),
            false => None
        };
        
        let noise_factor = postsynaptic_neuron.get_gaussian_factor();
        let presynaptic_inputs: Vec<f64> = (0..n)
            .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
            .collect();
        let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
            .map(|(presynaptic_neuron, input_value)| {
                presynaptic_neuron.iterate_and_spike(*input_value)
            })
            .collect();
        let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
            noise_factor * calculated_voltage,
            presynaptic_neurotransmitters.as_ref(),
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
            output_hashmap.get_mut(&format!("presynaptic_neuron_{}", index))
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

type IzhikevichApproximateKinetics = IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;

fn main() {
    let mut izhikevich_neuron: IzhikevichApproximateKinetics  = IzhikevichNeuron::default();

    let mut presynaptic_neurons = vec![izhikevich_neuron.clone(), izhikevich_neuron.clone()];

    let iterations = 10000;

    let weight_params = GaussianParameters {
        mean: 1.5,
        std: 0.3,
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
        false,
    );

    let mut file = BufWriter::new(File::create("stdp.csv").expect("Could not create file"));

    for i in output_hashmap.keys() {
        write!(file, "{},", i).expect("Could not write to file");
    }
    writeln!(file).expect("Could not write to file");
    for i in 0..iterations {
        for key in output_hashmap.keys() {
            write!(file, "{},", output_hashmap.get(key).expect("Cannot find hashmap value")[i])
                .expect("Could not write to file");
        }
        writeln!(file).expect("Could not write to file");
    }
}

// plot weights over time
