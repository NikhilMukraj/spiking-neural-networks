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
        plasticity::{Plasticity, STDP},
        gap_junction,
    },
    distribution::limited_distr,
};



#[derive(Debug, Clone, Copy)]
pub struct Trace {
    c: f32,
    tau_c: f32,
    dt: f32,
}

impl Default for Trace {
    fn default() -> Self {
        Trace { c: 0., tau_c: 1000., dt: 0.1 }
    }
}

impl Trace {
    pub fn update_trace(&mut self, weight_change: f32) {
        // self.c = (self.c + weight_change) * (-self.dt / self.tau_c).exp();
        self.c = self.c * (-self.dt / self.tau_c).exp() + weight_change;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RewardParameters {
    tau_d: f32,
    dopamine: f32,
    dt: f32,
}

impl Default for RewardParameters {
    fn default() -> Self {
        RewardParameters { tau_d: 200., dopamine: 0., dt: 0.1 }
    }
}

impl RewardParameters {
    pub fn update_dopamine(&mut self, reward: f32) {
        // self.dopamine = (self.dopamine + reward) * (-self.dt / self.tau_d).exp();
        self.dopamine = self.dopamine * (-self.dt / self.tau_d).exp() + reward;
    }
}

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
    for i in 0..n {
        keys_vector.push(format!("trace_{}", i));
    }
    keys_vector.push(String::from("dopamine"));

    keys_vector
}

fn test_isolated_r_stdp<T: IterateAndSpike>(
    presynaptic_neurons: &mut Vec<T>,
    postsynaptic_neuron: &mut T,
    stdp_params: &STDP,
    iterations: usize,
    input_current: f32,
    input_current_deviation: f32,
    weight_params: &GaussianParameters,
    reward_params: &mut RewardParameters,
    reward_times: &[usize],
    reward: f32,
    trace_init: Trace,
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
   
    let mut traces: Vec<Trace> = (0..n).map(|_| 
            trace_init.clone()
        )
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

        if reward_times.contains(&timestep) {
            reward_params.update_dopamine(reward);
        } else {
            reward_params.update_dopamine(0.);
        }

        for (n, i) in is_spikings.iter().enumerate() {
            if *i {
                presynaptic_neurons[n].set_last_firing_time(Some(timestep));
                <STDP as Plasticity<T, T, f32>>::update_weight(
                    stdp_params, &mut weights[n], &presynaptic_neurons[n], &*postsynaptic_neuron
                );
            }
        }

        if is_spiking {
            postsynaptic_neuron.set_last_firing_time(Some(timestep));
            for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
                <STDP as Plasticity<T, T, f32>>::update_weight(
                    stdp_params, &mut weights[n_neuron], i, &*postsynaptic_neuron
                );
            }
        }

        for (index, trace) in traces.iter_mut().enumerate() {
            // trace.update_trace(delta_ws[index]);
            weights[index] += trace.c * reward_params.dopamine;
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
        for (index, i) in traces.iter().enumerate() {
            output_hashmap.get_mut(&format!("trace_{}", index))
                .expect("Could not find hashmap value")
                .push(i.c);
        }
        output_hashmap.get_mut("dopamine").expect("Could not find hashmap value")
            .push(reward_params.dopamine);
    }

    output_hashmap
}

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

    let mut reward_params = RewardParameters::default();
    let trace_init = Trace::default();
    let reward_times: &[usize] = &[4000, 8000];
    let reward = 0.001;

    let stdp_params = STDP::default();

    let output_hashmap = test_isolated_r_stdp(
        &mut presynaptic_neurons, 
        &mut izhikevich_neuron,
        &stdp_params,
        iterations, 
        30., 
        0.1, 
        &weight_params, 
        &mut reward_params,
        reward_times,
        reward,
        trace_init,
        true,
        false,
    );

    let keys_vector = generate_keys(presynaptic_neurons.len());
    let mut file = BufWriter::new(File::create("r-stdp.csv").expect("Could not create file"));

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
