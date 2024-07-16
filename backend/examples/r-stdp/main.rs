// use std::{
//     collections::HashMap,
//     io::{BufWriter, Write},
//     fs::File,
// };
// extern crate spiking_neural_networks;
// use crate::spiking_neural_networks::{
//     neuron::{
//         integrate_and_fire::IzhikevichNeuron,
//         iterate_and_spike::{
//             IterateAndSpike, GaussianParameters, NeurotransmitterConcentrations,
//             weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
//         },
//         update_weight_stdp, gap_junction,
//     },
//     distribution::limited_distr,
// };



// pub struct Trace {
//     c: f32,
//     tau_c: f32,
//     dt: f32,
// }

// impl Default for Trace {
//     fn default() -> Self {
//         Trace { c: 0., tau_c: 0.01, dt: 0.1 }
//     }
// }

// impl Trace {
//     pub fn update_trace(&mut self, weight_change: f32) {
//         self.c += (-self.c / self.tau_c + weight_change) * self.dt;
//     }
// }

// pub struct RewardParameters {
//     tau_d: f32,
//     dopamine: f32,
//     dt: f32,
// }

// impl Default for RewardParameters {
//     fn default() -> Self {
//         RewardParameters { tau_d: 0.002, dopamine: 0., dt: 0.1 }
//     }
// }

// impl RewardParameters {
//     pub fn update_dopamine(&mut self, reward: f32) {
//         self.dopamine += (-self.dopamine / self.tau_d + reward) * self.dt; 
//     }
// }

// /// Generates keys in an ordered manner to ensure columns in file are ordered
// fn generate_keys(n: usize) -> Vec<String> {
//     let mut keys_vector: Vec<String> = vec![];

//     for i in 0..n {
//         keys_vector.push(format!("presynaptic_voltage_{}", i))
//     }
//     keys_vector.push(String::from("postsynaptic_voltage"));
//     for i in 0..n {
//         keys_vector.push(format!("weight_{}", i));
//     }

//     keys_vector
// }

// fn test_isolated_stdp<T: IterateAndSpike>(
//     presynaptic_neurons: &mut Vec<T>,
//     postsynaptic_neuron: &mut T,
//     iterations: usize,
//     input_current: f32,
//     input_current_deviation: f32,
//     weight_params: &GaussianParameters,
//     reward_params: &mut RewardParameters,
//     reward_times: &[usize],
//     reward: f32,
//     electrical_synapse: bool,
//     chemical_synapse: bool,
// ) -> HashMap<String, Vec<f32>> {
//     let n = presynaptic_neurons.len();

//     let input_currents: Vec<f32> = (0..n).map(|_| 
//             input_current * limited_distr(1.0, input_current_deviation, 0., 2.)
//         )
//         .collect();

//     let mut weights: Vec<f32> = (0..n).map(|_| weight_params.get_random_number())
//         .collect();
//     let c_init = 1.;
//     let tau_c = 0.01;
//     let trace_dt = 0.1;
//     let mut traces: Vec<Trace> = (0..n).map(|_| 
//             Trace { c: c_init, tau_c: tau_c, dt: trace_dt }
//         )
//         .collect();

//     let mut output_hashmap: HashMap<String, Vec<f32>> = HashMap::new();
//     let keys_vector = generate_keys(n);
//     for i in keys_vector.iter() {
//         output_hashmap.insert(String::from(i), vec![]);
//     }

//     for timestep in 0..iterations {
//         let mut delta_ws: Vec<f32> = (0..n)
//             .map(|_| 0.0)
//             .collect();

//         let calculated_current: f32 = if electrical_synapse { 
//             (0..n).map(
//                     |i| {
//                         let output = weights[i] * gap_junction(
//                             &presynaptic_neurons[i], 
//                             &*postsynaptic_neuron
//                         );

//                         output / (n as f32)
//                     }
//                 ) 
//                 .collect::<Vec<f32>>()
//                 .iter()
//                 .sum()
//             } else {
//                 0.
//             };
            
//         let presynaptic_neurotransmitters: NeurotransmitterConcentrations = if chemical_synapse {
//             let neurotransmitters_vec = (0..n) 
//                 .map(|i| {
//                     let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
//                     weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

//                     presynaptic_neurotransmitter
//                 }
//             ).collect::<Vec<NeurotransmitterConcentrations>>();

//             let mut neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);

//             weight_neurotransmitter_concentration(&mut neurotransmitters, (1 / n) as f32); 

//             neurotransmitters
//         } else {
//             HashMap::new()
//         };
        
//         let presynaptic_inputs: Vec<f32> = (0..n)
//             .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
//             .collect();
//         let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
//             .map(|(presynaptic_neuron, input_value)| {
//                 presynaptic_neuron.iterate_and_spike(*input_value)
//             })
//             .collect();
//         let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//             calculated_current,
//             &presynaptic_neurotransmitters,
//         );

//         if reward_times.contains(&timestep) {
//             reward_params.update_dopamine(reward);
//         } else {
//             reward_params.update_dopamine(0.);
//         }

//         for (n, i) in is_spikings.iter().enumerate() {
//             if *i {
//                 presynaptic_neurons[n].set_last_firing_time(Some(timestep));
//                 delta_ws[n] = update_weight_stdp(&presynaptic_neurons[n], &*postsynaptic_neuron);
//             }
//         }

//         if is_spiking {
//             postsynaptic_neuron.set_last_firing_time(Some(timestep));
//             for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
//                 delta_ws[n_neuron] = update_weight_stdp(i, &*postsynaptic_neuron);
//             }
//         }

//         for (index, trace) in traces.iter_mut().enumerate() {
//             trace.update_trace(delta_ws[index]);
//             weights[index] += trace.c * reward_params.dopamine;
//         }

//         for (index, i) in presynaptic_neurons.iter().enumerate() {
//             output_hashmap.get_mut(&format!("presynaptic_voltage_{}", index))
//                 .expect("Could not find hashmap value")
//                 .push(i.get_current_voltage());
//         }
//         output_hashmap.get_mut("postsynaptic_voltage").expect("Could not find hashmap value")
//             .push(postsynaptic_neuron.get_current_voltage());
//         for (index, i) in weights.iter().enumerate() {
//             output_hashmap.get_mut(&format!("weight_{}", index))
//                 .expect("Could not find hashmap value")
//                 .push(*i);
//         }
//     }

//     output_hashmap
// }

fn main() {

}