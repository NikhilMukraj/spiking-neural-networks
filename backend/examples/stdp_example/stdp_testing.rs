// fn update_isolated_presynaptic_neuron_weights<T: IterateAndSpike>(
//     neurons: &mut Vec<T>,
//     neuron: &T,
//     weights: &mut Vec<f64>,
//     delta_ws: &mut Vec<f64>,
//     timestep: usize,
//     is_spikings: Vec<bool>,
// ) {
//     for (n, i) in is_spikings.iter().enumerate() {
//         if *i {
//             neurons[n].set_last_firing_time(Some(timestep));
//             delta_ws[n] = update_weight(&neurons[n], &*neuron);
//             weights[n] += delta_ws[n];
//         }
//     }
// }

// let calculated_voltage: f64 = (0..n)
//     .map(
//         |i| {
//             let output = weights[i] * signed_gap_junction(&presynaptic_neurons[i], &*postsynaptic_neuron);

//             if averaged {
//                 output / (n as f64)
//             } else {
//                 output
//             }
//         }
//     ) 
//     .collect::<Vec<f64>>()
//     .iter()
//     .sum();
// let presynaptic_neurotransmitters: Option<NeurotransmitterConcentrations> = match do_receptor_kinetics {
//     true => Some({
//         let neurotransmitters_vec = (0..n) 
//             .map(|i| {
//                 let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
//                 weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

//                 if averaged {
//                     weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, (1 / n) as f64);
//                 } 

//                 presynaptic_neurotransmitter
//             }
//         ).collect::<Vec<NeurotransmitterConcentrations>>();

//         let neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);

//         neurotransmitters
//     }),
//     false => None
// };

// let noise_factor = postsynaptic_neuron.get_gaussian_factor();
// let presynaptic_inputs: Vec<f64> = (0..n)
//     .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
//     .collect();
// let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
//     .map(|(presynaptic_neuron, input_value)| {
//         presynaptic_neuron.iterate_and_spike(*input_value)
//     })
//     .collect();
// let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//     noise_factor * calculated_voltage,
//     presynaptic_neurotransmitters.as_ref(),
// );

// update_isolated_presynaptic_neuron_weights(
//     presynaptic_neurons, 
//     &postsynaptic_neuron,
//     &mut weights, 
//     &mut delta_ws, 
//     timestep, 
//     is_spikings,
// );

// if is_spiking {
//     postsynaptic_neuron.set_last_firing_time(Some(timestep));
//     for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
//         delta_ws[n_neuron] = update_weight(i, postsynaptic_neuron);
//         weights[n_neuron] += delta_ws[n_neuron];
//     }
// }

// plot weights over time
