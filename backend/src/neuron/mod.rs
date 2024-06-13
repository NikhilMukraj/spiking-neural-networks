pub mod integrate_and_fire;
pub mod hodgkin_huxley;
pub mod attractors;
pub mod spike_train;
use spike_train::{SpikeTrain, NeuralRefractoriness};
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, Potentiation, BayesianFactor, LastFiringTime, STDP,
    IterateAndSpike, BayesianParameters, STDPParameters, PotentiationType,
    Neurotransmitters, NeurotransmitterType, NeurotransmitterKinetics, 
    ApproximateNeurotransmitter, weight_neurotransmitter_concentration,
    LigandGatedChannels,
    impl_current_voltage_with_kinetics,
    impl_gap_conductance_with_kinetics,
    impl_potentiation_with_kinetics,
    impl_bayesian_factor_with_kinetics,
    impl_last_firing_time_with_kinetics,
    impl_stdp_with_kinetics,
    impl_necessary_iterate_and_spike_traits,
};


pub type CellGrid<T> = Vec<Vec<T>>;

pub fn gap_junction<T: CurrentVoltage, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    postsynaptic_neuron.get_gap_conductance() * 
    (presynaptic_neuron.get_current_voltage() - postsynaptic_neuron.get_current_voltage())
}

pub fn signed_gap_junction<T: CurrentVoltage + Potentiation, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    sign * gap_junction(presynaptic_neuron, postsynaptic_neuron)
}

pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    do_receptor_kinetics: bool,
    bayesian: bool,
    input_current: f64,
) {
    let (t_total, post_current, input_current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let input_current = input_current * pre_bayesian_factor;

        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    } else {
        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    };

    let _pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);

    let _post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        t_total.as_ref(),
    );
}

pub fn spike_train_gap_juncton<T: SpikeTrain + Potentiation, U: GapConductance>(
    presynaptic_neuron: &T,
    postsynaptic_neuron: &U,
    timestep: usize,
) -> f64 {
    let (v_max, v_resting) = presynaptic_neuron.get_height();

    if let None = presynaptic_neuron.get_last_firing_time() {
        return v_resting;
    }

    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    let last_firing_time = presynaptic_neuron.get_last_firing_time().unwrap();
    let refractoriness_function = presynaptic_neuron.get_refractoriness_function();
    let dt = presynaptic_neuron.get_refractoriness_timestep();
    let conductance = postsynaptic_neuron.get_gap_conductance();

    sign * conductance * refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
}

pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
    spike_train: &mut T,
    presynaptic_neuron: &mut U, 
    postsynaptic_neuron: &mut U,
    timestep: usize,
    do_receptor_kinetics: bool,
    bayesian: bool,
) {
    let input_current = spike_train_gap_juncton(spike_train, presynaptic_neuron, timestep);

    let (pre_t_total, post_t_total, current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let pre_t_total = if do_receptor_kinetics {
            let mut t = spike_train.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, pre_bayesian_factor);

            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    } else {
        let pre_t_total = if do_receptor_kinetics {
            let t = spike_train.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    };

    let spike_train_spiking = spike_train.iterate();   
    if spike_train_spiking {
        spike_train.set_last_firing_time(Some(timestep));
    }
    
    let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        input_current,
        pre_t_total.as_ref(),
    );
    if pre_spiking {
        presynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        current,
        post_t_total.as_ref(),
    ); 
    if post_spiking {
        postsynaptic_neuron.set_last_firing_time(Some(timestep));
    }
}

pub fn update_weight<T: LastFiringTime, U: IterateAndSpike>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let mut delta_w: f64 = 0.;

    match (presynaptic_neuron.get_last_firing_time(), postsynaptic_neuron.get_last_firing_time()) {
        (Some(t_pre), Some(t_post)) => {
            let (t_pre, t_post): (f64, f64) = (t_pre as f64, t_post as f64);

            if t_pre < t_post {
                delta_w = postsynaptic_neuron.get_stdp_params().a_plus * 
                    (-1. * (t_pre - t_post).abs() / postsynaptic_neuron.get_stdp_params().tau_plus).exp();
            } else if t_pre > t_post {
                delta_w = -1. * postsynaptic_neuron.get_stdp_params().a_minus * 
                    (-1. * (t_post - t_pre).abs() / postsynaptic_neuron.get_stdp_params().tau_minus).exp();
            }
        },
        _ => {}
    };

    return delta_w;
}

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

// let noise_factor = postsynaptic_neuron.get_bayesian_factor();
// let presynaptic_inputs: Vec<f64> = (0..n)
//     .map(|i| input_currents[i] * presynaptic_neurons[i].get_bayesian_factor())
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
