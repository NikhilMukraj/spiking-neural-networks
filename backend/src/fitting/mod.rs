// #[path = "../neuron/mod.rs"]
// mod neuron;
// use crate::neuron::{Cell, HodgkinHuxleyCell, find_peaks, diff};
// #[path = "../ga/mod.rs"]
// use crate::ga::{BitString, decode, genetic_algo};


// fn get_average_spike(peaks: &Vec<usize>, voltages: &Vec<f64>) {
    // peaks.iter()
        // .map(|n| voltages[n])
        // .sum::<f64>() / (peaks.len() as f64)
// }

// struct ActionPotentialSummary {
//     average_pre_spike_amplitude: f64,
//     average_post_spike_amplitude: f64,
//     average_pre_spike_time_difference: f64,
//     average_post_spike_time_difference: f64,
// }

// fn get_summary(
//     pre_voltages: &Vec<f64>, 
//     post_voltages: &Vec<f64>, 
//     pre_peaks: &Vec<usize>, 
//     post_peaks: &Vec<usize>,
//     tolerance: f64,
// ) -> ActionPotentialSummary {
//     let pre_peaks = find_peaks(pre_voltages, tolerance);
//     let post_peaks = find_peaks(post_voltages, tolerance);

//     let average_pre_spike: f64 = get_average_spike(pre_peaks, pre_voltages);
//     let average_post_spike: f64 = get_average_spike(post_peaks, post_voltages);

//     let average_pre_spike_difference: f64 = diff(pre_peaks) / (pre_peaks.len() as f64);
//     let average_post_spike_difference: f64 = diff(post_peaks) / (post_peaks.len() as f64);

//     struct ActionPotentialSummary {
//         average_pre_spike: f64,
//         average_post_spike: f64,
//         average_pre_spike_difference: f64,
//         average_post_spike_difference: f64,
//     }
// }

// fn get_hodgkin_huxley_voltages(hodgkin_huxley_mode: &HodgkinHuxleyCell, bayesian: bool, tolerance: f64) -> ActionPotentialSummary {
    // let mut presynaptic_neuron = settings.hodgkin_huxley_model.clone();
    // let mut postsynaptic_neuron = settings.hodgkin_huxley_model.clone();

    // let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    // let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    // for _ in 0..settings.iterations {
    //     if bayesian {
    //         let bayesian_factor = limited_distr(
    //             postsynaptic_neuron.bayesian_params.mean, 
    //             postsynaptic_neuron.bayesian_params.std, 
    //             postsynaptic_neuron.bayesian_params.min, 
    //             postsynaptic_neuron.bayesian_params.max,
    //         );

    //         postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage * bayesian_factor);

    //         presynaptic_neuron.iterate(
    //             input_voltage * limited_distr(
    //                 presynaptic_neuron.bayesian_params.mean, 
    //                 presynaptic_neuron.bayesian_params.std, 
    //                 presynaptic_neuron.bayesian_params.min, 
    //                 presynaptic_neuron.bayesian_params.max,
    //             )
    //         );

    //         let current = voltage_change_to_current(
    //             presynaptic_neuron.current_voltage - past_presynaptic_voltage, &presynaptic_neuron
    //         );

    //         postsynaptic_neuron.iterate(
    //             current * bayesian_factor
    //         );
    //     } else {
    //         postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage);
    //         presynaptic_neuron.iterate(input_voltage);

    //         let current = voltage_change_to_current(
    //             presynaptic_neuron.current_voltage - past_presynaptic_voltage, &presynaptic_neuron
    //         );

    //         postsynaptic_neuron.iterate(current);
    //     }

        // pre_voltages.push(presynaptic_neuron.current_voltage);
        // post_voltages.push(postsynaptic_neuron.current_voltage);
    // }

    // get_summary(&pre_voltages, &post_voltages, &pre_peaks, &post_peaks, tolerance)
// }

// struct FittingSettings {
//     hodgkin_huxley_model: HodgkinHuxleyCell,
    // if_params: &IFParameters,
    // action_potential_summary: &ActionPotentialSummary,
    // iterations: usize,
    // bayesian: bool,
// }

// // bounds should be a, b, c, d, and v_th for now
// // if fitting does not generalize, optimize other coefs in equation
// write trait for iterate so this code can be shared in an abstraction
// fn objective(
//     bitstring: &BitString, 
//     bounds: &Vec<Vec<f64>>, 
//     n_bits: usize, 
//     settings: &HashMap<&str, FittingSettings>
// ) -> Result<f64> {
    // let decoded = match decode(bitstring, bounds, n_bits) {
    //     Ok(decoded_value) => decoded_value,
    //     Err(e) => return Err(e),
    // };

    // let a: f64 = decoded[0];
    // let b: f64 = decoded[1];
    // let c: f64 = decoded[2];
    // let d: f64 = decoded[3];
    // let v_th: f64 = decoded[4];

    // let mut if_params = settings.if_params.clone()
    // if_params.v_th = v_th;

    // let mut presynaptic_neuron = Cell { 
    //     current_voltage: if_params.v_init, 
    //     refractory_count: 0.0,
    //     leak_constant: -1.,
    //     integration_constant: 1.,
    //     potentiation_type: PotentiationType::Excitatory,
    //     neurotransmission_concentration: 0., 
    //     neurotransmission_release: 1.,
    //     receptor_density: 1.,
    //     chance_of_releasing: 0.5, 
    //     dissipation_rate: 0.1, 
    //     chance_of_random_release: 0.2,
    //     random_release_concentration: 0.1,
    //     w_value: if_params.w_init,
    //     stdp_params: STDPParameters::default(),
    //     last_firing_time: None,
    //     alpha: a,
    //     beta: b,
    //     c: c,
    //     d: d,
    //     last_dv: 0.,
    // };

    // let mut post_synaptic_neuron = presynaptic_neuron.clone();

    // let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    // let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    // for _ in 0..settings.iterations {
    //     if bayesian {
    //         let bayesian_factor = limited_distr(
    //             postsynaptic_neuron.bayesian_params.mean, 
    //             postsynaptic_neuron.bayesian_params.std, 
    //             postsynaptic_neuron.bayesian_params.min, 
    //             postsynaptic_neuron.bayesian_params.max,
    //         );

    //         postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage * bayesian_factor);

    //         presynaptic_neuron.iterate(
    //             input_voltage * limited_distr(
    //                 presynaptic_neuron.bayesian_params.mean, 
    //                 presynaptic_neuron.bayesian_params.std, 
    //                 presynaptic_neuron.bayesian_params.min, 
    //                 presynaptic_neuron.bayesian_params.max,
    //             )
    //         );

    //         let current = voltage_change_to_current(
    //             presynaptic_neuron.current_voltage - past_presynaptic_voltage, &presynaptic_neuron
    //         );

    //         postsynaptic_neuron.iterate(
    //             current * bayesian_factor
    //         );
    //     } else {
    //         postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage);
    //         presynaptic_neuron.iterate(input_voltage);

    //         let current = voltage_change_to_current(
    //             presynaptic_neuron.current_voltage - past_presynaptic_voltage, &presynaptic_neuron
    //         );

    //         postsynaptic_neuron.iterate(current);
    //     }

        // pre_voltages.push(presynaptic_neuron.current_voltage);
        // post_voltages.push(postsynaptic_neuron.current_voltage);
    // }

    // let summary = get_summary(&pre_voltages, &post_voltages, &pre_peaks, &post_peaks, tolerance);
// }
