use std::{
    collections::HashMap,
    io::Result,
};
#[path = "../distribution/mod.rs"]
mod distribution;
use crate::distribution::limited_distr;
#[path = "../neuron/mod.rs"]
mod neuron;
use crate::neuron::{
    Cell, HodgkinHuxleyCell, IFParameters, PotentiationType, STDPParameters,
    find_peaks, diff, hodgkin_huxley_bayesian, if_params_bayesian,
    voltage_change_to_current, voltage_change_to_current_integrate_and_fire,
};
#[path = "../ga/mod.rs"]
use crate::ga::{BitString, decode};


fn get_average_spike(peaks: &Vec<usize>, voltages: &Vec<f64>) -> f64 {
    peaks.iter()
        .map(|n| voltages[*n])
        .sum::<f64>() / (peaks.len() as f64)
}

pub struct ActionPotentialSummary {
    pub average_pre_spike_amplitude: f64,
    pub average_post_spike_amplitude: f64,
    pub average_pre_spike_time_difference: f64,
    pub average_post_spike_time_difference: f64,
}

fn get_summary(
    pre_voltages: &Vec<f64>, 
    post_voltages: &Vec<f64>, 
    pre_peaks: Option<&Vec<usize>>,
    post_peaks: Option<&Vec<usize>>,
    tolerance: f64,
) -> ActionPotentialSummary {
    let pre_peaks = match pre_peaks {
        Some(values) => values,
        None => find_peaks(pre_voltages, tolerance),
    };
    let post_peaks = match post_peaks {
        Some(values) => values,
        None => find_peaks(post_voltages, tolerance),
    };

    let average_pre_spike: f64 = get_average_spike(&pre_peaks, pre_voltages);
    let average_post_spike: f64 = get_average_spike(&post_peaks, post_voltages);

    let average_pre_spike_difference: f64 = diff(&pre_peaks).iter()
        .sum::<usize>() as f64 / (pre_peaks.len() as f64);
    let average_post_spike_difference: f64 = diff(&post_peaks).iter()
        .sum::<usize>() as f64 / (post_peaks.len() as f64);

    ActionPotentialSummary {
        average_pre_spike_amplitude: average_pre_spike,
        average_post_spike_amplitude: average_post_spike,
        average_pre_spike_time_difference: average_pre_spike_difference,
        average_post_spike_time_difference: average_post_spike_difference,
    }
}

fn compare_summary(summary1: &ActionPotentialSummary, summary2: &ActionPotentialSummary) -> f64 {
    let pre_spike_amplitude = (summary1.average_pre_spike_amplitude - summary2.average_pre_spike_amplitude).powf(2.);
    let post_spike_amplitude = (summary1.average_post_spike_amplitude - summary2.average_post_spike_amplitude).powf(2.);

    let pre_spike_difference = (summary1.average_pre_spike_time_difference - summary2.average_pre_spike_time_difference).powf(2.);
    let post_spike_difference = (summary1.average_post_spike_time_difference - summary2.average_post_spike_time_difference).powf(2.);

    pre_spike_amplitude + post_spike_amplitude + pre_spike_difference + post_spike_difference
}

pub fn get_hodgkin_huxley_voltages(
    hodgkin_huxley_model: &HodgkinHuxleyCell, 
    input_current: f64, 
    iterations: usize,
    bayesian: bool, 
    tolerance: f64
) -> ActionPotentialSummary {
    let mut presynaptic_neuron = hodgkin_huxley_model.clone();
    let mut postsynaptic_neuron = hodgkin_huxley_model.clone();

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    for _ in 0..iterations {
        if bayesian {
            let bayesian_factor = hodgkin_huxley_bayesian(&postsynaptic_neuron);

            postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage * bayesian_factor);

            presynaptic_neuron.iterate(
                input_current * hodgkin_huxley_bayesian(&presynaptic_neuron)
            );

            let current = voltage_change_to_current(
                presynaptic_neuron.last_dv, &presynaptic_neuron
            );

            postsynaptic_neuron.iterate(
                current * bayesian_factor
            );
        } else {
            postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage);
            presynaptic_neuron.iterate(input_current);

            let current = voltage_change_to_current(
                presynaptic_neuron.last_dv, &presynaptic_neuron
            );

            postsynaptic_neuron.iterate(current);
        }

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    get_summary(&pre_voltages, &post_voltages, None, None, tolerance)
}

pub struct FittingSettings<'a> {
    pub hodgkin_huxley_model: HodgkinHuxleyCell,
    pub if_params: &'a IFParameters,
    pub action_potential_summary: &'a ActionPotentialSummary,
    pub input_current: f64,
    pub iterations: usize,
    pub bayesian: bool,
}

fn bayesian_izhikevich_get_dv_change(
    izhikevich_neuron: &mut Cell, 
    if_params: &IFParameters, 
    input_current: f64,
    bayesian: bool
) -> f64 {
    if bayesian {
        izhikevich_neuron.izhikevich_get_dv_change(
            &if_params, 
            input_current * limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.)
        )
    } else {
        izhikevich_neuron.izhikevich_get_dv_change(
            &if_params, 
            input_current,
        )
    }
}

// bounds should be a, b, c, d, and v_th for now
// if fitting does not generalize, optimize other coefs in equation
pub fn fitting_objective(
    bitstring: &BitString, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: usize, 
    settings: &HashMap<&str, FittingSettings>
) -> Result<f64> {
    let decoded = match decode(bitstring, bounds, n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(e) => return Err(e),
    };

    let a: f64 = decoded[0];
    let b: f64 = decoded[1];
    let c: f64 = decoded[2];
    let d: f64 = decoded[3];
    let v_th: f64 = decoded[4];

    let settings = settings.get("settings").unwrap();

    let mut if_params = settings.if_params.clone();
    if_params.v_th = v_th;

    let mut presynaptic_neuron = Cell { 
        current_voltage: if_params.v_init, 
        refractory_count: 0.0,
        leak_constant: -1.,
        integration_constant: 1.,
        potentiation_type: PotentiationType::Excitatory,
        neurotransmission_concentration: 0., 
        neurotransmission_release: 1.,
        receptor_density: 1.,
        chance_of_releasing: 0.5, 
        dissipation_rate: 0.1, 
        chance_of_random_release: 0.2,
        random_release_concentration: 0.1,
        w_value: if_params.w_init,
        stdp_params: STDPParameters::default(),
        last_firing_time: None,
        alpha: a,
        beta: b,
        c: c,
        d: d,
        last_dv: 0.,
    };

    let mut postynaptic_neuron = presynaptic_neuron.clone();

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postynaptic_neuron.current_voltage];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..settings.iterations {
        let pre_spike = presynaptic_neuron.izhikevich_apply_dw_and_get_spike(&if_params);
        let pre_dv = bayesian_izhikevich_get_dv_change(
            &mut presynaptic_neuron, 
            &if_params, 
            settings.input_current, 
            settings.bayesian,
        );

        if pre_spike {
            pre_peaks.push(timestep);
        }

        presynaptic_neuron.last_dv = pre_dv;

        let postsynaptic_input = voltage_change_to_current_integrate_and_fire(
            presynaptic_neuron.last_dv, if_params.dt, 1.0
        );

        let post_spike = presynaptic_neuron.izhikevich_apply_dw_and_get_spike(&if_params);
        let post_dv = bayesian_izhikevich_get_dv_change(
            &mut postynaptic_neuron, 
            &if_params, 
            postsynaptic_input, 
            settings.bayesian,
        );

        if post_spike {
            post_peaks.push(timestep);
        }

        postynaptic_neuron.last_dv = post_dv;
    
        presynaptic_neuron.current_voltage += pre_dv;
        postynaptic_neuron.current_voltage += post_dv;

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postynaptic_neuron.current_voltage);
    }

    let summary = get_summary(
        &pre_voltages, &post_voltages, Some(&pre_peaks), Some(&post_peaks), settings.tolerance
    );

    Ok(compare_summary(&summary, settings.action_potential_summary))
}
