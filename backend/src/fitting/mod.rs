use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Result},
};
use crate::distribution::limited_distr;
use crate::neuron::{
    Cell, HodgkinHuxleyCell, IFParameters, PotentiationType, STDPParameters,
    find_peaks, diff, hodgkin_huxley_bayesian,
    voltage_change_to_current, voltage_change_to_current_integrate_and_fire,
};
use crate::ga::{BitString, decode};


fn get_average_spike(peaks: &Vec<usize>, voltages: &Vec<f64>) -> f64 {
    let average = peaks.iter()
        .map(|n| voltages[*n])
        .sum::<f64>() / (peaks.len() as f64);

    match average {
        x if x.is_nan() => f64::INFINITY,
        _ => average,
    }
}

#[derive(Debug)]
pub struct ActionPotentialSummary {
    pub average_pre_spike_amplitude: f64,
    pub average_post_spike_amplitude: f64,
    pub average_pre_spike_time_difference: f64,
    pub average_post_spike_time_difference: f64,
    // pub num_pre_spikes: f64,
    // pub num_post_spikes: f64,
}

fn get_summary(
    pre_voltages: &Vec<f64>, 
    post_voltages: &Vec<f64>, 
    pre_peaks: Option<&Vec<usize>>,
    post_peaks: Option<&Vec<usize>>,
    tolerance: Option<f64>,
) -> Result<ActionPotentialSummary> {
    let pre_peaks = match (pre_peaks, tolerance) {
        (Some(values), None) | (Some(values), Some(_)) => values.to_owned(),
        (None, Some(tolerance)) => find_peaks(pre_voltages, tolerance),
        (None, None) => { 
            return Err(
                Error::new(
                    ErrorKind::InvalidInput, 
                    "Peaks must be precalculated or provide a tolerance to calculate peaks"
                )
            );
        },
    };
    let post_peaks = match (post_peaks, tolerance) {
        (Some(values), None) | (Some(values), Some(_)) => values.to_owned(),
        (None, Some(tolerance)) => find_peaks(post_voltages, tolerance),
        (None, None) => { 
            return Err(
                Error::new(
                    ErrorKind::InvalidInput, 
                    "Peaks must be precalculated or provide a tolerance to calculate peaks"
                )
            );
        },
    };

    let average_pre_spike: f64 = get_average_spike(&pre_peaks, pre_voltages);
    let average_post_spike: f64 = get_average_spike(&post_peaks, post_voltages);

    let mut average_pre_spike_difference: f64 = diff(&pre_peaks).iter()
        .sum::<usize>() as f64 / (pre_peaks.len() as f64);
    let mut average_post_spike_difference: f64 = diff(&post_peaks).iter()
        .sum::<usize>() as f64 / (post_peaks.len() as f64);

    if average_pre_spike_difference.is_nan() {
        average_pre_spike_difference = f64::INFINITY;
    }
    if average_post_spike_difference.is_nan() {
        average_post_spike_difference = f64::INFINITY;
    }

    Ok(
        ActionPotentialSummary {
            average_pre_spike_amplitude: average_pre_spike,
            average_post_spike_amplitude: average_post_spike,
            average_pre_spike_time_difference: average_pre_spike_difference,
            average_post_spike_time_difference: average_post_spike_difference,
        }
    )
}

// struct SummaryScalingFactors {
//     amplitude_scale: f64,
//     time_difference_scale: f64,
//     // num_peaks_scale: f64,
// }

// fn get_f64_max(x: &Vec<f64>) -> Option<&f64> {
//     x.iter()
//         .max_by(|a, b| a.total_cmp(b))
// }

// fn reference_scale(
//     reference_summary: &ActionPotentialSummary, 
// ) -> (ActionPotentialSummary, SummaryScalingFactors) {
//     let amplitudes = vec![
//         reference_summary.average_pre_spike_amplitude, reference_summary.average_post_spike_amplitude
//     ];

//     let time_differences = vec![
//         reference_summary.average_pre_spike_time_difference, reference_summary.average_post_spike_time_difference
//     ];

//     let amplitude_scale = *get_f64_max(&amplitudes).unwrap();

//     let time_difference_scale = *get_f64_max(&time_differences).unwrap();

//     let scaled_reference = ActionPotentialSummary {
//         average_post_spike_amplitude: reference_summary.average_pre_spike_amplitude / amplitude_scale,
//         average_pre_spike_amplitude: reference_summary.average_post_spike_amplitude / amplitude_scale,
//         average_pre_spike_time_difference: reference_summary.average_pre_spike_time_difference / time_difference_scale,
//         average_post_spike_time_difference: reference_summary.average_post_spike_time_difference / time_difference_scale,
//     };

//     let scaling_factors = SummaryScalingFactors {
//         amplitude_scale: amplitude_scale, 
//         time_difference_scale: time_difference_scale,
//     };

//     (scaled_reference, scaling_factors)
// }

// fn scale_summary(
//     summary: &ActionPotentialSummary, 
//     scaling_factors: &SummaryScalingFactors
// ) -> ActionPotentialSummary {
//     ActionPotentialSummary {
//         average_post_spike_amplitude: summary.average_pre_spike_amplitude / scaling_factors.amplitude_scale,
//         average_pre_spike_amplitude: summary.average_post_spike_amplitude / scaling_factors.amplitude_scale,
//         average_pre_spike_time_difference: summary.average_pre_spike_time_difference / scaling_factors.time_difference_scale,
//         average_post_spike_time_difference: summary.average_post_spike_time_difference / scaling_factors.time_difference_scale,
//     }
// }

fn compare_summary(summary1: &ActionPotentialSummary, summary2: &ActionPotentialSummary) -> f64 {
    let pre_spike_amplitude = (summary1.average_pre_spike_amplitude - summary2.average_pre_spike_amplitude).powf(2.);
    let post_spike_amplitude = (summary1.average_post_spike_amplitude - summary2.average_post_spike_amplitude).powf(2.);

    let pre_spike_difference = (summary1.average_pre_spike_time_difference - summary2.average_pre_spike_time_difference).powf(2.);
    let post_spike_difference = (summary1.average_post_spike_time_difference - summary2.average_post_spike_time_difference).powf(2.);

    let output = pre_spike_amplitude + post_spike_amplitude + pre_spike_difference + post_spike_difference;

    if output.is_nan() {
        f64::INFINITY
    } else {
        output
    }
}

pub fn get_hodgkin_huxley_voltages(
    hodgkin_huxley_model: &HodgkinHuxleyCell, 
    input_current: f64, 
    iterations: usize,
    bayesian: bool, 
    tolerance: f64
) -> Result<ActionPotentialSummary> {
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
                &presynaptic_neuron
            );

            postsynaptic_neuron.iterate(
                current * bayesian_factor
            );
        } else {
            postsynaptic_neuron.update_neurotransmitter(presynaptic_neuron.current_voltage);
            presynaptic_neuron.iterate(input_current);

            let current = voltage_change_to_current(
                &presynaptic_neuron
            );

            postsynaptic_neuron.iterate(current);
        }

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    Ok(get_summary(&pre_voltages, &post_voltages, None, None, Some(tolerance))?)
}

#[derive(Clone)]
pub struct FittingSettings<'a> {
    pub hodgkin_huxley_model: HodgkinHuxleyCell,
    pub if_params: &'a IFParameters,
    pub action_potential_summary: &'a ActionPotentialSummary,
    // pub scaling_factors: &'a SummaryScalingFactors,
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

pub fn get_izhikevich_summary(
    presynaptic_neuron: &mut Cell, 
    postsynaptic_neuron: &mut Cell,
    if_params: &IFParameters,
    settings: &FittingSettings,
) -> Result<ActionPotentialSummary> {
    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..settings.iterations {
        let pre_spike = presynaptic_neuron.izhikevich_apply_dw_and_get_spike(&if_params);
        let pre_dv = bayesian_izhikevich_get_dv_change(
            presynaptic_neuron, 
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

        let post_spike = postsynaptic_neuron.izhikevich_apply_dw_and_get_spike(&if_params);
        let post_dv = bayesian_izhikevich_get_dv_change(
            postsynaptic_neuron, 
            &if_params, 
            postsynaptic_input, 
            settings.bayesian,
        );

        if post_spike {
            post_peaks.push(timestep);
        }

        postsynaptic_neuron.last_dv = post_dv;
    
        presynaptic_neuron.current_voltage += pre_dv;
        postsynaptic_neuron.current_voltage += post_dv;

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    get_summary(
        &pre_voltages, &post_voltages, Some(&pre_peaks), Some(&post_peaks), None
    )
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
        neurotransmission_release: 0.,
        receptor_density: 0.,
        chance_of_releasing: 0., 
        dissipation_rate: 0., 
        chance_of_random_release: 0.,
        random_release_concentration: 0.,
        w_value: if_params.w_init,
        stdp_params: STDPParameters::default(),
        last_firing_time: None,
        alpha: a,
        beta: b,
        c: c,
        d: d,
        last_dv: 0.,
    };

    let mut postsynaptic_neuron = presynaptic_neuron.clone();

    let summary = get_izhikevich_summary(
        &mut presynaptic_neuron, 
        &mut postsynaptic_neuron, 
        &if_params,
        settings
    )?;

    Ok(compare_summary(&summary, settings.action_potential_summary))
}
