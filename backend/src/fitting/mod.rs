use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Result},
};
use crate::distribution::limited_distr;
use crate::neuron::{
    IntegrateAndFireCell, HodgkinHuxleyCell, IFParameters, PotentiationType, STDPParameters,
    find_peaks, diff, gap_junction, iterate_coupled_hodgkin_huxley,
    handle_receptor_kinetics
};
use crate::ga::{BitString, decode};


fn get_average_spike(peaks: &Vec<usize>, voltages: &Vec<f64>, default: f64) -> f64 {
    if peaks.len() == 0 {
        return default;
    }

    peaks.iter()
        .map(|n| voltages[*n])
        .sum::<f64>() / (peaks.len() as f64)
}

#[derive(Debug)]
pub struct ActionPotentialSummary {
    pub average_pre_spike_amplitude: f64,
    pub average_post_spike_amplitude: f64,
    pub average_pre_spike_time_difference: f64,
    pub average_post_spike_time_difference: f64,
    pub num_pre_spikes: f64,
    pub num_post_spikes: f64,
}

fn get_summary(
    pre_voltages: &Vec<f64>, 
    post_voltages: &Vec<f64>, 
    pre_peaks: Option<&Vec<usize>>,
    post_peaks: Option<&Vec<usize>>,
    tolerance: Option<f64>,
    spike_amplitude_default: f64,
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

    let average_pre_spike: f64 = get_average_spike(&pre_peaks, pre_voltages, spike_amplitude_default);
    let average_post_spike: f64 = get_average_spike(&post_peaks, post_voltages, spike_amplitude_default);

    let average_pre_spike_difference: f64 = if pre_peaks.len() != 0 {
        diff(&pre_peaks).iter()
            .sum::<usize>() as f64 / (pre_peaks.len() as f64)
    } else {
        0.
    };

    let average_post_spike_difference: f64 = if post_peaks.len() != 0 {
        diff(&post_peaks).iter()
            .sum::<usize>() as f64 / (post_peaks.len() as f64)
    } else {
        0.
    };

    Ok(
        ActionPotentialSummary {
            average_pre_spike_amplitude: average_pre_spike,
            average_post_spike_amplitude: average_post_spike,
            average_pre_spike_time_difference: average_pre_spike_difference,
            average_post_spike_time_difference: average_post_spike_difference,
            num_pre_spikes: pre_peaks.len() as f64,
            num_post_spikes: post_peaks.len() as f64,
        }
    )
}

pub struct SummaryScalingDefaults {
    pub default_amplitude_scale: f64,
    pub default_time_difference_scale: f64,
    pub default_num_peaks_scale: f64,
}

#[derive(Clone, Copy)]
pub struct SummaryScalingFactors {
    pub amplitude_scale: f64,
    pub time_difference_scale: f64,
    pub num_peaks_scale: f64,
}

fn get_f64_max(x: &Vec<f64>) -> Option<&f64> {
    x.iter()
        .max_by(|a, b| a.total_cmp(b))
}

fn replace_with_default(value: f64, default: f64) -> f64 {
    if value == 0. {
        default
    } else {
        value
    }
}

pub fn get_reference_scale(
    reference_summary: &ActionPotentialSummary, 
    scaling_defaults: &SummaryScalingDefaults,
) -> (ActionPotentialSummary, SummaryScalingFactors) {
    let amplitudes = vec![
        reference_summary.average_pre_spike_amplitude, reference_summary.average_post_spike_amplitude
    ];
    let time_differences = vec![
        reference_summary.average_pre_spike_time_difference, reference_summary.average_post_spike_time_difference
    ];
    let peaks_lens = vec![
        reference_summary.num_pre_spikes, reference_summary.num_post_spikes,
    ];

    let amplitude_scale = replace_with_default(
        *get_f64_max(&amplitudes).unwrap(), scaling_defaults.default_amplitude_scale
    );
    let time_difference_scale = replace_with_default(
        *get_f64_max(&time_differences).unwrap(), scaling_defaults.default_time_difference_scale
    );
    let num_peaks_scale = replace_with_default(
        *get_f64_max(&peaks_lens).unwrap(), scaling_defaults.default_num_peaks_scale
    );

    let scaled_reference = ActionPotentialSummary {
        average_post_spike_amplitude: reference_summary.average_pre_spike_amplitude / amplitude_scale,
        average_pre_spike_amplitude: reference_summary.average_post_spike_amplitude / amplitude_scale,
        average_pre_spike_time_difference: reference_summary.average_pre_spike_time_difference / time_difference_scale,
        average_post_spike_time_difference: reference_summary.average_post_spike_time_difference / time_difference_scale,
        num_pre_spikes: reference_summary.num_pre_spikes / num_peaks_scale,
        num_post_spikes: reference_summary.num_post_spikes / num_peaks_scale,
    };

    let scaling_factors = SummaryScalingFactors {
        amplitude_scale: amplitude_scale, 
        time_difference_scale: time_difference_scale,
        num_peaks_scale: num_peaks_scale,
    };

    (scaled_reference, scaling_factors)
}

fn scale_summary(
    summary: &ActionPotentialSummary, 
    scaling_factors: &SummaryScalingFactors
) -> ActionPotentialSummary {
    ActionPotentialSummary {
        average_post_spike_amplitude: summary.average_pre_spike_amplitude / scaling_factors.amplitude_scale,
        average_pre_spike_amplitude: summary.average_post_spike_amplitude / scaling_factors.amplitude_scale,
        average_pre_spike_time_difference: summary.average_pre_spike_time_difference / scaling_factors.time_difference_scale,
        average_post_spike_time_difference: summary.average_post_spike_time_difference / scaling_factors.time_difference_scale,
        num_pre_spikes: summary.num_pre_spikes / scaling_factors.num_peaks_scale,
        num_post_spikes: summary.num_post_spikes / scaling_factors.num_peaks_scale,
    }
}

fn compare_summary(summary1: &ActionPotentialSummary, summary2: &ActionPotentialSummary, use_amplitudes: bool) -> f64 {
    let mut score = 0.;

    if use_amplitudes {
        let pre_spike_amplitude = (summary1.average_pre_spike_amplitude - summary2.average_pre_spike_amplitude).powf(2.);
        let post_spike_amplitude = (summary1.average_post_spike_amplitude - summary2.average_post_spike_amplitude).powf(2.);

        score += pre_spike_amplitude + post_spike_amplitude;
    }

    let pre_spike_difference = (summary1.average_pre_spike_time_difference - summary2.average_pre_spike_time_difference).powf(2.);
    let post_spike_difference = (summary1.average_post_spike_time_difference - summary2.average_post_spike_time_difference).powf(2.);

    let num_pre_spikes = (summary1.num_pre_spikes - summary2.num_pre_spikes).powf(2.);
    let num_post_spikes = (summary1.num_post_spikes - summary2.num_post_spikes).powf(2.);

    score += pre_spike_difference + post_spike_difference +
        num_pre_spikes + num_post_spikes;

    if score.is_nan() {
        f64::INFINITY
    } else {
        score
    }
}

pub fn get_hodgkin_huxley_voltages(
    hodgkin_huxley_model: &HodgkinHuxleyCell, 
    input_current: f64, 
    iterations: usize,
    bayesian: bool, 
    tolerance: f64,
    spike_amplitude_default: f64,
) -> Result<ActionPotentialSummary> {
    let mut presynaptic_neuron = hodgkin_huxley_model.clone();
    let mut postsynaptic_neuron = hodgkin_huxley_model.clone();

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    for _ in 0..iterations {
        iterate_coupled_hodgkin_huxley(&mut presynaptic_neuron, &mut postsynaptic_neuron, bayesian, input_current);

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    Ok(
        get_summary(
            &pre_voltages, &post_voltages, None, None, Some(tolerance), spike_amplitude_default
        )?
    )
}

#[derive(Clone)]
pub struct FittingSettings<'a> {
    pub hodgkin_huxley_model: HodgkinHuxleyCell,
    pub if_params: &'a IFParameters,
    pub action_potential_summary: &'a [ActionPotentialSummary],
    pub scaling_factors: &'a [Option<SummaryScalingFactors>],
    pub use_amplitude: bool,
    pub spike_amplitude_default: f64,
    pub input_currents: &'a [f64],
    pub iterations: usize,
    pub bayesian: bool,
    pub do_receptor_kinetics: bool,
}

fn bayesian_izhikevich_get_dv_change(
    izhikevich_neuron: &mut IntegrateAndFireCell, 
    if_params: &IFParameters, 
    input_current: f64,
    bayesian: bool,
    do_receptor_kinetics: bool,
) -> f64 {
    if bayesian {
        let bayesian_factor = limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.);
        let bayesian_input = input_current * bayesian_factor;
        handle_receptor_kinetics(izhikevich_neuron, &if_params, bayesian_input, do_receptor_kinetics);

        izhikevich_neuron.izhikevich_get_dv_change(
            &if_params, 
            bayesian_input,
        ) + izhikevich_neuron.get_neurotransmitter_currents(if_params)
    } else {
        handle_receptor_kinetics(izhikevich_neuron, &if_params, input_current, do_receptor_kinetics);

        izhikevich_neuron.izhikevich_get_dv_change(
            &if_params, 
            input_current,
        ) + izhikevich_neuron.get_neurotransmitter_currents(if_params)
    }
}

pub fn get_izhikevich_summary(
    presynaptic_neuron: &mut IntegrateAndFireCell, 
    postsynaptic_neuron: &mut IntegrateAndFireCell,
    if_params: &IFParameters,
    settings: &FittingSettings,
    index: usize,
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
            settings.input_currents[index], 
            settings.bayesian,
            settings.do_receptor_kinetics,
        );

        if pre_spike {
            pre_peaks.push(timestep);
        }

        let postsynaptic_input = gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_spike = postsynaptic_neuron.izhikevich_apply_dw_and_get_spike(&if_params);
        let post_dv = bayesian_izhikevich_get_dv_change(
            postsynaptic_neuron, 
            &if_params, 
            postsynaptic_input, 
            settings.bayesian,
            settings.do_receptor_kinetics,
        );

        if post_spike {
            post_peaks.push(timestep);
        }
    
        presynaptic_neuron.current_voltage += pre_dv;
        postsynaptic_neuron.current_voltage += post_dv;

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    let summary = get_summary(
        &pre_voltages, &post_voltages, Some(&pre_peaks), Some(&post_peaks), None, settings.spike_amplitude_default
    )?;

    match settings.scaling_factors[index] {
        Some(factors) => Ok(scale_summary(&summary, &factors)),
        None => Ok(summary),
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
    let gap_conductance: f64 = decoded[5];

    let settings = settings.get("settings").unwrap();

    let mut if_params = settings.if_params.clone();
    if_params.v_th = v_th;

    let test_cell = IntegrateAndFireCell { 
        current_voltage: if_params.v_init, 
        refractory_count: 0.0,
        leak_constant: -1.,
        integration_constant: 1.,
        gap_conductance: gap_conductance,
        potentiation_type: PotentiationType::Excitatory,
        w_value: if_params.w_init,
        stdp_params: STDPParameters::default(),
        last_firing_time: None,
        alpha: a,
        beta: b,
        c: c,
        d: d,
        ligand_gates: if_params.ligand_gates_init.clone(),
    };

    let summaries_results = (0..settings.input_currents.len())
        .map(|i| {
            get_izhikevich_summary(
                &mut test_cell.clone(), 
                &mut test_cell.clone(), 
                &if_params,
                settings,
                i
            )
        })
        .collect::<Vec<Result<ActionPotentialSummary>>>();

    for result in summaries_results.iter() {
        if let Err(_) = result {
            return Err(Error::new(ErrorKind::InvalidData, "Summary calculation could not be completed"));
        }
    }

    let summaries = summaries_results.into_iter().map(|res| res.unwrap())
        .collect::<Vec<ActionPotentialSummary>>();

    let score = (0..settings.input_currents.len())
        .map(|i| {
            compare_summary(&settings.action_potential_summary[i], &summaries[i], settings.use_amplitude)
        })
        .sum::<f64>();

    Ok(score)
}

pub fn print_action_potential_summaries(
    summaries: &[ActionPotentialSummary], 
    scaling_factors: &[Option<SummaryScalingFactors>],
    use_amplitude: bool,
) {
    let mut pre_spike_amplitudes: Vec<f64> = Vec::new();
    let mut post_spike_amplitudes: Vec<f64> = Vec::new();
    let mut pre_spike_time_differences: Vec<f64> = Vec::new();
    let mut post_spike_time_differences: Vec<f64> = Vec::new();
    let mut num_pre_spikes: Vec<f64> = Vec::new();
    let mut num_post_spikes: Vec<f64> = Vec::new();

    for (summary, scaling) in summaries.iter().zip(scaling_factors) {
        let (amplitude_scaling, time_scaling, peaks_scaling) = match scaling {
            Some(value) => (value.amplitude_scale, value.time_difference_scale, value.num_peaks_scale),
            None => (1., 1., 1.)
        };

        if use_amplitude {
            pre_spike_amplitudes.push(summary.average_pre_spike_amplitude * amplitude_scaling);
            post_spike_amplitudes.push(summary.average_post_spike_amplitude * amplitude_scaling);
        }
        pre_spike_time_differences.push(summary.average_pre_spike_time_difference * time_scaling);
        post_spike_time_differences.push(summary.average_post_spike_time_difference * time_scaling);
        num_pre_spikes.push(summary.num_pre_spikes * peaks_scaling);
        num_post_spikes.push(summary.num_post_spikes * peaks_scaling);
    }

    if use_amplitude {
        println!("Presynaptic spike amplitudes: {:?}", pre_spike_amplitudes);
        println!("Postsynaptic spike amplitudes: {:?}", post_spike_amplitudes);
    }
    println!("Presynaptic spike time Differences: {:?}", pre_spike_time_differences);
    println!("Postsynaptic spike time Differences: {:?}", post_spike_time_differences);
    println!("# of presynaptic spikes: {:?}", num_pre_spikes);
    println!("# of postsynaptic spikes: {:?}", num_post_spikes);
}
