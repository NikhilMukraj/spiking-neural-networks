use std::{
    collections::HashMap,
    io::{Error, ErrorKind, Result},
};
use crate::distribution::limited_distr;
use crate::neuron::{
    IntegrateAndFireCell, HodgkinHuxleyCell,
    find_peaks, diff, gap_junction, iterate_coupled_spiking_neurons,
    iterate_and_spike::NeurotransmitterKinetics,
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

pub fn get_hodgkin_huxley_summary<T: NeurotransmitterKinetics>(
    hodgkin_huxley_model: &HodgkinHuxleyCell<T>, 
    input_current: f64, 
    iterations: usize,
    do_receptor_kinetics: bool,
    bayesian: bool, 
    tolerance: f64,
    spike_amplitude_default: f64,
) -> Result<ActionPotentialSummary> {
    let mut presynaptic_neuron = hodgkin_huxley_model.clone();
    let mut postsynaptic_neuron = hodgkin_huxley_model.clone();

    presynaptic_neuron.initialize_parameters(presynaptic_neuron.current_voltage);
    postsynaptic_neuron.initialize_parameters(postsynaptic_neuron.current_voltage);

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    for _ in 0..iterations {
        iterate_coupled_spiking_neurons(
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron, 
            do_receptor_kinetics,
            bayesian, 
            input_current
        );

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
pub struct FittingSettings<'a, T: NeurotransmitterKinetics, U: NeurotransmitterKinetics> {
    pub hodgkin_huxley_model: HodgkinHuxleyCell<T>,
    pub if_neuron: &'a IntegrateAndFireCell<U>,
    pub action_potential_summary: &'a [ActionPotentialSummary],
    pub scaling_factors: &'a [Option<SummaryScalingFactors>],
    pub use_amplitude: bool,
    pub spike_amplitude_default: f64,
    pub input_currents: &'a [f64],
    pub iterations: usize,
    pub bayesian: bool,
    // pub do_receptor_kinetics: bool,
}

fn bayesian_izhikevich_iterate<T: NeurotransmitterKinetics>(
    izhikevich_neuron: &mut IntegrateAndFireCell<T>, 
    input_current: f64,
    // t_total: NeurotransmitterConcentrations,
    bayesian: bool,
    // do_receptor_kinetics: bool,
) -> bool {
    let processed_input = if bayesian {
        let bayesian_factor = limited_distr(izhikevich_neuron.bayesian_params.mean, izhikevich_neuron.bayesian_params.std, 0., 1.);
        let bayesian_input = input_current * bayesian_factor;

        bayesian_input
    } else {
        input_current
    };

    let spike = izhikevich_neuron.izhikevich_iterate_and_spike(
        processed_input,
    );

    spike
}

pub fn get_izhikevich_summary<T: NeurotransmitterKinetics, U: NeurotransmitterKinetics>(
    presynaptic_neuron: &mut IntegrateAndFireCell<U>, 
    postsynaptic_neuron: &mut IntegrateAndFireCell<U>,
    settings: &FittingSettings<T, U>,
    index: usize,
) -> Result<ActionPotentialSummary> {
    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..settings.iterations {
        let postsynaptic_input = gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let pre_spike = bayesian_izhikevich_iterate(presynaptic_neuron, settings.input_currents[index], settings.bayesian);
        let post_spike = bayesian_izhikevich_iterate(postsynaptic_neuron, postsynaptic_input, settings.bayesian);

        if pre_spike {
            pre_peaks.push(timestep);
        }

        if post_spike {
            post_peaks.push(timestep);
        }

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
pub fn fitting_objective<T: NeurotransmitterKinetics, U: NeurotransmitterKinetics>(
    bitstring: &BitString, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: usize, 
    settings: &HashMap<&str, FittingSettings<T, U>>
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

    let mut test_cell = settings.if_neuron.clone();
    test_cell.v_th = v_th;
    test_cell.gap_conductance = gap_conductance;
    test_cell.alpha = a;
    test_cell.beta = b;
    test_cell.c = c;
    test_cell.d = d;

    let summaries_results = (0..settings.input_currents.len())
        .map(|i| {
            get_izhikevich_summary(
                &mut test_cell.clone(), 
                &mut test_cell.clone(), 
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
