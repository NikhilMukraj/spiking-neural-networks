//! A set of tools to fit a given neuron to another neuron

use std::{
    collections::HashMap,
    result,
    io::{Error, ErrorKind, Result},
    ops::Sub,
};
use crate::error::GeneticAlgorithmError;
use crate::neuron::{   
    iterate_and_spike::IterateAndSpike, 
    spike_train::SpikeTrain,
    iterate_coupled_spiking_neurons_and_spike_train,
};
use crate::ga::{BitString, decode, genetic_algo, GeneticAlgorithmParameters};


fn diff<T: Sub<Output = T> + Copy>(x: &Vec<T>) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i-1])
        .collect()
}

/// Summarizes various characteristics of two voltage time series,
/// one from a presynaptic neuron and one from a postsynaptic neuron
#[derive(Debug)]
pub struct ActionPotentialSummary {
    /// Average difference in timing between spikes from the presynaptic neuron
    pub average_pre_spike_time_difference: f32,
    /// Average difference in timing between spikes from the postsynaptic neuron
    pub average_post_spike_time_difference: f32,
    /// Number of spikes throughout voltage time series from presynaptic neuron
    pub num_pre_spikes: f32,
    /// Number of spikes throughout voltage time series from postsynaptic neuron
    pub num_post_spikes: f32,
}

/// Generates an action potential summary given two voltage time series
/// and a list of times where the neurons have spiked, `spike_amplitude_default`
/// refers to the default voltage to be used if no peaks are found
pub fn get_summary(
    pre_voltages: &Vec<f32>, 
    post_voltages: &Vec<f32>, 
    pre_peaks: &Vec<usize>,
    post_peaks: &Vec<usize>,
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    if pre_voltages.len() != post_voltages.len() {
        return Err(
            GeneticAlgorithmError::ObjectiveFunctionFailure(
                String::from("Voltage time series must be of the same length")
            )
        );
    }

    let average_pre_spike_difference: f32 = if pre_peaks.len() != 0 {
        diff(&pre_peaks).iter()
            .sum::<usize>() as f32 / (pre_peaks.len() as f32)
    } else {
        0.
    };

    let average_post_spike_difference: f32 = if post_peaks.len() != 0 {
        diff(&post_peaks).iter()
            .sum::<usize>() as f32 / (post_peaks.len() as f32)
    } else {
        0.
    };

    Ok(
        ActionPotentialSummary {
            average_pre_spike_time_difference: average_pre_spike_difference,
            average_post_spike_time_difference: average_post_spike_difference,
            num_pre_spikes: pre_peaks.len() as f32,
            num_post_spikes: post_peaks.len() as f32,
        }
    )
}

/// A set of defaults to use for scaling if no spikes are
/// found within inputs for [`fit_neuron_to_neuron`]
pub struct SummaryScalingDefaults {
    /// Default scaling for height of spikes
    pub default_amplitude_scale: f32,
    /// Default scaling for times between spikes
    pub default_time_difference_scale: f32,
    /// Default scaling for number of spikes
    pub default_num_peaks_scale: f32,
}

impl Default for SummaryScalingDefaults {
    fn default() -> Self {
        SummaryScalingDefaults {
            default_amplitude_scale: 70.,
            default_time_difference_scale: 800.,
            default_num_peaks_scale: 10.,
        }
    }
}

/// Scaling factors for action potential summaries used in [`fit_neuron_to_neuron`]
#[derive(Clone, Copy)]
pub struct SummaryScalingFactors {
    /// Scaling for times between spikes
    pub time_difference_scale: f32,
    /// Scaling for number of spikes
    pub num_peaks_scale: f32,
}

fn get_f32_max(x: &Vec<f32>) -> Option<&f32> {
    x.iter()
        .max_by(|a, b| a.total_cmp(b))
}

fn replace_with_default(value: f32, default: f32) -> f32 {
    if value == 0. {
        default
    } else {
        value
    }
}

/// Generates a new action potential summary and scaling factors to use to normalize
/// summary given a reference action potential summary and default scaling factors to
/// use if there are no spikes according to the action potential summary
pub fn get_reference_scale(
    reference_summary: &ActionPotentialSummary, 
    scaling_defaults: &SummaryScalingDefaults,
) -> (ActionPotentialSummary, SummaryScalingFactors) {
    let time_differences = vec![
        reference_summary.average_pre_spike_time_difference, reference_summary.average_post_spike_time_difference
    ];
    let peaks_lens = vec![
        reference_summary.num_pre_spikes, reference_summary.num_post_spikes,
    ];

    let time_difference_scale = replace_with_default(
        *get_f32_max(&time_differences).unwrap(), scaling_defaults.default_time_difference_scale
    );
    let num_peaks_scale = replace_with_default(
        *get_f32_max(&peaks_lens).unwrap(), scaling_defaults.default_num_peaks_scale
    );

    let scaled_reference = ActionPotentialSummary {
        average_pre_spike_time_difference: reference_summary.average_pre_spike_time_difference / time_difference_scale,
        average_post_spike_time_difference: reference_summary.average_post_spike_time_difference / time_difference_scale,
        num_pre_spikes: reference_summary.num_pre_spikes / num_peaks_scale,
        num_post_spikes: reference_summary.num_post_spikes / num_peaks_scale,
    };

    let scaling_factors = SummaryScalingFactors {
        time_difference_scale: time_difference_scale,
        num_peaks_scale: num_peaks_scale,
    };

    (scaled_reference, scaling_factors)
}

/// Scales summary given scaling factors
pub fn scale_summary(
    summary: &ActionPotentialSummary, 
    scaling_factors: &SummaryScalingFactors
) -> ActionPotentialSummary {
    ActionPotentialSummary {
        average_pre_spike_time_difference: summary.average_pre_spike_time_difference / scaling_factors.time_difference_scale,
        average_post_spike_time_difference: summary.average_post_spike_time_difference / scaling_factors.time_difference_scale,
        num_pre_spikes: summary.num_pre_spikes / scaling_factors.num_peaks_scale,
        num_post_spikes: summary.num_post_spikes / scaling_factors.num_peaks_scale,
    }
}

/// Compares the spike amplitudes, spike time differences, and number of spikes between action potentials
/// by summing the square of the difference between each field across summaries, if any value is not a number
/// `f32::INFINITY` is returned
pub fn compare_summary(summary1: &ActionPotentialSummary, summary2: &ActionPotentialSummary) -> f32 {
    let mut score = 0.;

    let pre_spike_difference = (summary1.average_pre_spike_time_difference - summary2.average_pre_spike_time_difference).powf(2.);
    let post_spike_difference = (summary1.average_post_spike_time_difference - summary2.average_post_spike_time_difference).powf(2.);

    let num_pre_spikes = (summary1.num_pre_spikes - summary2.num_pre_spikes).powf(2.);
    let num_post_spikes = (summary1.num_post_spikes - summary2.num_post_spikes).powf(2.);

    score += pre_spike_difference + post_spike_difference + num_pre_spikes + num_post_spikes;

    if score.is_nan() {
        f32::INFINITY
    } else {
        score
    }
}

/// Generates an action potential summary from coupled neurons where the presynaptic neuron
/// is coupled to a preset spike train, set `electrical_synapse` to `true` to update neurons based on
/// electrical gap junctions, set `chemical_synapse` to `true` to update receptor kinetics based
/// on neurotransmitter input or to `false` to not account for chemical neurotransmission, 
/// use `gaussian` to add normally distributed random noise
pub fn get_reference_summary<
    T: IterateAndSpike,
    U: SpikeTrain,
>(
    neuron: &T, 
    input_spike_train: &U, 
    iterations: usize,
    electrical_synapse: bool,
    chemical_synapse: bool,
    gaussian: bool, 
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    let mut current_spike_train = input_spike_train.clone();

    let mut presynaptic_neuron = neuron.clone();
    let mut postsynaptic_neuron = neuron.clone();

    let mut pre_voltages: Vec<f32> = vec![presynaptic_neuron.get_current_voltage()];
    let mut post_voltages: Vec<f32> = vec![postsynaptic_neuron.get_current_voltage()];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..iterations {
        let (_, pre_spiking, post_spiking) = iterate_coupled_spiking_neurons_and_spike_train(
            &mut current_spike_train,
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron, 
            timestep,
            electrical_synapse,
            chemical_synapse,
            gaussian, 
        );

        if pre_spiking {
            pre_peaks.push(timestep);
        };

        if post_spiking {
            post_peaks.push(timestep);
        };

        pre_voltages.push(presynaptic_neuron.get_current_voltage());
        post_voltages.push(postsynaptic_neuron.get_current_voltage());
    }

    Ok(
        get_summary(
            &pre_voltages, &post_voltages, &pre_peaks, &post_peaks,
        )?
    )
}

/// Settings used to scale action potential summary during [`fit_neuron_to_neuron`]
/// as well as settings used to run neuron to fit during fitting process
#[derive(Clone)]
pub struct FittingSettings<
    'a,  
    T: IterateAndSpike,
    U: SpikeTrain,
>{
    /// Neuron to fit to reference neuron
    pub neuron_to_fit: T,
    /// Spike trains to as input to the first neuron
    pub spike_trains: Vec<U>,
    /// Reference summaries to compare neuron to fit against
    pub action_potential_summary: &'a [ActionPotentialSummary],
    /// Scalars to use when comparing summaries
    pub scaling_factors: &'a [Option<SummaryScalingFactors>],
    /// Number of iterations to run simulation for
    pub iterations: usize,
    /// Use `true` to add normally distributed random noise to inputs of simulation
    pub gaussian: bool,
    /// Use `true` to neurons based on electrical gap junction input,
    pub electrical_synapse: bool,
    /// Use `true` to update receptor gating values of neurons based on neurotransmitter input,
    pub chemical_synapse: bool,
    /// Function that takes the decoded bitstring and generates the 
    /// neuron to fit based on those parameters
    pub converter: fn(&Vec<f32>) -> T,
}

/// Generates a summary of the neuron's action potentials over time
/// given a presynaptic and postsynaptic neuron
fn get_summary_given_settings<
    T: IterateAndSpike,
    U: SpikeTrain,
>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    settings: &FittingSettings<T, U>,
    index: usize,
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    let mut current_spike_train = settings.spike_trains[index].clone();

    let mut pre_voltages: Vec<f32> = vec![presynaptic_neuron.get_current_voltage()];
    let mut post_voltages: Vec<f32> = vec![postsynaptic_neuron.get_current_voltage()];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..settings.iterations {
        let (_, pre_spiking, post_spiking) = iterate_coupled_spiking_neurons_and_spike_train(
            &mut current_spike_train,
            presynaptic_neuron, 
            postsynaptic_neuron, 
            timestep,
            settings.electrical_synapse,
            settings.chemical_synapse,
            settings.gaussian, 
        );

        if pre_spiking {
            pre_peaks.push(timestep);
        };

        if post_spiking {
            post_peaks.push(timestep);
        };

        pre_voltages.push(presynaptic_neuron.get_current_voltage());
        post_voltages.push(postsynaptic_neuron.get_current_voltage());
    }

    let summary = get_summary(
        &pre_voltages, &post_voltages, &pre_peaks, &post_peaks
    )?;

    match settings.scaling_factors[index] {
        Some(factors) => Ok(scale_summary(&summary, &factors)),
        None => Ok(summary),
    }
}

// bounds should be a, b, c, d, v_th, and gap conductance for now
// if fitting does not generalize, optimize other coefs in equation
// or can try optimizing tau_m and c_m
fn fitting_objective<
    T: IterateAndSpike,
    U: SpikeTrain,
>(
    bitstring: &BitString, 
    bounds: &Vec<(f32, f32)>, 
    n_bits: usize, 
    settings: &HashMap<&str, FittingSettings<T, U>>
) -> result::Result<f32, GeneticAlgorithmError> {
    let settings = settings.get("settings").unwrap();

    let decoded = match decode(bitstring, bounds, n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(e) => return Err(e),
    };

    let test_cell = (settings.converter)(&decoded);

    let summaries_results = (0..settings.spike_trains.len())
        .map(|i| {
            get_summary_given_settings(
                &mut test_cell.clone(), 
                &mut test_cell.clone(), 
                settings,
                i
            )
        })
        .collect::<Vec<result::Result<ActionPotentialSummary, GeneticAlgorithmError>>>();

    for result in summaries_results.iter() {
        if let Err(_) = result {
            return Err(
                GeneticAlgorithmError::ObjectiveFunctionFailure(
                    String::from("Summary calculation could not be completed")
                )
            );
        }
    }

    let summaries = summaries_results.into_iter().map(|res| res.unwrap())
        .collect::<Vec<ActionPotentialSummary>>();

    let score = (0..settings.spike_trains.len())
        .map(|i| {
            compare_summary(&settings.action_potential_summary[i], &summaries[i])
        })
        .sum::<f32>();

    Ok(score)
}

/// Fits a given neuron to another given neuron by modulating a given set of parameters in a converter
/// function, returns an action potential summary for the reference model in second item of tuple,
/// returns an action potential summary for the neuron to fit in third item of tuple,
/// and returns scaling factors used during simulations in fourth item of tuple
/// 
/// - `neuron_to_fit` : neuron to simulate for fitting
/// 
/// - `reference_neuron` : neuron to reference as a target to meet
/// 
/// - `converter` : function to use to take decoded values and translate them to a neuron
/// 
/// - `scaling_defaults` : a set of default values to use when scaling action potential summaries,
/// use `None` to not scale summaries during fitting
/// 
/// - `iterations` : number of iterations to run each simulation for
/// 
/// - `input_spike_trains` : a set of preset spike trains to use when simulating each neuron, essentially
/// a set of conditions to observe the neurons over in order to ensure the models are fit
/// 
/// - `genetic_algo_parameters` : a set of hyperparameters for the genetic algorithm that fits
/// the neurons to use
/// 
/// - `reference_electrical_synapse` : use `true` to update neurons based on electrical gap junctions for
/// the reference neuron
/// 
/// - `reference_chemical_synapse` : use `true` to update receptor gating values of 
/// the neurons based on neurotransmitter input during the simulation for the reference neuron
/// 
/// - `neuron_to_fit_electrical_synapse` : use `true` to update neurons based on electrical gap junctions for
/// the neuron to fit
/// 
/// - `neuron_to_fit_chemical_synapse` : use `true` to update receptor gating values of 
/// the neurons based on neurotransmitter input during the simulation for the neuron to fit
/// 
/// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
/// 
/// - `verbose` : use `true` to print extra information
pub fn fit_neuron_to_neuron<
    T: IterateAndSpike,
    U: IterateAndSpike,
    V: SpikeTrain,
>(
    neuron_to_fit: &T,
    reference_neuron: &U,
    converter: fn(&Vec<f32>) -> T,
    scaling_defaults: Option<SummaryScalingDefaults>,
    iterations: usize,
    input_spike_trains: &Vec<V>,
    genetic_algo_params: &GeneticAlgorithmParameters,
    reference_electrical_synapse: bool,
    reference_chemical_synapse: bool,
    neuron_to_fit_electrical_synapse: bool,
    neuron_to_fit_chemical_synapse: bool,
    gaussian: bool,
    verbose: bool,
) -> result::Result<
    (
        Vec<f32>, 
        Vec<ActionPotentialSummary>, 
        Vec<ActionPotentialSummary>,
        Vec<Option<SummaryScalingFactors>>,
    ),
    GeneticAlgorithmError
> {
    let (reference_summaries, scaling_factors) = match scaling_defaults {
        Some(scaling_defaults_values) => {
            let mut reference_summaries: Vec<ActionPotentialSummary> = vec![];
            let mut scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![];

            for current_spike_train in input_spike_trains.iter() {
                let reference_summary = get_reference_summary(
                    reference_neuron, 
                    current_spike_train, 
                    iterations,
                    reference_electrical_synapse,
                    reference_chemical_synapse,
                    gaussian, 
                )?;

                let (reference_summary, scaling_factors) = get_reference_scale(
                    &reference_summary, &scaling_defaults_values
                );

                reference_summaries.push(reference_summary);
                scaling_factors_vector.push(Some(scaling_factors));
            }

            (reference_summaries, scaling_factors_vector)
        },
        None => {
            let mut reference_summaries: Vec<ActionPotentialSummary> = vec![];
            let scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![None; input_spike_trains.len()];

            for current_spike_train in input_spike_trains.iter() {
                let reference_summary = get_reference_summary(
                    reference_neuron, 
                    current_spike_train, 
                    iterations,
                    reference_electrical_synapse,
                    reference_chemical_synapse,
                    gaussian, 
                )?;

                reference_summaries.push(reference_summary);
            }

            (reference_summaries, scaling_factors_vector)
        }
    };

    let fitting_settings = FittingSettings {
        neuron_to_fit: neuron_to_fit.clone(),
        action_potential_summary: &reference_summaries.as_slice(),
        scaling_factors: &scaling_factors.as_slice(),
        spike_trains: input_spike_trains.clone(),
        iterations: iterations,
        gaussian: gaussian,
        electrical_synapse: neuron_to_fit_electrical_synapse,
        chemical_synapse: neuron_to_fit_chemical_synapse,
        converter: converter,
    };

    let mut fitting_settings_map: HashMap<&str, FittingSettings<T, V>> = HashMap::new();
    fitting_settings_map.insert("settings", fitting_settings.clone());

    if verbose {
        println!("Starting genetic algorithm...");
    }
    let (best_bitstring, _best_score, _scores) = genetic_algo(
        fitting_objective, 
        genetic_algo_params,
        &fitting_settings_map,
        verbose,
    )?;
    if verbose {
        println!("Finished genetic algorithm...");
    }

    let decoded = match decode(&best_bitstring, &genetic_algo_params.bounds, genetic_algo_params.n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(e) => return Err(e),
    };

    let test_cell = converter(&decoded);

    let summaries_results = (0..fitting_settings.spike_trains.len())
        .map(|i| {
            get_summary_given_settings(
                &mut test_cell.clone(), 
                &mut test_cell.clone(), 
                &fitting_settings,
                i
            )
        })
        .collect::<Vec<result::Result<ActionPotentialSummary, GeneticAlgorithmError>>>();

    for result in summaries_results.iter() {
        if let Err(e) = result {
            return Err(e.clone());
        }
    }

    let generated_summaries = summaries_results.into_iter().map(|res| res.unwrap())
        .collect::<Vec<ActionPotentialSummary>>();

    Ok((decoded, reference_summaries, generated_summaries, scaling_factors))
}

/// Prints out the given action potential summaries and scales them appropriately
/// based on the given set of scaling factors, length of action potential summaries
/// and the length of scaling factors must be the same
pub fn print_action_potential_summaries(
    summaries: &[ActionPotentialSummary], 
    scaling_factors: &[Option<SummaryScalingFactors>],
) -> Result<()> {
    if summaries.len() != scaling_factors.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "summaries and scaling_factors length must be the same"));
    }

    let mut pre_spike_time_differences: Vec<f32> = Vec::new();
    let mut post_spike_time_differences: Vec<f32> = Vec::new();
    let mut num_pre_spikes: Vec<f32> = Vec::new();
    let mut num_post_spikes: Vec<f32> = Vec::new();

    for (summary, scaling) in summaries.iter().zip(scaling_factors) {
        let (time_scaling, peaks_scaling) = match scaling {
            Some(value) => (value.time_difference_scale, value.num_peaks_scale),
            None => (1., 1.)
        };

        pre_spike_time_differences.push(summary.average_pre_spike_time_difference * time_scaling);
        post_spike_time_differences.push(summary.average_post_spike_time_difference * time_scaling);
        num_pre_spikes.push(summary.num_pre_spikes * peaks_scaling);
        num_post_spikes.push(summary.num_post_spikes * peaks_scaling);
    }

    println!("Presynaptic spike time Differences: {:?}", pre_spike_time_differences);
    println!("Postsynaptic spike time Differences: {:?}", post_spike_time_differences);
    println!("# of presynaptic spikes: {:?}", num_pre_spikes);
    println!("# of postsynaptic spikes: {:?}", num_post_spikes);

    Ok(())
}
