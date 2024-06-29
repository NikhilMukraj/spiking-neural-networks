//! A set of tools to fit a given Izhikevich neuron to a Hodgkin Huxley neuron.

use std::{
    collections::HashMap,
    result,
    io::{Error, ErrorKind, Result},
    ops::Sub,
};
use crate::error::{GeneticAlgorithmError, GeneticAlgorithmErrorKind};
use crate::neuron::{   
    hodgkin_huxley::HodgkinHuxleyNeuron, 
    integrate_and_fire::IzhikevichNeuron, 
    iterate_and_spike::{NeurotransmitterKinetics, ReceptorKinetics}, 
    spike_train::{PresetSpikeTrain, NeuralRefractoriness},
    iterate_coupled_spiking_neurons_and_spike_train,
};
use crate::ga::{BitString, decode, genetic_algo, GeneticAlgorithmParameters};


fn diff<T: Sub<Output = T> + Copy>(x: &Vec<T>) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i-1])
        .collect()
}

fn get_average_spike(peaks: &Vec<usize>, voltages: &Vec<f64>, default: f64) -> f64 {
    if peaks.len() == 0 {
        return default;
    }

    peaks.iter()
        .map(|n| voltages[*n])
        .sum::<f64>() / (peaks.len() as f64)
}

/// Summarizes various characteristics of two voltage time series,
/// one from a presynaptic neuron and one from a postsynaptic neuron
#[derive(Debug)]
pub struct ActionPotentialSummary {
    /// Average height of the presynaptic spikes (mV)
    pub average_pre_spike_amplitude: f64,
    /// Average height of the postynaptic spikes (mV)
    pub average_post_spike_amplitude: f64,
    /// Average difference in timing between spikes from the presynaptic neuron
    pub average_pre_spike_time_difference: f64,
    /// Average difference in timing between spikes from the postsynaptic neuron
    pub average_post_spike_time_difference: f64,
    /// Number of spikes throughout voltage time series from presynaptic neuron
    pub num_pre_spikes: f64,
    /// Number of spikes throughout voltage time series from postsynaptic neuron
    pub num_post_spikes: f64,
}

/// Generates an action potential summary given two voltage time series
/// and a list of times where the neurons have spiked, `spike_amplitude_default`
/// refers to the default voltage to be used if no peaks are found
pub fn get_summary(
    pre_voltages: &Vec<f64>, 
    post_voltages: &Vec<f64>, 
    pre_peaks: &Vec<usize>,
    post_peaks: &Vec<usize>,
    spike_amplitude_default: f64,
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    if pre_voltages.len() != post_voltages.len() {
        return Err(
            GeneticAlgorithmError::new(
                GeneticAlgorithmErrorKind::ObjectiveFunctionFailure(
                    String::from("Voltage time series must be of the same length"),
                ), file!(), line!()
            )
        );
    }

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

/// A set of defaults to use for scaling if no spikes are
/// found within inputs for [`fit_izhikevich_to_hodgkin_huxley`]
pub struct SummaryScalingDefaults {
    /// Default scaling for height of spikes
    pub default_amplitude_scale: f64,
    /// Default scaling for times between spikes
    pub default_time_difference_scale: f64,
    /// Default scaling for number of spikes
    pub default_num_peaks_scale: f64,
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

/// Scaling factors for action potential summaries used in [`fit_izhikevich_to_hodgkin_huxley`]
#[derive(Clone, Copy)]
pub struct SummaryScalingFactors {
    /// Scaling for height of spikes
    pub amplitude_scale: f64,
    /// Scaling for times between spikes
    pub time_difference_scale: f64,
    /// Scaling for number of spikes
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

/// Generates a new action potential summary and scaling factors to use to normalize
/// summary given a reference action potential summary and default scaling factors to
/// use if there are no spikes according to the action potential summary
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

/// Scales summary given scaling factors
pub fn scale_summary(
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

/// Compares the spike amplitudes, spike time differences, and number of spikes between action potentials
/// by summing the square of the difference between each field across summaries, if any value is not a number
/// `f64::INFINITY` is returned
pub fn compare_summary(summary1: &ActionPotentialSummary, summary2: &ActionPotentialSummary, use_amplitudes: bool) -> f64 {
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

    score += pre_spike_difference + post_spike_difference + num_pre_spikes + num_post_spikes;

    if score.is_nan() {
        f64::INFINITY
    } else {
        score
    }
}

/// Generates an action potential summary from coupled Hodgkin Huxley neurons where the presynaptic neuron
/// is coupled to a preset spike train, set `do_receptor_kinetics` to `true` to update receptor kinetics based
/// on neurotransmitter input or to `false` to keep it state, use `gaussian` to add normally distributed 
/// random noise, use `spike_amplitude_default` to set a default spike height if no spikes are generated,
/// use `resting_potential` to set the resting potential of both Hodgkin Huxley neurons (mV)
pub fn get_hodgkin_huxley_summary<
    T: NeurotransmitterKinetics, 
    U: ReceptorKinetics, 
    V: NeurotransmitterKinetics,
    W: NeuralRefractoriness,
>(
    hodgkin_huxley_neuron: &HodgkinHuxleyNeuron<T, U>, 
    input_spike_train: &PresetSpikeTrain<V, W>, 
    iterations: usize,
    do_receptor_kinetics: bool,
    gaussian: bool, 
    spike_amplitude_default: f64,
    resting_potential: f64
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    let mut current_spike_train = input_spike_train.clone();

    let mut presynaptic_neuron = hodgkin_huxley_neuron.clone();
    let mut postsynaptic_neuron = hodgkin_huxley_neuron.clone();

    presynaptic_neuron.initialize_parameters(presynaptic_neuron.current_voltage);
    postsynaptic_neuron.initialize_parameters(postsynaptic_neuron.current_voltage);

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..iterations {
        let (_, pre_spiking, post_spiking) = iterate_coupled_spiking_neurons_and_spike_train(
            &mut current_spike_train,
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron, 
            timestep,
            do_receptor_kinetics,
            gaussian, 
        );

        if pre_spiking {
            pre_peaks.push(timestep);
        };

        if post_spiking {
            post_peaks.push(timestep);
        };

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    let pre_voltages = pre_voltages.iter()
        .map(|i| i + resting_potential)
        .collect();
    let post_voltages = post_voltages.iter()
        .map(|i| i + resting_potential)
        .collect();

    Ok(
        get_summary(
            &pre_voltages, &post_voltages, &pre_peaks, &post_peaks, spike_amplitude_default
        )?
    )
}

/// Settings used to scale action potential summary during [`fit_izhikevich_to_hodgkin_huxley`]
/// as well as settings used to run Izhikevich neuron during fitting process
#[derive(Clone)]
pub struct FittingSettings<
    'a, 
    T: NeurotransmitterKinetics, 
    U: ReceptorKinetics, 
    V: NeurotransmitterKinetics,
    W: NeuralRefractoriness,
>{
    /// Izhikevich neuron to reference for parameters during fitting
    pub izhikevich_neuron: IzhikevichNeuron<T, U>,
    /// Spike trains to use when simulating Izhikevich neurons
    pub spike_trains: Vec<PresetSpikeTrain<V, W>>,
    /// Reference summaries to compare Izhikevich neuron against
    pub action_potential_summary: &'a [ActionPotentialSummary],
    /// Scalars to use when comparing summaries
    pub scaling_factors: &'a [Option<SummaryScalingFactors>],
    /// Whether or not to include the average amplitudes as a parameter when
    /// calculating similarity of action potential summaries
    pub use_amplitude: bool,
    /// Default value to use if no spikes are found
    pub spike_amplitude_default: f64,
    /// Number of iterations to run simulation for
    pub iterations: usize,
    /// Use `true` to add normally distributed random noise to inputs of simulation
    pub gaussian: bool,
    /// Use `true` to update receptor gating values of Izhikevich neuron based on neurotransmitter input,
    /// use `false` to keep gating values static throughout simulation
    pub do_receptor_kinetics: bool,
}

/// Generates a summary of the Izhikevich neuron's action potentials over time
/// given a presynaptic and postsynaptic neuron
pub fn get_izhikevich_summary<
    T: NeurotransmitterKinetics, 
    U: ReceptorKinetics, 
    V: NeurotransmitterKinetics, 
    W: NeuralRefractoriness
>(
    presynaptic_neuron: &mut IzhikevichNeuron<T, U>, 
    postsynaptic_neuron: &mut IzhikevichNeuron<T, U>,
    settings: &FittingSettings<T, U, V, W>,
    index: usize,
) -> result::Result<ActionPotentialSummary, GeneticAlgorithmError> {
    let mut current_spike_train = settings.spike_trains[index].clone();

    let mut pre_voltages: Vec<f64> = vec![presynaptic_neuron.current_voltage];
    let mut post_voltages: Vec<f64> = vec![postsynaptic_neuron.current_voltage];

    let mut pre_peaks: Vec<usize> = vec![];
    let mut post_peaks: Vec<usize> = vec![];

    for timestep in 0..settings.iterations {
        let (_, pre_spiking, post_spiking) = iterate_coupled_spiking_neurons_and_spike_train(
            &mut current_spike_train,
            presynaptic_neuron, 
            postsynaptic_neuron, 
            timestep,
            settings.do_receptor_kinetics,
            settings.gaussian, 
        );

        if pre_spiking {
            pre_peaks.push(timestep);
        };

        if post_spiking {
            post_peaks.push(timestep);
        };

        pre_voltages.push(presynaptic_neuron.current_voltage);
        post_voltages.push(postsynaptic_neuron.current_voltage);
    }

    let summary = get_summary(
        &pre_voltages, &post_voltages, &pre_peaks, &post_peaks, settings.spike_amplitude_default
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
    T: NeurotransmitterKinetics, 
    U: ReceptorKinetics,
    W: NeurotransmitterKinetics,
    V: NeuralRefractoriness,
>(
    bitstring: &BitString, 
    bounds: &Vec<(f64, f64)>, 
    n_bits: usize, 
    settings: &HashMap<&str, FittingSettings<T, U, W, V>>
) -> result::Result<f64, GeneticAlgorithmError> {
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

    let mut test_cell = settings.izhikevich_neuron.clone();
    test_cell.v_th = v_th;
    test_cell.gap_conductance = gap_conductance;
    test_cell.a = a;
    test_cell.b = b;
    test_cell.c = c;
    test_cell.d = d;

    let summaries_results = (0..settings.spike_trains.len())
        .map(|i| {
            get_izhikevich_summary(
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
                GeneticAlgorithmError::new(
                    GeneticAlgorithmErrorKind::ObjectiveFunctionFailure(
                        String::from("Summary calculation could not be completed")
                    ), file!(), line!()
                )
            );
        }
    }

    let summaries = summaries_results.into_iter().map(|res| res.unwrap())
        .collect::<Vec<ActionPotentialSummary>>();

    let score = (0..settings.spike_trains.len())
        .map(|i| {
            compare_summary(&settings.action_potential_summary[i], &summaries[i], settings.use_amplitude)
        })
        .sum::<f64>();

    Ok(score)
}

/// Fits a given Izhikevich neuron to a given Hodgkin Huxley by modulating the
/// `a`, `b`, `c`, `d`, `v_th`, and `gap_conductance` parameters, 
/// returns `a`, `b`, `c`, `d`, `v_th`, and `gap_conductance` in first item of tuple,
/// returns an action potential summary for the reference Hodgkin Huxley model in second item of tuple,
/// returns an action potential summary for the fit Izhikevich neuron in third item of tuple,
/// and returns scaling factors used during simulations in fourth item of tuple
/// 
/// - `izhikevich_neuron` : Izhikevich neuron to simulate for fitting
/// 
/// - `hodgkin_huxley_neuron` : Hodgkin Huxley neuron to reference
/// 
/// - `scaling_defaults` : a set of default values to use when scaling action potential summaries,
/// use `None` to not scale summaries during fitting
/// 
/// - `iterations` : number of iterations to run each Hodgkin Huxley and Izhikevich simulation for
/// 
/// - `input_spike_trains` : a set of preset spike trains to use when simulating each neuron, essentially
/// a set of conditions to observe the neurons over in order to ensure the models are fit
/// 
/// - `genetic_algo_parameters` : a set of hyperparameters for the genetic algorithm that fits
/// the neurons to use
/// 
/// - `hodgkin_huxley_do_receptor_kinetics` : use `true` to update receptor gating values of 
/// Hodgkin Huxley neuron based on neurotransmitter input during the simulation
/// 
/// - `izhikevich_do_receptor_kinetics` : use `true` to update receptor gating values of 
/// Izhikevich neuron based on neurotransmitter input during the simulation
/// 
/// - `resting_potential` : resting potential of the Hodgkin Huxley neuron
/// 
/// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
/// 
/// - `use_amplitude` : use `true` to compare the average spike amplitudes of the simulations
/// in the fitting function
/// 
/// - `spike_amplitude_default` : default height for a spike if no spikes are found in a given simulation
/// 
/// - `verbose` : use `true` to print extra information
pub fn fit_izhikevich_to_hodgkin_huxley<
    T: NeurotransmitterKinetics, 
    U: NeurotransmitterKinetics, 
    V: NeurotransmitterKinetics, 
    W: ReceptorKinetics,
    Y: ReceptorKinetics,
    X: NeuralRefractoriness,
>(
    izhikevich_neuron: &IzhikevichNeuron<T, W>,
    hodgkin_huxley_neuron: &HodgkinHuxleyNeuron<U, Y>,
    scaling_defaults: Option<SummaryScalingDefaults>,
    iterations: usize,
    input_spike_trains: &Vec<PresetSpikeTrain<V, X>>,
    genetic_algo_params: &GeneticAlgorithmParameters,
    hodgkin_huxley_do_receptor_kinetics: bool,
    izhikevich_do_receptor_kinetics: bool,
    resting_potential: f64,
    gaussian: bool,
    use_amplitude: bool,
    spike_amplitude_default: f64,
    debug: bool,
) -> result::Result<
    (
        (f64, f64, f64, f64, f64, f64), 
        Vec<ActionPotentialSummary>, 
        Vec<ActionPotentialSummary>,
        Vec<Option<SummaryScalingFactors>>,
    ),
    GeneticAlgorithmError
> {
    let (hodgkin_huxley_summaries, scaling_factors) = match scaling_defaults {
        Some(scaling_defaults_values) => {
            let mut hodgkin_huxley_summaries: Vec<ActionPotentialSummary> = vec![];
            let mut scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![];

            for current_spike_train in input_spike_trains.iter() {
                let hodgkin_huxley_summary = get_hodgkin_huxley_summary(
                    &hodgkin_huxley_neuron, 
                    &current_spike_train, 
                    iterations,
                    hodgkin_huxley_do_receptor_kinetics,
                    gaussian, 
                    spike_amplitude_default,
                    resting_potential,
                )?;

                let (hodgkin_huxley_summary, scaling_factors) = get_reference_scale(
                    &hodgkin_huxley_summary, &scaling_defaults_values
                );

                hodgkin_huxley_summaries.push(hodgkin_huxley_summary);
                scaling_factors_vector.push(Some(scaling_factors));
            }

            (hodgkin_huxley_summaries, scaling_factors_vector)
        },
        None => {
            let mut hodgkin_huxley_summaries: Vec<ActionPotentialSummary> = vec![];
            let scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![None; input_spike_trains.len()];

            for current_spike_train in input_spike_trains.iter() {
                let hodgkin_huxley_summary = get_hodgkin_huxley_summary(
                    &hodgkin_huxley_neuron, 
                    &current_spike_train, 
                    iterations,
                    hodgkin_huxley_do_receptor_kinetics,
                    gaussian, 
                    spike_amplitude_default,
                    resting_potential,
                )?;

                hodgkin_huxley_summaries.push(hodgkin_huxley_summary);
            }

            (hodgkin_huxley_summaries, scaling_factors_vector)
        }
    };

    let fitting_settings = FittingSettings {
        izhikevich_neuron: izhikevich_neuron.clone(),
        action_potential_summary: &hodgkin_huxley_summaries.as_slice(),
        scaling_factors: &scaling_factors.as_slice(),
        use_amplitude: use_amplitude,
        spike_amplitude_default: spike_amplitude_default,
        spike_trains: input_spike_trains.clone(),
        iterations: iterations,
        gaussian: gaussian,
        do_receptor_kinetics: izhikevich_do_receptor_kinetics,
    };

    let mut fitting_settings_map: HashMap<&str, FittingSettings<T, W, V, X>> = HashMap::new();
    fitting_settings_map.insert("settings", fitting_settings.clone());

    if debug {
        println!("Starting genetic algorithm...");
    }
    let (best_bitstring, _best_score, _scores) = genetic_algo(
        fitting_objective, 
        genetic_algo_params,
        &fitting_settings_map,
        debug,
    )?;
    if debug {
        println!("Finished genetic algorithm...");
    }

    let decoded = match decode(&best_bitstring, &genetic_algo_params.bounds, genetic_algo_params.n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(e) => return Err(e),
    };

    let a: f64 = decoded[0];
    let b: f64 = decoded[1];
    let c: f64 = decoded[2];
    let d: f64 = decoded[3];
    let v_th: f64 = decoded[4];
    let gap_conductance: f64 = decoded[5];

    let mut test_cell = izhikevich_neuron.clone();

    test_cell.a = a;
    test_cell.b = b;
    test_cell.c = c;
    test_cell.d = d;
    test_cell.v_th = v_th;
    test_cell.gap_conductance = gap_conductance;

        let summaries_results = (0..fitting_settings.spike_trains.len())
            .map(|i| {
                get_izhikevich_summary(
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


    Ok(((a, b, c, d, v_th, gap_conductance), hodgkin_huxley_summaries, generated_summaries, scaling_factors))
}

/// Prints out the given action potential summaries and scales them appropriately
/// based on the given set of scaling factors, length of action potential summaries
/// and the length of scaling factors must be the same set `use_amplitude` to `false` to
/// not print out amplitude related statistics
pub fn print_action_potential_summaries(
    summaries: &[ActionPotentialSummary], 
    scaling_factors: &[Option<SummaryScalingFactors>],
    use_amplitude: bool,
) -> Result<()> {
    if summaries.len() != scaling_factors.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "summaries and scaling_factors length must be the same"));
    }

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

    Ok(())
}
