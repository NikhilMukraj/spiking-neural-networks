//! An implementation of a Hodgkin Huxley neuron with neurotransmission, receptor kinetics,
//! and the option to add an arbitrary additional ion channel gate to the model

use std::{
    collections::HashMap,
    ops::Sub,
};
use iterate_and_spike_traits::IterateAndSpikeBase;
use super::iterate_and_spike::{
    CurrentVoltage, DestexheNeurotransmitter, DestexheReceptor, GapConductance, 
    GaussianParameters, IonotropicNeurotransmitterType, IsSpiking, IterateAndSpike, 
    LastFiringTime, LigandGatedChannels, NeurotransmitterConcentrations, NeurotransmitterKinetics, 
    Neurotransmitters, ReceptorKinetics, Timestep
};
use super::ion_channels::{
    NaIonChannel, KIonChannel, KLeakChannel, IonChannel, TimestepIndependentIonChannel
};
use crate::neuron::intermediate_delegate::Intermediate;


// multicomparment stuff, refer to dopamine modeling paper as well
// https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/25282547/
// https://github.com/antgon/msn-model/blob/main/msn/cell.py 
// https://github.com/jrieke/NeuroSim
// MULTICOMPARTMENT EXPLAINED
// https://neuronaldynamics.epfl.ch/online/Ch3.S2.html

// dendrite should include receptors while soma should not
// in two compartmental model neurotransmitters are seperate from soma and dendrite
// in multicompartmental model with an axon neurotransmitters should be with the axon

// pub struct Soma {

// }

// pub struct Dendrite<R> {

// }

// there should be a gap junction between the dendrite and the soma to calculate how current travels
// for now current voltage could just be the sum of the two compartment's voltages
// pub TwoCompartmentNeuron<T, R> {
    // pub soma: Soma,
    // pub dendrite: Dendrite<R>,
    // pub synaptic_neurotransmitters: Neurotransmitters<T>
// }

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct HodgkinHuxleyNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32,
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32,
    /// Timestep (ms)
    pub dt: f32,
    /// Membrane capacitance (nF)
    pub c_m: f32,
    /// Sodium ion channel
    pub na_channel: NaIonChannel,
    /// Potassium ion channel
    pub k_channel: KIonChannel,
    /// Potassium leak channel
    pub k_leak_channel: KLeakChannel,
    /// Voltage threshold for spike calculation (mV)
    pub v_th: f32,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Whether the voltage was increasing in the last step
    pub was_increasing: bool,
    /// Whether the neuron is currently spiking
    pub is_spiking: bool,
    /// Parameters used in generating noise
    pub gaussian_params: GaussianParameters,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for HodgkinHuxleyNeuron<T, R> {
    fn default() -> Self {
        HodgkinHuxleyNeuron { 
            current_voltage: -65.,
            gap_conductance: 7.,
            dt: 0.01,
            c_m: 1., 
            na_channel: NaIonChannel::default(),
            k_channel: KIonChannel::default(),
            k_leak_channel: KLeakChannel::default(),
            v_th: 0.,
            last_firing_time: None,
            is_spiking: false,
            was_increasing: false,
            synaptic_neurotransmitters: Neurotransmitters::default(), 
            ligand_gates: LigandGatedChannels::default(),
            gaussian_params: GaussianParameters::default(),
        }
    }
}

impl HodgkinHuxleyNeuron<DestexheNeurotransmitter, DestexheReceptor> {
    /// Returns the default implementation of the neuron
    pub fn default_impl() -> Self {
        HodgkinHuxleyNeuron::default()
    }
}

fn diff<T: Sub<Output = T> + Copy>(x: &[T]) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i-1])
        .collect()
}

/// Returns indices of where voltages have peaked given a certain tolerance
pub fn find_peaks(voltages: &[f32], tolerance: f32) -> Vec<usize> {
    let first_diff: Vec<f32> = diff(voltages);
    let second_diff: Vec<f32> = diff(&first_diff);

    let local_optima = first_diff.iter()
        .enumerate()
        .filter(|(_, i)| i.abs() <= tolerance)
        .map(|(n, i)| (n, *i))
        .collect::<Vec<(usize, f32)>>();

    let local_maxima = local_optima.iter()
        .map(|(n, i)| (*n, *i))
        .filter(|(n, _)| *n < second_diff.len() - 1 && second_diff[n+1] < 0.)
        .collect::<Vec<(usize, f32)>>();

    let local_maxima: Vec<usize> = local_maxima.iter()
        .map(|(n, _)| (n + 2))
        .collect();

    let mut peak_spans: Vec<Vec<usize>> = Vec::new();

    let mut index: usize = 0;
    for (n, i) in local_maxima.iter().enumerate() {
        if n > 0 && local_maxima[n] - local_maxima[n-1] != 1 {
            index += 1;
        }

        if peak_spans.len() - 1 != index {
            peak_spans.push(Vec::new());
        }

        peak_spans[index].push(*i);
    }

    peak_spans.iter()
        .map(|i| i[i.len() / 2])
        .collect::<Vec<usize>>()
}

// https://github.com/swharden/pyHH/blob/master/src/pyhh/models.py
impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> HodgkinHuxleyNeuron<T, R> {
    /// Updates cell voltage given an input current
    pub fn update_cell_voltage(&mut self, input_current: f32) {
        let i_na = self.na_channel.current;
        let i_k = self.k_channel.current;
        let i_k_leak = self.k_leak_channel.current;

        let i_ligand_gates = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        let i_sum = input_current - (i_na + i_k + i_k_leak);
        self.current_voltage += self.dt * i_sum / self.c_m - i_ligand_gates;
    }

    /// Updates neurotransmitter concentrations based on membrane potential
    pub fn update_neurotransmitters(&mut self) {
        self.synaptic_neurotransmitters.apply_t_changes(&Intermediate::from_neuron(self));
    }

    /// Updates receptor gating based on neurotransmitter input
    pub fn update_receptors(
        &mut self, 
        t_total: &NeurotransmitterConcentrations<IonotropicNeurotransmitterType>
    ) {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);
    }

    /// Updates additional ion channels
    pub fn update_gates(&mut self) {
        self.na_channel.update_current(self.current_voltage, self.dt);
        self.k_channel.update_current(self.current_voltage, self.dt);
        self.k_leak_channel.update_current(self.current_voltage);
    }

    fn iterate(&mut self, input: f32) {
        self.update_gates();
        self.update_cell_voltage(input);
        self.update_neurotransmitters();
    }

    fn iterate_with_neurotransmitter(
        &mut self, 
        input: f32, 
        t_total: &NeurotransmitterConcentrations<IonotropicNeurotransmitterType>
    ) {
        self.update_receptors(t_total);
        self.iterate(input);
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for HodgkinHuxleyNeuron<T, R> {
    type N = IonotropicNeurotransmitterType;

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate(input_current);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }

    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<IonotropicNeurotransmitterType> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<IonotropicNeurotransmitterType>,
    ) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate_with_neurotransmitter(input_current, t_total);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }
}

/// Takes in a static current as an input and iterates the given
/// neuron for a given duration, set `gaussian` to true to add 
/// normally distributed noise to the input as it iterates,
/// returns various state variables over time including voltages
/// and gating states, output hashmap has keys `""`,
/// `"m"`, `"n"`, and `"h"`
pub fn run_static_input_hodgkin_huxley<T: NeurotransmitterKinetics, R: ReceptorKinetics>(
    hodgkin_huxley_neuron: &mut HodgkinHuxleyNeuron<T, R>,
    input_current: f32,
    iterations: usize,
    gaussian: Option<GaussianParameters>,
) -> HashMap<String, Vec<f32>> {
    let mut state_output = HashMap::new();
    state_output.insert("current_voltage".to_string(), vec![]);
    state_output.insert("m".to_string(), vec![]);
    state_output.insert("n".to_string(), vec![]);
    state_output.insert("h".to_string(), vec![]);

    for _ in 0..iterations {
        let _is_spiking = match gaussian {
            Some(ref params) => hodgkin_huxley_neuron.iterate_and_spike(params.get_random_number() * input_current),
            None => hodgkin_huxley_neuron.iterate_and_spike(input_current),
        };

        if let Some(val) = state_output.get_mut("current_voltage") { val.push(hodgkin_huxley_neuron.current_voltage) }
        if let Some(val) = state_output.get_mut("m") { val.push(hodgkin_huxley_neuron.na_channel.m.state) }
        if let Some(val) = state_output.get_mut("n") { val.push(hodgkin_huxley_neuron.na_channel.h.state) }
        if let Some(val) = state_output.get_mut("h") { val.push(hodgkin_huxley_neuron.k_channel.n.state) }
    }

    state_output
}
