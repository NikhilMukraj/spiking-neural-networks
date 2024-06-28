//! A few implementations of different spike trains that can be coupled with `IterateAndSpike`
//! based neurons.

use std::collections::{HashMap, HashSet};
use rand::Rng;
use super::{
    iterate_and_spike::{ApproximateNeurotransmitter, NeurotransmitterKinetics}, 
    CurrentVoltage, LastFiringTime, NeurotransmitterType, 
    Neurotransmitters, Potentiation, PotentiationType,
};

/// Handles dynamics of spike train effect on another neuron given the current timestep
/// of the simulation (neural refractoriness function), when the spike train spikes
/// the total effect also spikes while every subsequent iteration after that spike
/// results in the effect decaying back to a resting point mimicking an action potential
pub trait NeuralRefractoriness: Default + Clone + Send + Sync {
    /// Sets decay value
    fn set_decay(&mut self, decay_factor: f64);
    /// Gets decay value
    fn get_decay(&self) -> f64;
    /// Calculates neural refractoriness based on the current time of the simulation, the 
    /// last spiking time, the maximum and minimum voltage (mV)
    /// for scaling, and the simulation timestep (ms)
    fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f64, v_resting: f64, dt: f64) -> f64;
}

macro_rules! impl_default_neural_refractoriness {
    ($name:ident, $effect:expr) => {
        impl Default for $name {
            fn default() -> Self {
                $name {
                    k: 1000.,
                }
            }
        }

        impl NeuralRefractoriness for $name {
            fn set_decay(&mut self, decay_factor: f64) {
                self.k = decay_factor;
            }

            fn get_decay(&self) -> f64 {
                self.k
            }

            fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f64, v_resting: f64, dt: f64) -> f64 {
                let a = v_max - v_resting;
                let time_difference = (timestep - last_firing_time) as f64;

                $effect(self.k, a, time_difference, v_resting, dt)
            }
        }
    };
}

/// Calculates refractoriness based on the delta dirac function
#[derive(Debug, Clone, Copy)]
pub struct DeltaDiracRefractoriness {
    /// Decay value
    pub k: f64,
}

fn delta_dirac_effect(k: f64, a: f64, time_difference: f64, v_resting: f64, dt: f64) -> f64 {
    a * ((-1. / (k / dt)) * time_difference.powf(2.)).exp() + v_resting
}

impl_default_neural_refractoriness!(DeltaDiracRefractoriness, delta_dirac_effect);

/// Calculates refactoriness based on exponential decay
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayRefractoriness {
    /// Decay value
    pub k: f64
}

fn exponential_decay_effect(k: f64, a: f64, time_difference: f64, v_resting: f64, dt: f64) -> f64 {
    a * ((-1. / (k / dt)) * time_difference).exp() + v_resting
}

impl_default_neural_refractoriness!(ExponentialDecayRefractoriness, exponential_decay_effect);

/// Handles spike train dynamics
pub trait SpikeTrain: CurrentVoltage + Potentiation + LastFiringTime + Clone {
    type T: NeurotransmitterKinetics;
    type U: NeuralRefractoriness;
    /// Updates spike train
    fn iterate(&mut self) -> bool;
    /// Gets maximum and minimum voltage values
    fn get_height(&self) -> (f64, f64);
    /// Gets timestep or `dt` (ms)
    fn get_refractoriness_timestep(&self) -> f64;
    /// Returns neurotransmitters
    fn get_neurotransmitters(&self) -> &Neurotransmitters<Self::T>;
    /// Returns neurotransmitter concentrations
    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64>;
    /// Returns refractoriness dynamics
    fn get_refractoriness_function(&self) -> &Self::U;
}

macro_rules! impl_current_voltage_spike_train {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> CurrentVoltage for $struct<T, U> {
            fn get_current_voltage(&self) -> f64 {
                self.current_voltage
            }
        }
    };
}

macro_rules! impl_potentiation_spike_train {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Potentiation for $struct<T, U> {
            fn get_potentiation_type(&self) -> PotentiationType {
                self.potentiation_type
            }
        }
    };
}


macro_rules! impl_last_firing_time_spike_train {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> LastFiringTime for $struct<T, U> {
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.last_firing_time = timestep;
            }
        
            fn get_last_firing_time(&self) -> Option<usize> {
                self.last_firing_time
            }
        }
    };
}

/// A Poisson neuron
#[derive(Debug, Clone)]
pub struct PoissonNeuron<T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f64,
    /// Maximum voltage (mV)
    pub v_th: f64,
    /// Minimum voltage (mV)
    pub v_resting: f64,
    /// Last firing time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    /// Potentiation type of neuron
    pub potentiation_type: PotentiationType,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Chance of neuron firing at a given timestep
    pub chance_of_firing: f64,
    /// Timestep for refractoriness (ms)
    pub refractoriness_dt: f64,
}

macro_rules! impl_default_spike_train_methods {
    () => {
        type T = T;
        type U = U;

        fn get_height(&self) -> (f64, f64) {
            (self.v_th, self.v_resting)
        }
    
        fn get_refractoriness_timestep(&self) -> f64 {
            self.refractoriness_dt
        }
    
        fn get_neurotransmitters(&self) -> &Neurotransmitters<Self::T> {
            &self.synaptic_neurotransmitters
        }
    
        fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    
        fn get_refractoriness_function(&self) -> &Self::U {
            &self.neural_refractoriness
        }
    }
}

impl_current_voltage_spike_train!(PoissonNeuron);
impl_potentiation_spike_train!(PoissonNeuron);
impl_last_firing_time_spike_train!(PoissonNeuron);

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PoissonNeuron<T, U> {
    fn default() -> Self {
        PoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            potentiation_type: PotentiationType::Excitatory,
            neural_refractoriness: U::default(),
            chance_of_firing: 0.01,
            refractoriness_dt: 0.1,
        }
    }
}

impl PoissonNeuron<ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f64, dt: f64) -> Self {
        PoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> PoissonNeuron<T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f64, dt: f64) -> Self {
        let mut poisson_neuron = PoissonNeuron::<T, U>::default();

        poisson_neuron.refractoriness_dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.refractoriness_dt) / hertz);

        poisson_neuron
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PoissonNeuron<T, U> {
    fn iterate(&mut self) -> bool {
        let is_spiking = if rand::thread_rng().gen_range(0.0..=1.0) <= self.chance_of_firing {
            self.current_voltage = self.v_th;

            true
        } else {
            self.current_voltage = self.v_resting;

            false
        };

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        is_spiking
    }

    impl_default_spike_train_methods!();
}

/// A preset spike train that has a set of designated firing times and an internal clock,
/// the internal clock is updated every iteration and once the internal clock reaches one of the 
/// firing times the neuron fires, the internal clock is reset once it reaches the maximum value
/// and cyclees through the designated firing times again
#[derive(Debug, Clone)]
pub struct PresetSpikeTrain<T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f64,
    /// Maximum voltage (mV)
    pub v_th: f64,
    /// Minimum voltage (mV)
    pub v_resting: f64,
    /// Last spiking time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    /// Potentiation type of neuron
    pub potentiation_type: PotentiationType,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Set of times to fire at
    pub firing_times: HashSet<usize>,
    /// Internal clock to track when to fire
    pub internal_clock: usize,
    /// Value to reset internal clock at
    pub max_clock_value: usize,
    /// Timestep for refractoriness (ms)
    pub refractoriness_dt: f64,
}

impl_current_voltage_spike_train!(PresetSpikeTrain);
impl_potentiation_spike_train!(PresetSpikeTrain);
impl_last_firing_time_spike_train!(PresetSpikeTrain);

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PresetSpikeTrain<T, U> {
    fn default() -> Self {
        PresetSpikeTrain {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            potentiation_type: PotentiationType::Excitatory,
            neural_refractoriness: U::default(),
            firing_times: HashSet::from([100, 300, 500]),
            internal_clock: 0,
            max_clock_value: 600,
            refractoriness_dt: 0.1,
        }
    }
}

impl PresetSpikeTrain<ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PresetSpikeTrain::default()
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> PresetSpikeTrain<T, U> {
    /// Generates a spike train that evenly divides up a preset spike train's
    /// firing times across a timeframe given the number of spikes
    /// and the timestep (dt)
    pub fn from_evenly_divided(num_spikes: usize, dt: f64) -> Self {
        let mut firing_times: HashSet<usize> =  HashSet::new();
        let interval = ((1000. / dt) / (num_spikes as f64)) as usize;

        let mut current_timestep = 0;
        for _ in 0..num_spikes {
            firing_times.insert(current_timestep);
            current_timestep += interval;
        }

        let mut preset_spike_train = PresetSpikeTrain::<T, U>::default();
        preset_spike_train.refractoriness_dt = dt;
        preset_spike_train.firing_times = firing_times;
        preset_spike_train.max_clock_value = current_timestep;

        preset_spike_train
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PresetSpikeTrain<T, U> {
    fn iterate(&mut self) -> bool {
        self.internal_clock += 1;
        if self.internal_clock > self.max_clock_value {
            self.internal_clock = 0;
        }

        let is_spiking = if self.firing_times.contains(&self.internal_clock) {
            self.current_voltage = self.v_th;

            true
        } else {
            self.current_voltage = self.v_resting;

            false
        };

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        is_spiking
    }

    impl_default_spike_train_methods!();
}
