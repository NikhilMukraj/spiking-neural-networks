//! A few implementations of different spike trains that can be coupled with `IterateAndSpike`
//! based neurons.

use rand::Rng;
use super::iterate_and_spike::{
    ApproximateNeurotransmitter, CurrentVoltage, IsSpiking, LastFiringTime, 
    NeurotransmitterConcentrations, NeurotransmitterKinetics, Neurotransmitters, Timestep,
};
use super::iterate_and_spike_traits::Timestep;

/// Handles dynamics of spike train effect on another neuron given the current timestep
/// of the simulation (neural refractoriness function), when the spike train spikes
/// the total effect also spikes while every subsequent iteration after that spike
/// results in the effect decaying back to a resting point mimicking an action potential
pub trait NeuralRefractoriness: Default + Clone + Send + Sync {
    /// Sets decay value
    fn set_decay(&mut self, decay_factor: f32);
    /// Gets decay value
    fn get_decay(&self) -> f32;
    /// Calculates neural refractoriness based on the current time of the simulation, the 
    /// last spiking time, the maximum and minimum voltage (mV)
    /// for scaling, and the simulation timestep (ms)
    fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32;
}

macro_rules! impl_default_neural_refractoriness {
    ($name:ident, $effect:expr) => {
        impl Default for $name {
            fn default() -> Self {
                $name {
                    k: 10000.,
                }
            }
        }

        impl NeuralRefractoriness for $name {
            fn set_decay(&mut self, decay_factor: f32) {
                self.k = decay_factor;
            }

            fn get_decay(&self) -> f32 {
                self.k
            }

            fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32 {
                let a = v_max - v_resting;
                let time_difference = (timestep - last_firing_time) as f32;

                $effect(self.k, a, time_difference, v_resting, dt)
            }
        }
    };
}

/// Calculates refractoriness based on the delta dirac function
#[derive(Debug, Clone, Copy)]
pub struct DeltaDiracRefractoriness {
    /// Decay value
    pub k: f32,
}

fn delta_dirac_effect(k: f32, a: f32, time_difference: f32, v_resting: f32, dt: f32) -> f32 {
    a * ((-1. / (k / dt)) * time_difference.powf(2.)).exp() + v_resting
}

impl_default_neural_refractoriness!(DeltaDiracRefractoriness, delta_dirac_effect);

/// Calculates refactoriness based on exponential decay
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayRefractoriness {
    /// Decay value
    pub k: f32
}

fn exponential_decay_effect(k: f32, a: f32, time_difference: f32, v_resting: f32, dt: f32) -> f32 {
    a * ((-1. / (k / dt)) * time_difference).exp() + v_resting
}

impl_default_neural_refractoriness!(ExponentialDecayRefractoriness, exponential_decay_effect);

/// Handles spike train dynamics
pub trait SpikeTrain: CurrentVoltage + IsSpiking + LastFiringTime + Timestep + Clone + Send + Sync {
    type U: NeuralRefractoriness;
    /// Updates spike train
    fn iterate(&mut self) -> bool;
    /// Gets maximum and minimum voltage values
    fn get_height(&self) -> (f32, f32);
    /// Gets timestep or `dt` (ms)
    fn get_refractoriness_timestep(&self) -> f32;
    /// Returns neurotransmitter concentrations
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations;
    /// Returns refractoriness dynamics
    fn get_refractoriness_function(&self) -> &Self::U;
}

macro_rules! impl_current_voltage_spike_train {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> CurrentVoltage for $struct<T, U> {
            fn get_current_voltage(&self) -> f32 {
                self.current_voltage
            }
        }
    };
}

macro_rules! impl_is_spiking_spike_train {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> IsSpiking for $struct<T, U> {
            fn is_spiking(&self) -> bool {
                self.is_spiking
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
    pub current_voltage: f32,
    /// Maximum voltage (mV)
    pub v_th: f32,
    /// Minimum voltage (mV)
    pub v_resting: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last firing time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Chance of neuron firing at a given timestep
    pub chance_of_firing: f32,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

macro_rules! impl_default_spike_train_methods {
    () => {
        type U = U;

        fn get_height(&self) -> (f32, f32) {
            (self.v_th, self.v_resting)
        }
    
        fn get_refractoriness_timestep(&self) -> f32 {
            self.dt
        }
    
        fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    
        fn get_refractoriness_function(&self) -> &Self::U {
            &self.neural_refractoriness
        }
    }
}

impl_current_voltage_spike_train!(PoissonNeuron);
impl_is_spiking_spike_train!(PoissonNeuron);
impl_last_firing_time_spike_train!(PoissonNeuron);

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PoissonNeuron<T, U> {
    fn default() -> Self {
        PoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            neural_refractoriness: U::default(),
            chance_of_firing: 0.,
            dt: 0.1,
        }
    }
}

impl PoissonNeuron<ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f32, dt: f32) -> Self {
        PoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> PoissonNeuron<T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f32, dt: f32) -> Self {
        let mut poisson_neuron = PoissonNeuron::<T, U>::default();

        poisson_neuron.dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.dt) / hertz);

        poisson_neuron
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Timestep for PoissonNeuron<T, U> {
    fn get_dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        let scalar = dt / self.dt;
        self.chance_of_firing *= scalar;
        self.dt = dt;
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
        self.is_spiking = is_spiking;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        is_spiking
    }

    impl_default_spike_train_methods!();
}

/// A preset spike train that has a set of designated firing times and an internal clock,
/// the internal clock is updated every iteration by `dt` and once the internal clock reaches one of the 
/// firing times the neuron fires, the internal clock is reset and the clock will iterate until the
/// next firing time is reached, this will cycle until the last firing time is reached and the 
/// next firing time becomes the first firing time
#[derive(Debug, Clone, Timestep)]
pub struct PresetSpikeTrain<T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f32,
    /// Maximum voltage (mV)
    pub v_th: f32,
    /// Minimum voltage (mV)
    pub v_resting: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last spiking time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Set of times to fire at
    pub firing_times: Vec<f32>,
    /// Internal clock to track when to fire
    pub internal_clock: f32,
    /// Pointer to which firing time is next
    pub counter: usize,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

impl_current_voltage_spike_train!(PresetSpikeTrain);
impl_is_spiking_spike_train!(PresetSpikeTrain);
impl_last_firing_time_spike_train!(PresetSpikeTrain);

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PresetSpikeTrain<T, U> {
    fn default() -> Self {
        PresetSpikeTrain {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            neural_refractoriness: U::default(),
            firing_times: Vec::new(),
            internal_clock: 0.,
            counter: 0,
            dt: 0.1,
        }
    }
}

impl PresetSpikeTrain<ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PresetSpikeTrain::default()
    }
}

impl<T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PresetSpikeTrain<T, U> {
    fn iterate(&mut self) -> bool {
        self.internal_clock += self.dt;

        let is_spiking = if self.internal_clock > self.firing_times[self.counter] {
            self.current_voltage = self.v_th;

            self.internal_clock = 0.;

            self.counter += 1;
            if self.counter > self.firing_times.len() {
                self.counter = 0;
            }

            true
        } else {
            self.current_voltage = self.v_resting;

            false
        };
        self.is_spiking = is_spiking;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        is_spiking
    }

    impl_default_spike_train_methods!();
}
