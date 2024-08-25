//! A few implementations of different spike trains that can be coupled with `IterateAndSpike`
//! based neurons.

use rand::Rng;
use super::iterate_and_spike::{
    ApproximateNeurotransmitter, CurrentVoltage, IonotropicNeurotransmitterType, 
    IsSpiking, LastFiringTime, NeurotransmitterConcentrations, NeurotransmitterKinetics,
    NeurotransmitterType, Neurotransmitters, Timestep,
};
use super::iterate_and_spike_traits::{SpikeTrainBase, Timestep};
use super::plasticity::BCMActivity;

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
    type N: NeurotransmitterType;
    /// Updates spike train
    fn iterate(&mut self) -> bool;
    /// Gets maximum and minimum voltage values
    fn get_height(&self) -> (f32, f32);
    /// Returns neurotransmitter concentrations
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N>;
    /// Returns refractoriness dynamics
    fn get_refractoriness_function(&self) -> &Self::U;
}

/// A Poisson neuron
#[derive(Debug, Clone, SpikeTrainBase)]
pub struct PoissonNeuron<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
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
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
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
        type N = N;

        fn get_height(&self) -> (f32, f32) {
            (self.v_th, self.v_resting)
        }
    
        fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    
        fn get_refractoriness_function(&self) -> &Self::U {
            &self.neural_refractoriness
        }
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PoissonNeuron<N, T, U> {
    fn default() -> Self {
        PoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            chance_of_firing: 0.,
            dt: 0.1,
        }
    }
}

impl PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f32, dt: f32) -> Self {
        PoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> PoissonNeuron<N, T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f32, dt: f32) -> Self {
        let mut poisson_neuron = PoissonNeuron::<N, T, U>::default();

        poisson_neuron.dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.dt) / hertz);

        poisson_neuron
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Timestep for PoissonNeuron<N, T, U> {
    fn get_dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        let scalar = dt / self.dt;
        self.chance_of_firing *= scalar;
        self.dt = dt;
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PoissonNeuron<N, T, U> {
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
#[derive(Debug, Clone, SpikeTrainBase, Timestep)]
pub struct PresetSpikeTrain<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
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
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
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

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PresetSpikeTrain<N, T, U> {
    fn default() -> Self {
        PresetSpikeTrain {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            firing_times: Vec::new(),
            internal_clock: 0.,
            counter: 0,
            dt: 0.1,
        }
    }
}

impl PresetSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PresetSpikeTrain::default()
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PresetSpikeTrain<N, T, U> {
    fn iterate(&mut self) -> bool {
        self.internal_clock += self.dt;

        let is_spiking = if self.internal_clock > self.firing_times[self.counter] {
            self.current_voltage = self.v_th;

            self.internal_clock = 0.;

            self.counter += 1;
            if self.counter == self.firing_times.len() {
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

/// A BCM compatible Poisson neuron
#[derive(Debug, Clone, SpikeTrainBase)]
pub struct BCMPoissonNeuron<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
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
    /// Average activity
    pub average_activity: f32,
    /// Current activity
    pub current_activity: f32,
    /// Smoothing factor for updating activity
    pub bcm_smoothing_factor: f32,
    /// Current number of spikes in the firing window
    pub num_spikes: usize,
    /// Clock for firing rate calculation
    pub firing_rate_clock: f32,
    /// Current window for firing rate
    pub firing_rate_window: f32,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Chance of neuron firing at a given timestep
    pub chance_of_firing: f32,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for BCMPoissonNeuron<N, T, U> {
    fn default() -> Self {
        BCMPoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            average_activity: 0.,
            current_activity: 0.,
            bcm_smoothing_factor: 0.1,
            num_spikes: 0,
            firing_rate_clock: 0.,
            firing_rate_window: 100.,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            chance_of_firing: 0.,
            dt: 0.1,
        }
    }
}

impl BCMPoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        BCMPoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f32, dt: f32) -> Self {
        BCMPoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> BCMPoissonNeuron<N, T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f32, dt: f32) -> Self {
        let mut poisson_neuron = BCMPoissonNeuron::<N, T, U>::default();

        poisson_neuron.dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.dt) / hertz);

        poisson_neuron
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Timestep for BCMPoissonNeuron<N, T, U> {
    fn get_dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        let scalar = dt / self.dt;
        self.chance_of_firing *= scalar;
        self.dt = dt;
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for BCMPoissonNeuron<N, T, U> {
    // activity measured as current voltage - last voltage
    
    fn iterate(&mut self) -> bool {
        let is_spiking = if rand::thread_rng().gen_range(0.0..=1.0) <= self.chance_of_firing {
            self.current_activity = self.v_th - self.current_voltage;
            self.current_voltage = self.v_th;

            true
        } else {
            self.current_activity = self.v_resting - self.current_voltage;
            self.current_voltage = self.v_resting;

            false
        };

        if is_spiking {
            self.num_spikes += 1;
        }
        self.firing_rate_clock += self.dt;
        if self.firing_rate_clock >= self.firing_rate_window {
            self.firing_rate_clock = 0.;
            self.current_activity = self.num_spikes as f32 / self.firing_rate_window;
            self.average_activity += (self.bcm_smoothing_factor * (self.current_activity - self.average_activity)) * self.dt;
        }

        self.is_spiking = is_spiking;

        self.average_activity += (self.bcm_smoothing_factor * (self.current_activity - self.average_activity)) * self.dt;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        is_spiking
    }

    impl_default_spike_train_methods!();
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> BCMActivity for BCMPoissonNeuron<N, T, U> {
    fn get_activity(&self) -> f32 {
        self.current_activity
    }
    
    fn get_averaged_activity(&self) -> f32 {
        self.average_activity
    }
}
