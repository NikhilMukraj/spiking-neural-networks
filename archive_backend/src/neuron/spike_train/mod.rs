use std::collections::{HashMap, HashSet};
use rand::Rng;
use super::{
    CurrentVoltage, Potentiation, LastFiringTime, PotentiationType,
    Neurotransmitters, NeurotransmitterType, ApproximateNeurotransmitter
};


pub trait NeuralRefractoriness: Default {
    fn set_decay(&mut self, decay_factor: f64);
    fn get_decay(&self) -> f64;
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

#[derive(Debug, Clone, Copy)]
pub struct DeltaDiracRefractoriness {
    pub k: f64,
}

fn delta_dirac_effect(k: f64, a: f64, time_difference: f64, v_resting: f64, dt: f64) -> f64 {
    a * ((-1. / (k / dt)) * time_difference.powf(2.)).exp() + v_resting
}

impl_default_neural_refractoriness!(DeltaDiracRefractoriness, delta_dirac_effect);

#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayRefractoriness {
    pub k: f64
}

fn exponential_decay_effect(k: f64, a: f64, time_difference: f64, v_resting: f64, dt: f64) -> f64 {
    a * ((-1. / (k / dt)) * time_difference).exp() + v_resting
}

impl_default_neural_refractoriness!(ExponentialDecayRefractoriness, exponential_decay_effect);

pub trait SpikeTrain: CurrentVoltage + Potentiation + LastFiringTime {
    type T: NeuralRefractoriness;
    fn iterate(&mut self) -> bool;
    fn get_height(&self) -> (f64, f64);
    fn get_refractoriness_timestep(&self) -> f64;
    fn get_neurotransmitters(&self) -> &Neurotransmitters<ApproximateNeurotransmitter>;
    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64>;
    fn get_refractoriness_function(&self) -> &Self::T;
}

macro_rules! impl_current_voltage_spike_train {
    ($struct:ident) => {
        impl<T: NeuralRefractoriness> CurrentVoltage for $struct<T> {
            fn get_current_voltage(&self) -> f64 {
                self.current_voltage
            }
        }
    };
}

macro_rules! impl_potentiation_spike_train {
    ($struct:ident) => {
        impl<T: NeuralRefractoriness> Potentiation for $struct<T> {
            fn get_potentiation_type(&self) -> PotentiationType {
                self.potentiation_type
            }
        }
    };
}


macro_rules! impl_last_firing_time_spike_train {
    ($struct:ident) => {
        impl<T: NeuralRefractoriness> LastFiringTime for $struct<T> {
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.last_firing_time = timestep;
            }
        
            fn get_last_firing_time(&self) -> Option<usize> {
                self.last_firing_time
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct PoissonNeuron<T: NeuralRefractoriness> {
    pub current_voltage: f64,
    pub v_th: f64,
    pub v_resting: f64,
    pub last_firing_time: Option<usize>,
    pub synaptic_neurotransmitters: Neurotransmitters<ApproximateNeurotransmitter>,
    pub potentiation_type: PotentiationType,
    pub neural_refractoriness: T,
    pub chance_of_firing: f64,
    pub refractoriness_dt: f64,
}

macro_rules! impl_default_spike_train_methods {
    () => {
        type T = T;

        fn get_height(&self) -> (f64, f64) {
            (self.v_th, self.v_resting)
        }
    
        fn get_refractoriness_timestep(&self) -> f64 {
            self.refractoriness_dt
        }
    
        fn get_neurotransmitters(&self) -> &Neurotransmitters<ApproximateNeurotransmitter> {
            &self.synaptic_neurotransmitters
        }
    
        fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    
        fn get_refractoriness_function(&self) -> &T {
            &self.neural_refractoriness
        }
    }
}

impl_current_voltage_spike_train!(PoissonNeuron);
impl_potentiation_spike_train!(PoissonNeuron);
impl_last_firing_time_spike_train!(PoissonNeuron);

impl<T: NeuralRefractoriness> Default for PoissonNeuron<T> {
    fn default() -> Self {
        PoissonNeuron {
            current_voltage: -70.,
            v_th: 30.,
            v_resting: -70.,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::default(),
            potentiation_type: PotentiationType::Excitatory,
            neural_refractoriness: T::default(),
            chance_of_firing: 0.01,
            refractoriness_dt: 0.1,
        }
    }
}

impl<T: NeuralRefractoriness> PoissonNeuron<T> {
    // hertz is in seconds not ms
    pub fn from_firing_rate(hertz: f64, dt: f64) -> Self {
        let mut poisson_neuron = PoissonNeuron::<T>::default();

        poisson_neuron.refractoriness_dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.refractoriness_dt) / hertz);

        poisson_neuron
    }
}

impl<T: NeuralRefractoriness> SpikeTrain for PoissonNeuron<T> {
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

#[derive(Debug, Clone)]
struct PresetSpikeTrain<T: NeuralRefractoriness> {
    pub current_voltage: f64,
    pub v_th: f64,
    pub v_resting: f64,
    pub last_firing_time: Option<usize>,
    pub synaptic_neurotransmitters: Neurotransmitters<ApproximateNeurotransmitter>,
    pub potentiation_type: PotentiationType,
    pub neural_refractoriness: T,
    pub firing_times: HashSet<usize>,
    pub internal_clock: usize,
    pub max_clock_value: usize,
    pub refractoriness_dt: f64,
}

impl_current_voltage_spike_train!(PresetSpikeTrain);
impl_potentiation_spike_train!(PresetSpikeTrain);
impl_last_firing_time_spike_train!(PresetSpikeTrain);

impl<T: NeuralRefractoriness> Default for PresetSpikeTrain<T> {
    fn default() -> Self {
        PresetSpikeTrain {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<ApproximateNeurotransmitter>::default(),
            potentiation_type: PotentiationType::Excitatory,
            neural_refractoriness: T::default(),
            firing_times: HashSet::from([100, 300, 500]),
            internal_clock: 0,
            max_clock_value: 600,
            refractoriness_dt: 0.1,
        }
    }
}

// impl<T: NeuralRefractoriness> PresetSpikeTrain<T> {
//     fn from_evenly_divided(num_spikes: usize, dt: f64) -> Self {
//         let mut firing_times: HashSet<usize> =  HashSet::new();
//         let interval = ((1000. / dt) / (num_spikes as f64)) as usize;

//         let mut current_timestep = 0;
//         for _ in 0..num_spikes {
//             firing_times.insert(current_timestep);
//             current_timestep += interval;
//         }

//         let mut preset_spike_train = PresetSpikeTrain::<T>::default();
//         preset_spike_train.refractoriness_dt = dt;
//         preset_spike_train.firing_times = firing_times;
//         preset_spike_train.max_clock_value = current_timestep;

//         preset_spike_train
//     }
// }

impl<T: NeuralRefractoriness> SpikeTrain for PresetSpikeTrain<T> {
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
