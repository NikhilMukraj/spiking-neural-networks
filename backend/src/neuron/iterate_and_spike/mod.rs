//! The [`IterateAndSpike`] trait for encapsulating basic neuronal and spiking dynamics
//! as well as [`NeurotransmitterKinetics`] for neurotransmission and [`ReceptorKinetics`]
//! for receptor dynamics over time.

use std::{
    collections::{hash_map::{Keys, Values, ValuesMut}, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};
#[cfg(feature = "gpu")]
use opencl3::{
    kernel::Kernel, context::Context, command_queue::CommandQueue,
    memory::{Buffer, CL_MEM_READ_WRITE}, 
    types::{cl_float, cl_uint, cl_int, CL_BLOCKING, CL_NON_BLOCKING},
};
#[cfg(feature = "gpu")]
use std::{ptr, collections::BTreeSet};
#[cfg(feature = "gpu")]
use crate::error::GPUError;


/// Modifier for NMDA receptor current based on magnesium concentration and voltage
#[derive(Debug, Clone, Copy)]
pub struct BV {
    /// Calculates NMDA modifier based on voltage and magnesium concentration
    /// given a function to calculate the modfier
    pub bv_calc: fn(f32) -> f32,
}

fn default_bv_calc(voltage: f32) -> f32 {
    // 1.5 mM of Mg
    1. / (1. + ((-0.062 * voltage).exp() * 1.5 / 3.57)) 
}

impl Default for BV {
    fn default() -> Self {
        BV { 
            bv_calc: default_bv_calc
        }
    }
}

impl BV {
    /// Calculates effect of magnesium and voltage on NMDA receptor,
    /// voltage should be in mV
    fn calculate_b(&self, voltage: f32) -> f32 {
        (self.bv_calc)(voltage)
    }
}

/// Modifier for GABAb receptors
#[derive(Debug, Clone)]
pub struct GABAbDissociation {
    pub g: f32,
    pub n: f32,
    pub kd: f32,
    // k1: ,
    // k2: ,
    pub k3: f32,
    pub k4: f32,
}

impl Default for GABAbDissociation {
    fn default() -> Self {
        GABAbDissociation {
            g: 0.,
            n: 4.,
            kd: 100.,
            // k1: ,
            // k2: ,
            k3: 0.098,
            k4: 0.033, 
        }
    }
}

impl GABAbDissociation {
    /// Calculates effect of dissociation on GABAb receptor
    fn calculate_modifer(&self) -> f32 {
        self.g.powf(self.n) / (self.g.powf(self.n) * self.kd)
    }
}

/// Default for AMPA receptor
pub trait AMPADefault {
    fn ampa_default() -> Self;
}

/// Default for GABAa receptor
pub trait GABAaDefault {
    fn gabaa_default() -> Self;
}

/// Default for GABAb receptor
pub trait GABAbDefault {
    fn gabab_default() -> Self;
}

/// Secondary default for GABAb receptor
pub trait GABAbDefault2 {
    fn gabab_default2() -> Self;
}

/// Default for NMDA receptor
pub trait NMDADefault {
    fn nmda_default() -> Self;
}

/// Marker trait for neurotransmitter type
pub trait NeurotransmitterType: Hash + PartialEq + Eq + Clone + Copy + Debug + Send + Sync {}

/// Trait for GPU compatible neurotransmitter type
#[cfg(feature = "gpu")]
pub trait NeurotransmitterTypeGPU: NeurotransmitterType + PartialOrd + Ord {
    /// Converts the type to a numeric index (must be unique among types)
    fn type_to_numeric(&self) -> usize;
    /// Gets the number of availible types
    fn number_of_types() -> usize;
    /// Gets all neurotransmitter types availiable
    fn get_all_types() -> BTreeSet<Self>;
    /// Converts the type to a string
    fn to_string(&self) -> String;
}

/// Available neurotransmitter types for ionotropic receptor ligand gated channels
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub enum IonotropicNeurotransmitterType {
    /// Neurotransmitter type that effects only AMPA receptors
    AMPA,
    /// Neurotransmitter type that effects only NMDA receptors
    NMDA,
    /// Neurotransmitter type that effects only GABAa receptors
    GABAa,
    /// Neurotransmitter type that effects only GABAb receptors
    GABAb,
}

impl NeurotransmitterType for IonotropicNeurotransmitterType {}

#[cfg(feature = "gpu")]
impl NeurotransmitterTypeGPU for IonotropicNeurotransmitterType {
    fn type_to_numeric(&self) -> usize {
        match &self {
            IonotropicNeurotransmitterType::AMPA => 0,
            IonotropicNeurotransmitterType::NMDA => 1,
            IonotropicNeurotransmitterType::GABAa => 2,
            IonotropicNeurotransmitterType::GABAb => 3,
        }
    }

    fn number_of_types() -> usize {
        4
    }

    fn get_all_types() -> BTreeSet<Self> {
        BTreeSet::from([
            IonotropicNeurotransmitterType::AMPA,
            IonotropicNeurotransmitterType::NMDA,
            IonotropicNeurotransmitterType::GABAa,
            IonotropicNeurotransmitterType::GABAb,
        ])
    }

    fn to_string(&self) -> String {
        self.to_str().to_string()
    }
}

impl IonotropicNeurotransmitterType {
    /// Converts type to string
    pub fn to_str(&self) -> &str {
        match self {
            IonotropicNeurotransmitterType::AMPA => "AMPA",
            IonotropicNeurotransmitterType::GABAa => "GABAa",
            IonotropicNeurotransmitterType::GABAb => "GABAb",
            IonotropicNeurotransmitterType::NMDA => "NMDA",
        }
    }
}

/// Calculates neurotransmitter concentration over time based on voltage of neuron
pub trait NeurotransmitterKinetics: Clone + Send + Sync {
    /// Calculates change in neurotransmitter concentration based on voltage
    fn apply_t_change(&mut self, voltage: f32, dt: f32);
    /// Returns neurotransmitter concentration
    fn get_t(&self) -> f32;
    /// Manually sets neurotransmitter concentration
    fn set_t(&mut self, t: f32);
}

#[cfg(feature = "gpu")]
/// Neurotransmitter kinetics that are compatible with the GPU
pub trait NeurotransmitterKineticsGPU: NeurotransmitterKinetics + Default {
    /// Retrieves the given attribute
    fn get_attribute(&self, attribute: &str) -> Option<f32>;
    /// Sets the given value
    fn set_attribute(&mut self, attribute: &str, value: f32);
    /// Retrieves all attribute names
    fn get_attribute_names() -> HashSet<String>;
    /// Gets update function with the associated argument names
    fn get_update_function() -> (Vec<String>, String);
}

/// Neurotransmitter concentration based off of approximation 
/// found in this [paper](https://papers.cnl.salk.edu/PDFs/Kinetic%20Models%20of%20Synaptic%20Transmission%201998-3229.pdf)
#[derive(Debug, Clone, Copy)]
pub struct DestexheNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
    /// Half activated voltage threshold (mV)
    pub v_p: f32,
    /// Steepness (mV)
    pub k_p: f32,
}

macro_rules! impl_destexhe_neurotransmitter_default {
    ($trait:ident, $method:ident, $t_max:expr) => {
        impl $trait for DestexheNeurotransmitter {
            fn $method() -> Self {
                DestexheNeurotransmitter {
                    t_max: $t_max,
                    t: 0.,
                    v_p: 2., // 2 mV
                    k_p: 5., // 5 mV
                }
            }
        }
    };
}

impl_destexhe_neurotransmitter_default!(Default, default, 1.0);
impl_destexhe_neurotransmitter_default!(AMPADefault, ampa_default, 1.0);
impl_destexhe_neurotransmitter_default!(NMDADefault, nmda_default, 1.0);
impl_destexhe_neurotransmitter_default!(GABAaDefault, gabaa_default, 1.0);
impl_destexhe_neurotransmitter_default!(GABAbDefault, gabab_default, 0.5);

impl NeurotransmitterKinetics for DestexheNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f32, _: f32) {
        self.t = self.t_max / (1. + (-(voltage - self.v_p) / self.k_p).exp());
    }

    fn get_t(&self) -> f32 {
        self.t
    }

    fn set_t(&mut self, t: f32) {
        self.t = t;
    }
}

/// An approximation of neurotransmitter kinetics that sets the concentration to the 
/// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
/// slowly decreases the concentration over time by a factor of `dt` times `clearance_constant`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproximateNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
    /// Voltage threshold for detecting spikes (mV)
    pub v_th: f32,
    /// Amount to decrease neurotransmitter concentration by
    pub clearance_constant: f32,
}

macro_rules! impl_approximate_neurotransmitter_default {
    ($trait:ident, $method:ident, $t_max:expr) => {
        impl $trait for ApproximateNeurotransmitter {
            fn $method() -> Self {
                ApproximateNeurotransmitter {
                    t_max: $t_max,
                    t: 0.,
                    v_th: 25.,
                    clearance_constant: 0.01,
                }
            }
        }
    };
}

impl_approximate_neurotransmitter_default!(Default, default, 1.0);
impl_approximate_neurotransmitter_default!(AMPADefault, ampa_default, 1.0);
impl_approximate_neurotransmitter_default!(NMDADefault, nmda_default, 1.0);
impl_approximate_neurotransmitter_default!(GABAaDefault, gabaa_default, 1.0);
impl_approximate_neurotransmitter_default!(GABAbDefault, gabab_default, 0.5);

fn heaviside(x: f32) -> f32 {
    if x > 0. {
        1.
    } else {
        0.
    }
}

impl NeurotransmitterKinetics for ApproximateNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f32, dt: f32) {
        self.t += dt * -self.clearance_constant * self.t + (heaviside(voltage - self.v_th) * self.t_max);
        self.t = self.t_max.min(self.t.max(0.));
    }

    fn get_t(&self) -> f32 {
        self.t
    }

    fn set_t(&mut self, t: f32) {
        self.t = t;
    }
}

#[cfg(feature = "gpu")]
impl NeurotransmitterKineticsGPU for ApproximateNeurotransmitter {
    fn get_attribute(&self, value: &str) -> Option<f32> {
        match value {
            "t" => Some(self.t),
            "t_max" => Some(self.t_max),
            "v_th" => Some(self.v_th),
            "clearance_constant" => Some(self.clearance_constant),
            _ => None,
        }
    }

    fn set_attribute(&mut self, attribute: &str, value: f32) {
        match attribute {
            "t" => self.t = value,
            "t_max" => self.t_max = value,
            "v_th" => self.v_th = value,
            "clearance_constant" => self.clearance_constant = value,
            _ => unreachable!(),
        }
    }

    fn get_attribute_names() -> HashSet<String> {
        HashSet::from(
            [
                String::from("t"), String::from("t_max"), String::from("v_th"), 
                String::from("clearance_constant")
            ]
        )
    }

    fn get_update_function() -> (Vec<String>, String) {
        (
            vec![
                String::from("voltage"), String::from("dt"), String::from("t"),
                String::from("t_max"), String::from("v_th"), String::from("clearance_constant"),
            ],
            String::from("
                float get_t(
                    float voltage, 
                    float dt,
                    float t,
                    float t_max,
                    float v_th,
                    float clearance_constant,
                ) { 
                    float is_spiking_modifier = 0;
                    if (voltage > v_th) {
                        is_spiking_modifier = 1;
                    }
                    float new_t = dt * -clearance_constant * t + (is_spiking_modifier * t_max);

                    return clamp(t, 0, t_max);
                }
            ")
        )
    }
}

/// An approximation of neurotransmitter kinetics that sets the concentration to the 
/// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
/// then immediately sets it to 0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscreteSpikeNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
    /// Voltage threshold for detecting spikes (mV)
    pub v_th: f32,
}

impl NeurotransmitterKinetics for DiscreteSpikeNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f32, _: f32) {
        self.t = self.t_max * heaviside(voltage - self.v_th);
    }

    fn get_t(&self) -> f32 {
        self.t
    }

    fn set_t(&mut self, t: f32) {
        self.t = t;
    }
}

macro_rules! impl_discrete_neurotransmitter_default {
    ($trait:ident, $method:ident, $t_max:expr) => {
        impl $trait for DiscreteSpikeNeurotransmitter {
            fn $method() -> Self {
                DiscreteSpikeNeurotransmitter {
                    t_max: $t_max,
                    t: 0.,
                    v_th: 25.,
                }
            }
        }
    };
}

impl_discrete_neurotransmitter_default!(Default, default, 1.0);
impl_discrete_neurotransmitter_default!(AMPADefault, ampa_default, 1.0);
impl_discrete_neurotransmitter_default!(NMDADefault, nmda_default, 1.0);
impl_discrete_neurotransmitter_default!(GABAaDefault, gabaa_default, 1.0);
impl_discrete_neurotransmitter_default!(GABAbDefault, gabab_default, 0.5);

/// An approximation of neurotransmitter kinetics that sets the concentration to the 
/// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
/// slowly through exponential decay that scales based on the 
/// [`decay_constant`](Self::decay_constant) and [`dt`](Self::decay_constant)
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
    /// Voltage threshold for detecting spikes (mV)
    pub v_th: f32,
    /// Amount to decay neurotransmitter concentration by
    pub decay_constant: f32,
}

macro_rules! impl_exp_decay_neurotransmitter_default {
    ($trait:ident, $method:ident, $t_max:expr) => {
        impl $trait for ExponentialDecayNeurotransmitter {
            fn $method() -> Self {
                ExponentialDecayNeurotransmitter {
                    t_max: $t_max,
                    t: 0.,
                    v_th: 25.,
                    decay_constant: 2.0,
                }
            }
        }
    };
}

impl_exp_decay_neurotransmitter_default!(Default, default, 1.0);
impl_exp_decay_neurotransmitter_default!(AMPADefault, ampa_default, 1.0);
impl_exp_decay_neurotransmitter_default!(NMDADefault, nmda_default, 1.0);
impl_exp_decay_neurotransmitter_default!(GABAaDefault, gabaa_default, 1.0);
impl_exp_decay_neurotransmitter_default!(GABAbDefault, gabab_default, 0.5);

fn exp_decay(x: f32, l: f32, dt: f32) -> f32 {
    -x * (dt / -l).exp()
}

impl NeurotransmitterKinetics for ExponentialDecayNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f32, dt: f32) {
        let t_change = exp_decay(self.t, self.decay_constant, dt);
        self.t += t_change + (heaviside(voltage - self.v_th) * self.t_max);
        self.t = self.t_max.min(self.t.max(0.));
    }

    fn get_t(&self) -> f32 {
        self.t
    }

    fn set_t(&mut self, t: f32) {
        self.t = t;
    }
}

/// Calculates receptor gating values over time based on neurotransmitter concentration
pub trait ReceptorKinetics: Clone + Default + Sync + Send {
    /// Calculates the change in receptor gating based on neurotransmitter input
    fn apply_r_change(&mut self, t: f32, dt: f32);
    /// Gets the receptor gating value
    fn get_r(&self) -> f32;
    /// Sets the receptor gating value
    fn set_r(&mut self, r: f32);
}

#[cfg(feature = "gpu")]
/// Receptor kinetics that are compatible with the GPU
pub trait ReceptorKineticsGPU: ReceptorKinetics {
    /// Retrieves the given attribute
    fn get_attribute(&self, attribute: &str) -> Option<f32>;
    /// Sets the given value
    fn set_attribute(&mut self, attribute: &str, value: f32);
    /// Retrieves all attribute names
    fn get_attribute_names() -> HashSet<String>;
    /// Gets update function with the associated argument names
    fn get_update_function() -> (Vec<String>, String);
}

/// Receptor dynamics based off of model 
/// found in this [paper](https://papers.cnl.salk.edu/PDFs/Kinetic%20Models%20of%20Synaptic%20Transmission%201998-3229.pdf)
#[derive(Debug, Clone, Copy)]
pub struct DestexheReceptor {
    /// Receptor gating value
    pub r: f32,
    /// Forward rate constant (mM^-1 * ms^-1)
    pub alpha: f32,
    /// Backwards rate constant (ms^-1)
    pub beta: f32,
}

impl ReceptorKinetics for DestexheReceptor {
    fn apply_r_change(&mut self, t: f32, dt: f32) {
        self.r += (self.alpha * t * (1. - self.r) - self.beta * self.r) * dt;
    }

    fn get_r(&self) -> f32 {
        self.r
    }

    fn set_r(&mut self, r: f32) {
        self.r = r;
    }
}

macro_rules! impl_destexhe_receptor_default {
    ($trait:ident, $method:ident, $alpha:expr, $beta:expr) => {
        impl $trait for DestexheReceptor {
            fn $method() -> Self {
                DestexheReceptor {
                    r: 0.,
                    alpha: $alpha, // mM^-1 * ms^-1
                    beta: $beta, // ms^-1
                }
            }
        }
    };
}

impl_destexhe_receptor_default!(Default, default, 1., 1.);
impl_destexhe_receptor_default!(AMPADefault, ampa_default, 1.1, 0.19);
impl_destexhe_receptor_default!(GABAaDefault, gabaa_default, 5.0, 0.18);
impl_destexhe_receptor_default!(GABAbDefault, gabab_default, 0.016, 0.0047);
impl_destexhe_receptor_default!(GABAbDefault2, gabab_default2, 0.52, 0.0013);
impl_destexhe_receptor_default!(NMDADefault, nmda_default, 0.072, 0.0066);

/// Receptor dynamics approximation that just sets the receptor
/// gating value to the inputted neurotransmitter concentration
#[derive(Debug, Clone, Copy)]
pub struct ApproximateReceptor {
    pub r: f32,
}

impl ReceptorKinetics for ApproximateReceptor {
    fn apply_r_change(&mut self, t: f32, _: f32) {
        self.r = t;
    }

    fn get_r(&self) -> f32 {
        self.r
    }

    fn set_r(&mut self, r: f32) {
        self.r = r;
    }
}

#[cfg(feature = "gpu")]
impl ReceptorKineticsGPU for ApproximateReceptor {
    fn get_attribute(&self, value: &str) -> Option<f32> {
        match value {
            "r" => Some(self.r),
            _ => None,
        }
    }

    fn set_attribute(&mut self, attribute: &str, value: f32) {
        match attribute {
            "r" => self.r = value,
            _ => unreachable!(),
        }
    }

    fn get_attribute_names() -> HashSet<String> {
        HashSet::from([String::from("r")])
    }

    fn get_update_function() -> (Vec<String>, String) {
        (
            vec![
                String::from("t")
            ],
            String::from("
                float get_r(
                    float t,
                ) { 
                    return t;
                }
            ")
        )
    }
}

macro_rules! impl_approximate_receptor_default {
    ($trait:ident, $method:ident) => {
        impl $trait for ApproximateReceptor {
            fn $method() -> Self {
                ApproximateReceptor { r: 0. }
            }
        }
    };
}

impl_approximate_receptor_default!(Default, default);
impl_approximate_receptor_default!(AMPADefault, ampa_default);
impl_approximate_receptor_default!(GABAaDefault, gabaa_default);
impl_approximate_receptor_default!(GABAbDefault, gabab_default);
impl_approximate_receptor_default!(NMDADefault, nmda_default);

/// Receptor dynamics approximation that sets the receptor
/// gating value to the inputted neurotransmitter concentration and
/// then exponentially decays the receptor over time
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayReceptor {
    /// Maximal receptor gating value
    pub r_max: f32,
    /// Receptor gating value
    pub r: f32,
    /// Amount to decay neurotransmitter concentration by
    pub decay_constant: f32,
}

impl ReceptorKinetics for ExponentialDecayReceptor {
    fn apply_r_change(&mut self, t: f32, dt: f32) {
        self.r += exp_decay(self.r, self.decay_constant, dt) + t;
        self.r = self.r_max.min(self.r.max(0.));
    }

    fn get_r(&self) -> f32 {
        self.r
    }

    fn set_r(&mut self, r: f32) {
        self.r = r;
    }
}

macro_rules! impl_exp_decay_receptor_default {
    ($trait:ident, $method:ident) => {
        impl $trait for ExponentialDecayReceptor {
            fn $method() -> Self {
                ExponentialDecayReceptor { 
                    r_max: 1.0,
                    r: 0.,
                    decay_constant: 2.,
                }
            }
        }
    };
}

impl_exp_decay_receptor_default!(Default, default);
impl_exp_decay_receptor_default!(AMPADefault, ampa_default);
impl_exp_decay_receptor_default!(GABAaDefault, gabaa_default);
impl_exp_decay_receptor_default!(GABAbDefault, gabab_default);
impl_exp_decay_receptor_default!(NMDADefault, nmda_default);

/// Enum containing the type of ionotropic ligand gated receptor
/// containing a modifier to use when calculating current
#[derive(Debug, Clone)]
pub enum IonotropicLigandGatedReceptorType {
    /// AMPA receptor
    AMPA(f32),
    /// GABAa receptor
    GABAa(f32),
    /// GABAb receptor with dissociation modifier
    GABAb(GABAbDissociation),
    /// NMDA receptor with magnesium and voltage modifier
    NMDA(BV),
}

/// Singular ligand gated channel 
#[derive(Debug, Clone)]
pub struct LigandGatedChannel<T: ReceptorKinetics> {
    /// Maximal synaptic conductance (nS)
    pub g: f32,
    /// Reveral potential (mV)
    pub reversal: f32,
    // Receptor dynamics
    pub receptor: T,
    /// Type of receptor
    pub receptor_type: IonotropicLigandGatedReceptorType,
    /// Current generated by receptor
    pub current: f32,
}

impl<T: ReceptorKinetics + AMPADefault> AMPADefault for LigandGatedChannel<T> {
    fn ampa_default() -> Self {
        LigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            receptor: T::ampa_default(),
            receptor_type: IonotropicLigandGatedReceptorType::AMPA(1.0),
            current: 0.,
        }
    }
}

impl<T: ReceptorKinetics + GABAaDefault> GABAaDefault for LigandGatedChannel<T> {
    fn gabaa_default() -> Self {
        LigandGatedChannel {
            g: 1.2, // 1.2 nS
            reversal: -80., // -80 mV
            receptor: T::gabaa_default(),
            receptor_type: IonotropicLigandGatedReceptorType::GABAa(1.0),
            current: 0.,
        }
    }
}

impl<T: ReceptorKinetics + GABAbDefault> GABAbDefault for LigandGatedChannel<T> {
    fn gabab_default() -> Self {
        LigandGatedChannel {
            g: 0.06, // 0.06 nS
            reversal: -95., // -95 mV
            receptor: T::gabab_default(),
            receptor_type: IonotropicLigandGatedReceptorType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl<DestexheReceptor: ReceptorKinetics + GABAbDefault2> GABAbDefault2 for LigandGatedChannel<DestexheReceptor> {
    fn gabab_default2() -> Self {
        LigandGatedChannel {
            g: 0.06, // 0.06 nS
            reversal: -95., // -95 mV
            receptor: DestexheReceptor::gabab_default2(),
            receptor_type: IonotropicLigandGatedReceptorType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl<T: ReceptorKinetics + NMDADefault> NMDADefault for LigandGatedChannel<T> {
    fn nmda_default() -> Self {
        LigandGatedChannel {
            g: 0.6, // 0.6 nS
            reversal: 0., // 0.0 mV
            receptor: T::nmda_default(),
            receptor_type: IonotropicLigandGatedReceptorType::NMDA(BV::default()),
            current: 0.,
        }
    }
}

// /// Default implementation with a given `BV` as input
// pub trait NMDAWithBV {
//     fn nmda_with_bv(bv: BV) -> Self;
// }

// impl<T: ReceptorKinetics> NMDAWithBV for LigandGatedChannel<T> {
//     fn nmda_with_bv(bv: BV) -> Self {
//         LigandGatedChannel {
//             g: 0.6, // 0.6 nS
//             reversal: 0., // 0.0 mV
//             receptor: T::nmda_default(),
//             receptor_type: IonotropicLigandGatedReceptorType::NMDA(bv),
//             current: 0.,
//         }
//     }
// }

impl<T: ReceptorKinetics> LigandGatedChannel<T> {
    /// Calculates modifier for current calculation
    fn get_modifier(&mut self, voltage: f32, dt: f32) -> f32 {
        match &mut self.receptor_type {
            IonotropicLigandGatedReceptorType::AMPA(value) => *value,
            IonotropicLigandGatedReceptorType::GABAa(value) => *value,
            IonotropicLigandGatedReceptorType::GABAb(value) => {
                value.g += (value.k3 * self.receptor.get_r() - value.k4 * value.g) * dt;
                value.calculate_modifer() // G^N / (G^N + Kd)
            }, 
            IonotropicLigandGatedReceptorType::NMDA(value) => value.calculate_b(voltage),
        }
    }

    /// Calculates current generated from receptor based on input `voltage` in mV and `dt` in ms
    pub fn calculate_current(&mut self, voltage: f32, dt: f32) -> f32 {
        let modifier = self.get_modifier(voltage, dt);

        self.current = modifier * self.receptor.get_r() * self.g * (voltage - self.reversal);

        self.current
    }

    // /// Converts the receptor type of the channel to a string
    // pub fn to_str(&self) -> &str {
    //     match self.receptor_type {
    //         IonotropicLigandGatedReceptorType::Basic(_) => "Basic",
    //         IonotropicLigandGatedReceptorType::AMPA(_) => "AMPA",
    //         IonotropicLigandGatedReceptorType::GABAa(_) => "GABAa",
    //         IonotropicLigandGatedReceptorType::GABAb(_) => "GABAb",
    //         IonotropicLigandGatedReceptorType::NMDA(_) => "NMDA",
    //     }
    // }
}

/// Multiple igand gated channels with their associated neurotransmitter type
#[derive(Clone, Debug)]
pub struct LigandGatedChannels<T: ReceptorKinetics> { 
    pub ligand_gates: HashMap<IonotropicNeurotransmitterType, LigandGatedChannel<T>> 
}

impl<T: ReceptorKinetics> Default for LigandGatedChannels<T> {
    fn default() -> Self {
        LigandGatedChannels {
            ligand_gates: HashMap::new(),
        }
    }
}

impl<T: ReceptorKinetics> LigandGatedChannels<T> {
    /// Returns how many ligand gates there are
    pub fn len(&self) -> usize {
        self.ligand_gates.len()
    }

    /// Returns if ligand gates is empty
    pub fn is_empty(&self) -> bool {
        self.ligand_gates.is_empty()
    }

    /// Returns the neurotransmitter types as set of keys
    pub fn keys(&self) -> Keys<IonotropicNeurotransmitterType, LigandGatedChannel<T>> {
        self.ligand_gates.keys()
    }

    /// Returns the ligand gates as a set of values
    pub fn values(&self) -> Values<IonotropicNeurotransmitterType, LigandGatedChannel<T>> {
        self.ligand_gates.values()
    }

    /// Gets the ligand gate associated with the given [`NeurotransmitterType`]
    pub fn get(&self, neurotransmitter_type: &IonotropicNeurotransmitterType) -> Option<&LigandGatedChannel<T>> {
        self.ligand_gates.get(neurotransmitter_type)
    }

    /// Gets a mutable reference to the ligand gate associated with the given [`NeurotransmitterType`]
    pub fn get_mut(&mut self, neurotransmitter_type: &IonotropicNeurotransmitterType) -> Option<&mut LigandGatedChannel<T>> {
        self.ligand_gates.get_mut(neurotransmitter_type)
    }

    /// Inserts the given [`LigandGatedChannel`] with the associated [`NeurotransmitterType`]
    pub fn insert(
        &mut self, 
        neurotransmitter_type: IonotropicNeurotransmitterType, 
        ligand_gate: LigandGatedChannel<T>
    ) {
        self.ligand_gates.insert(neurotransmitter_type, ligand_gate);
    }

    /// Calculates the receptor currents for each channel based on a given voltage (mV)
    pub fn set_receptor_currents(&mut self, voltage: f32, dt: f32) {
        self.ligand_gates
            .values_mut()
            .for_each(|i| {
                i.calculate_current(voltage, dt);
        });
    }

    /// Returns the total sum of the currents given the timestep (ms) value 
    /// and capacitance of the model (nF)
    pub fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32 {
        self.ligand_gates
            .values()
            .map(|i| i.current)
            .sum::<f32>() * (dt / c_m)
    }

    /// Updates the receptor gating values based on the neurotransitter concentrations (mM)
    pub fn update_receptor_kinetics(&mut self, t_total: &NeurotransmitterConcentrations<IonotropicNeurotransmitterType>, dt: f32) {
        t_total.iter()
            .for_each(|(key, value)| {
                if let Some(gate) = self.ligand_gates.get_mut(key) {
                    gate.receptor.apply_r_change(*value, dt);
                }
            })
    }
}

// #[cfg(feature = "gpu")]
// impl <T: ReceptorKinetics> LigandGatedChannel<T> {
//     pub fn convert_to_gpu(
//         grid: &[Vec<Self>], context: &Context, queue: &CommandQueue, rows: usize, cols: usize,
//     ) -> Result<HashMap<String, BufferGPU>, GPUError> {
//         Ok(())
//     }

//     pub fn convert_to_cpu(
//         neurotransmitter_grid: &mut [Vec<Self>],
//         buffers: &HashMap<String, BufferGPU>,
//         queue: &CommandQueue,
//         rows: usize,
//         cols: usize,
//     ) -> Result<(), GPUError> {
//         Ok(())
//     }
// }

/// Multiple neurotransmitters with their associated types
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Neurotransmitters<N: NeurotransmitterType, T: NeurotransmitterKinetics> {
    pub neurotransmitters: HashMap<N, T>
}

/// A hashmap of neurotransmitter types and their associated concentration
pub type NeurotransmitterConcentrations<N> = HashMap<N, f32>;

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics> Default for Neurotransmitters<N, T> {
    fn default() -> Self {
        Neurotransmitters {
            neurotransmitters: HashMap::new(),
        }
    }
}

impl <N: NeurotransmitterType, T: NeurotransmitterKinetics> Neurotransmitters<N, T> {
    /// Returns how many neurotransmitters there are
    pub fn len(&self) -> usize {
        self.neurotransmitters.keys().len()
    }

    /// Returns if neurotransmitters is empty
    pub fn is_empty(&self) -> bool {
        self.neurotransmitters.is_empty()
    }

    /// Returns the neurotransmitter types as a set of keys
    pub fn keys(&self) -> Keys<N, T> {
        self.neurotransmitters.keys()
    }

    // Returns the neurotransmitter dynamics as a set of values
    pub fn values(&self) -> Values<N, T> {
        self.neurotransmitters.values()
    }

    /// Returns a set of mutable neurotransmitters 
    pub fn values_mut(&mut self) -> ValuesMut<N, T> {
        self.neurotransmitters.values_mut()
    }

    /// Gets the neurotransmitter associated with the given [`NeurotransmitterType`]
    pub fn get(&self, neurotransmitter_type: &N) -> Option<&T> {
        self.neurotransmitters.get(neurotransmitter_type)
    }

    /// Gets a mutable reference to the neurotransmitter associated with the given [`NeurotransmitterType`]
    pub fn get_mut(&mut self, neurotransmitter_type: &N) -> Option<&mut T> {
        self.neurotransmitters.get_mut(neurotransmitter_type)
    }

    /// Inserts the given neurotransmitter with the associated [`NeurotransmitterType`]
    pub fn insert(
        &mut self, 
        neurotransmitter_type: N, 
        neurotransmitter: T
    ) {
        self.neurotransmitters.insert(neurotransmitter_type, neurotransmitter);
    }

    /// Returns the neurotransmitter concentration (mM) with their associated types
    pub fn get_concentrations(&self) -> NeurotransmitterConcentrations<N> {
        self.neurotransmitters.iter()
            .map(|(neurotransmitter_type, neurotransmitter)| (*neurotransmitter_type, neurotransmitter.get_t()))
            .collect::<NeurotransmitterConcentrations<N>>()
    }

    /// Calculates the neurotransmitter concentrations based on the given voltage (mV)
    pub fn apply_t_changes(&mut self, voltage: f32, dt: f32) {
        self.neurotransmitters.values_mut()
            .for_each(|value| value.apply_t_change(voltage, dt));
    }
}


#[cfg(feature = "gpu")]
macro_rules! read_and_set_buffer {
    ($buffers:expr, $queue:expr, $buffer_name:expr, $vec:expr, Float) => {
        if let Some(BufferGPU::Float(buffer)) = $buffers.get($buffer_name) {
            let read_event = unsafe {
                match $queue.enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, $vec, &[]) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::BufferReadError),
                }
            };

            match read_event.wait() {
                Ok(value) => value,
                Err(_) => return Err(GPUError::WaitError),
            };
        }
    };
    
    ($buffers:expr, $queue:expr, $buffer_name:expr, $vec:expr, UInt) => {
        if let Some(BufferGPU::UInt(buffer)) = $buffers.get($buffer_name) {
            let read_event = unsafe {
                match $queue.enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, $vec, &[]) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::BufferReadError),
                }
            };

            match read_event.wait() {
                Ok(value) => value,
                Err(_) => return Err(GPUError::WaitError),
            };
        }
    };

    ($buffers:expr, $queue:expr, $buffer_name:expr, $vec:expr, OptionalUInt) => {
        if let Some(BufferGPU::OptionalUInt(buffer)) = $buffers.get($buffer_name) {
            let read_event = unsafe {
                match $queue.enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, $vec, &[]) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::BufferReadError),
                }
            };

            match read_event.wait() {
                Ok(value) => value,
                Err(_) => return Err(GPUError::WaitError),
            };
        }
    };
}

#[cfg(feature = "gpu")]
pub(crate) use read_and_set_buffer;

#[cfg(feature = "gpu")]
macro_rules! write_buffer {
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, Float) => {
        let mut $name = unsafe {
            match Buffer::<cl_float>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let _ = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };
    };
    
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, UInt) => {
        let mut $name = unsafe {
            match Buffer::<cl_uint>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let _ = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };
    };

    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, OptionalUInt) => {
        let mut $name = unsafe {
            match Buffer::<cl_int>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let _ = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };
    };

    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, Float, last) => {
        let mut $name = unsafe {
            match Buffer::<cl_float>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let last_event = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };

        match last_event.wait() {
            Ok(value) => value,
            Err(_) => return Err(GPUError::WaitError),
        };
    };
    
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, UInt, last) => {
        let mut $name = unsafe {
            match Buffer::<cl_uint>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let last_event = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };

        match last_event.wait() {
            Ok(value) => value,
            Err(_) => return Err(GPUError::WaitError),
        };
    };

    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, OptionalUInt, last) => {
        let mut $name = unsafe {
            match Buffer::<cl_int>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut()) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let last_event = unsafe { 
            match $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferWriteError),
            }
        };

        match last_event.wait() {
            Ok(value) => value,
            Err(_) => return Err(GPUError::WaitError),
        };
    };
}

#[cfg(feature = "gpu")]
pub(crate) use write_buffer;

#[cfg(feature = "gpu")]
macro_rules! flatten_and_retrieve_field {
    ($grid:expr, $field:ident, f32) => {
        $grid.iter()
            .flat_map(|inner| inner.iter())
            .map(|neuron| neuron.$field)
            .collect::<Vec<f32>>()
    };

    ($grid:expr, $field:ident, u32) => {
        $grid.iter()
            .flat_map(|inner| inner.iter())
            .map(|neuron| if neuron.$field { 1 } else { 0 })
            .collect::<Vec<u32>>()
    };

    ($grid:expr, $field:ident, OptionalUInt) => {
        $grid.iter()
            .flat_map(|inner| inner.iter())
            .map(|neuron| match neuron.$field { Some(value) => value as i32, None => -1 })
            .collect::<Vec<i32>>()
    };
}

#[cfg(feature = "gpu")]
pub(crate) use flatten_and_retrieve_field;

#[cfg(feature = "gpu")]
macro_rules! create_float_buffer {
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, f32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, Float);
    };

    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident, last) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, f32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, Float, last);
    };
}

#[cfg(feature = "gpu")]
pub(crate) use create_float_buffer;

#[cfg(feature = "gpu")]
macro_rules! create_uint_buffer {
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, u32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, UInt);
    };
    
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident, last) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, u32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, UInt, last);
    };
}

#[cfg(feature = "gpu")]
pub(crate) use create_uint_buffer;

#[cfg(feature = "gpu")]
macro_rules! create_optional_uint_buffer {
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, OptionalUInt);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, OptionalUInt);
    };
    
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident, last) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, OptionalUInt);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, OptionalUInt, last);
    };
}

#[cfg(feature = "gpu")]
pub(crate) use create_optional_uint_buffer;

fn extract_or_pad_neurotransmitter<N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU>(
    value: &Neurotransmitters<N, T>, 
    i: N, 
    buffers_contents: &mut HashMap<String, Vec<f32>>,
    flags: &mut HashMap<String, Vec<u32>>,
) {
    match value.get(&i) {
        Some(value) => {
            if let Some(current_flag) = flags.get_mut(&i.to_string()) {
                current_flag.push(1);
            }

            for attribute in T::get_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute) {
                    retrieved_attribute.push(
                        value.get_attribute(&attribute).expect("Attribute not found")
                    )
                } else {
                    unreachable!("Attribute not found");
                }
            }
        },
        None => {
            if let Some(current_flag) = flags.get_mut(&i.to_string()) {
                current_flag.push(0);
            }

            for attribute in T::get_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute) {
                    retrieved_attribute.push(0.)
                } else {
                    unreachable!("Attribute not found")
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl <N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU> Neurotransmitters<N, T> {
    pub fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue, rows: usize, cols: usize,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        let mut buffers_contents: HashMap<String, Vec<f32>> = HashMap::new();
        for i in T::get_attribute_names() {
            buffers_contents.insert(i.to_string(), vec![]);
        }

        let mut flags: HashMap<String, Vec<u32>> = HashMap::new();
        for i in N::get_all_types() {
            flags.insert(i.to_string().clone(), vec![]);
        }

        for row in grid.iter() {
            for value in row.iter() {
                for i in N::get_all_types() {
                    extract_or_pad_neurotransmitter(value, i, &mut buffers_contents, &mut flags);
                }
            }
        }

        let mut buffers: HashMap<String, BufferGPU> = HashMap::new();

        let size = rows * cols * N::number_of_types();

        for (key, value) in buffers_contents.iter() {
            write_buffer!(current_buffer, context, queue, size, value, Float, last);

            buffers.insert(key.clone(), BufferGPU::Float(current_buffer));
        }

        let size = rows * cols;

        for (key, value) in flags.iter() {
            write_buffer!(current_buffer, context, queue, size, value, UInt, last);

            buffers.insert(key.clone(), BufferGPU::UInt(current_buffer));
        }

        Ok(buffers)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn convert_to_cpu(
        neurotransmitter_grid: &mut [Vec<Self>],
        buffers: &HashMap<String, BufferGPU>,
        queue: &CommandQueue,
        rows: usize,
        cols: usize,
    ) -> Result<(), GPUError> {
        let mut cpu_conversion: HashMap<String, Vec<f32>> = HashMap::new();
        let mut flags: HashMap<String, Vec<bool>> = HashMap::new();

        let string_types: Vec<String> = N::get_all_types()
            .into_iter()
            .map(|i| i.to_string())
            .collect();

        for key in buffers.keys() {
            if !string_types.contains(key) {
                let mut current_contents = vec![0.; rows * cols * N::number_of_types()];
                read_and_set_buffer!(buffers, queue, key, &mut current_contents, Float);

                cpu_conversion.insert(key.clone(), current_contents);
            } else {
                let mut current_contents = vec![0; rows * cols];
                read_and_set_buffer!(buffers, queue, key, &mut current_contents, UInt);

                flags.insert(
                    key.clone(), 
                    current_contents.iter().map(|i| *i == 1).collect::<Vec<bool>>() // uint to bool
                );
            }
        }

        for row in 0..rows {
            for col in 0..cols {
                let grid_value = &mut neurotransmitter_grid[row][col];
                let flag_index = row * cols + col;
                for i in N::get_all_types() {
                    let i_str = i.to_string();
                    let index = row * cols * N::number_of_types() 
                        + col * N::number_of_types() + i.type_to_numeric();
    
                    if let Some(flag) = flags.get(&i_str) {
                        if flag[flag_index] {
                            if !grid_value.neurotransmitters.contains_key(&i) {
                                grid_value.insert(i, T::default());
                            }
    
                            for attribute in T::get_attribute_names() {
                                println!("{}: {}, {}", flag_index, index, attribute);
                                if let Some(values) = cpu_conversion.get(&attribute) {
                                    let attr_value = values[index];
                                    grid_value.neurotransmitters
                                        .get_mut(&i)
                                        .unwrap()
                                        .set_attribute(&attribute, attr_value);
                                }
                            }
                        } else {
                            grid_value.neurotransmitters.remove(&i);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Multiplies multiple neurotransmitters concentrations by a single scalar value
pub fn weight_neurotransmitter_concentration<N: NeurotransmitterType>(
    neurotransmitter_hashmap: &mut NeurotransmitterConcentrations<N>, 
    weight: f32
) {
    neurotransmitter_hashmap.values_mut().for_each(|value| *value *= weight);
}

/// Sums the neurotransmitter concentrations together, and averages each neurotransmitter
/// concentration individually
pub fn aggregate_neurotransmitter_concentrations<N: NeurotransmitterType>(
    neurotransmitter_hashmaps: &Vec<NeurotransmitterConcentrations<N>>
) -> NeurotransmitterConcentrations<N> {
    let mut cumulative_map: NeurotransmitterConcentrations<N> = HashMap::new();
    let mut scalar_map: HashMap<N, usize> = HashMap::new();

    for map in neurotransmitter_hashmaps {
        for (key, value) in map {
            *cumulative_map.entry(*key).or_insert(0.0) += value;
            *scalar_map.entry(*key).or_insert(0) += 1;
        }
    }

    for (key, value) in scalar_map {
        if let Some(neurotransmitter) = cumulative_map.get_mut(&key) {
            *neurotransmitter /= value as f32;
        }
    }

    cumulative_map
}

// NMDA
// alpha: 7.2 * 10^4 M^-1 * sec^-1, beta: 6.6 sec^-1

// AMPA
// alpha: 1.1 * 10^6 M^-1 * sec^-1, beta: 190 sec^-1

// GABAa
// alpha: 5 * 10^6 M^-1 * sec^-1, beta: 180 sec^-1, reversal: 80 mv

// I NMDA = Gsyn(t) * (Vm - Esyn)
// Gsyn(t) = G NMDA * gamma / (1 + mg_conc * (-alpha * Vm).exp() / beta) * ((-t / tau2).exp() - (-t / tau1).exp()) * H(t)
// Gsyn is basically just B(V)
// gamma = 1 / ((-tpk / tau2).exp() - (-tpk / tau1).exp())
// tpk = (tau1 * tau2) / (tau2 - tau1) * (tau2 / tau1).ln()
// H(t) = heaviside
// t is time

// percent of open receptors fraction is r, T is neurotrasnmitter concentration
// could vary Tmax to vary amount of nt conc
// dr/dt = alpha * T * (1 - r) - beta * r
// T = Tmax / (1 + (-(Vpre - Vp) / Kp).exp())

// I AMPA (or GABAa) = G AMPA (or GABAa) * (Vm - E AMPA (or GABAa))
// can also be modified with r

/// A set of parameters to use in generating gaussian noise
#[derive(Debug, Clone)]
pub struct GaussianParameters {
    /// Mean of distribution
    pub mean: f32,
    /// Standard deviation of distribution
    pub std: f32,
    /// Maximum cutoff value
    pub max: f32,
    /// Minimum cutoff value
    pub min: f32,
}

impl Default for GaussianParameters {
    fn default() -> Self {
        GaussianParameters { 
            mean: 1.0, // center of norm distr
            std: 0.0, // std of norm distr
            max: 2.0, // maximum cutoff for norm distr
            min: 0.0, // minimum cutoff for norm distr
        }
    }
}

impl GaussianParameters {
    /// Generates a normally distributed random number clamped between
    /// a minimum and a maximum
    pub fn get_random_number(&self) -> f32 {
        crate::distribution::limited_distr(
            self.mean, 
            self.std, 
            self.min, 
            self.max,
        )
    }
}

/// Gets current voltage (mV) of model
pub trait CurrentVoltage {
    fn get_current_voltage(&self) -> f32;
}

/// Gets conductance of the synapse of a given neuron
pub trait GapConductance {
    fn get_gap_conductance(&self) -> f32;
}

/// Gets whether the neuron is spiking
pub trait IsSpiking {
    fn is_spiking(&self) -> bool;
}

/// Handles the firing times of the neuron
pub trait LastFiringTime {
    /// Gets the last firing time of the neuron, (`None` if the neuron has not fired yet)
    fn get_last_firing_time(&self) -> Option<usize>;
    /// Sets the last firing time of the neuron, (use `None` to reset)
    fn set_last_firing_time(&mut self, timestep: Option<usize>);
}

/// Handles changes in simulation timestep information
pub trait Timestep {
    /// Retrieves timestep value
    fn get_dt(&self) -> f32;
    /// Updates instance with new timestep information
    fn set_dt(&mut self, dt: f32);
}

/// Handles dynamics neurons that can take in an input to update membrane potential
/// 
/// Example implementation:
/// 
/// ```rust
/// use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;
/// use spiking_neural_networks::neuron::iterate_and_spike::{
///     IsSpiking, Timestep, CurrentVoltage, GapConductance, IterateAndSpike, 
///     LastFiringTime, NeurotransmitterConcentrations, LigandGatedChannels, 
///     ReceptorKinetics, NeurotransmitterKinetics, Neurotransmitters,
///     ApproximateNeurotransmitter, ApproximateReceptor,
///     IonotropicNeurotransmitterType,
/// };
/// 
/// 
/// #[derive(Debug, Clone, IterateAndSpikeBase)]
/// pub struct QuadraticIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
///     /// Membrane potential (mV)
///     pub current_voltage: f32, 
///     /// Voltage threshold (mV)
///     pub v_th: f32, 
///     /// Voltage reset value/resting membrane potential (mV)
///     pub v_reset: f32, 
///     /// Voltage initialization value (mV)
///     pub v_init: f32, 
///     /// Counter for refractory period
///     pub refractory_count: f32, 
///     /// Total refractory period (ms)
///     pub tref: f32, 
///     /// Steepness of slope
///     pub alpha: f32, 
///     /// Critical voltage for spike initiation (mV)
///     pub v_c: f32,
///     /// Input value modifier
///     pub integration_constant: f32, 
///     /// Controls conductance of input gap junctions
///     pub gap_conductance: f32, 
///     /// Membrane time constant (ms)
///     pub tau_m: f32, 
///     /// Membrane capacitance (nF)
///     pub c_m: f32, 
///     /// Time step (ms)
///     pub dt: f32, 
///     /// Whether the neuron is spiking
///     pub is_spiking: bool,
///     /// Last timestep the neuron has spiked
///     pub last_firing_time: Option<usize>,
///     /// Postsynaptic neurotransmitters in cleft
///     pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
///     /// Ionotropic receptor ligand gated channels
///     pub ligand_gates: LigandGatedChannels<R>,
/// }
/// 
/// impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> QuadraticIntegrateAndFireNeuron<T, R> {
///     /// Calculates the change in voltage given an input current
///     pub fn quadratic_get_dv_change(&self, i: f32) -> f32 {
///         ((self.alpha * (self.current_voltage - self.v_reset) * (self.current_voltage - self.v_c)) + 
///         self.integration_constant * i) * (self.dt / self.tau_m)
///     }
/// 
///     /// Determines whether the neuron is spiking and resets the voltage
///     /// if so, also handles refractory period
///     pub fn handle_spiking(&mut self) -> bool {
///         let mut is_spiking = false;
/// 
///         if self.refractory_count > 0. {
///             self.current_voltage = self.v_reset;
///             self.refractory_count -= 1.;
///         } else if self.current_voltage >= self.v_th {
///             is_spiking = !is_spiking;
///             self.current_voltage = self.v_reset;
///             self.refractory_count = self.tref / self.dt
///         }
/// 
///         self.is_spiking = is_spiking;
/// 
///         is_spiking
///     }
/// }
/// 
/// impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for QuadraticIntegrateAndFireNeuron<T, R> {
///     type N = IonotropicNeurotransmitterType;
/// 
///     fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
///         self.synaptic_neurotransmitters.get_concentrations()
///     }
/// 
///     fn iterate_and_spike(&mut self, input_current: f32) -> bool {
///         let dv = self.quadratic_get_dv_change(input_current);
///         self.current_voltage += dv; // updates voltage
/// 
///         // calculates neurotransmitter concentration
///         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);
/// 
///         self.handle_spiking()
///     }
/// 
///     fn iterate_with_neurotransmitter_and_spike(
///         &mut self, 
///         input_current: f32, 
///         t_total: &NeurotransmitterConcentrations<Self::N>,
///     ) -> bool {
///         // accounts for receptor currents
///         self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
///         self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);
/// 
///         let dv = self.quadratic_get_dv_change(input_current);
///         let neurotransmitter_dv = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
/// 
///         self.current_voltage += dv + neurotransmitter_dv; // applies receptor currents and change in voltage
/// 
///         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);
/// 
///         self.handle_spiking()
///     }
/// } 
/// ```
pub trait IterateAndSpike: 
    CurrentVoltage + Timestep + GapConductance + IsSpiking + LastFiringTime + Clone + Send + Sync 
{
    /// Type of neurotransmitter to use
    type N: NeurotransmitterType;
    /// Takes in an input current and returns whether the model is spiking
    /// after the membrane potential is updated
    fn iterate_and_spike(&mut self, input_current: f32) -> bool;
    /// Gets the neurotransmitter concentrations of the neuron (mM)
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N>;
    /// Takes in an input current and neurotransmitter input and returns whether the model
    /// is spiking after the membrane potential is updated, neurotransmitter input updates
    /// receptor currents based on the associated neurotransmitter concentration,
    /// the current from the receptors is also factored into the change in membrane potential
    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool;
}

#[cfg(feature = "gpu")]
/// An encapsulation of necessary data for GPU kernels
pub struct KernelFunction {
    pub kernel: Kernel,
    pub program_source: String,
    pub kernel_name: String,
    pub argument_names: Vec<String>,
}

#[cfg(feature = "gpu")]
/// An encapsulation of a float or unsigned integer buffer for the GPU
pub enum BufferGPU {
    Float(Buffer<cl_float>),
    UInt(Buffer<cl_uint>),
    OptionalUInt(Buffer<cl_int>),
}

// set args on the fly using a for loop
// for n in x { kernel.set_arg(&n); } // modify this to use names instead
// could use lazy static for kernel compilation
// edit last firing time every kernel execution, -1 is considered none
// create chemical kernels after electrical ones
// should have seperate convert to electrical and convert to chemical
// that way ligand gates arent generated when not in use
// conversions should be falliable
#[cfg(feature = "gpu")]
pub trait IterateAndSpikeGPU: IterateAndSpike {
    /// Returns the compiled kernel for electrical inputs
    fn iterate_and_spike_electrical_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
    // /// Returns the compiled kernel for chemical inputs
    // fn iterate_and_spike_chemical_kernel(&self) -> KernelFunction;
    // /// Returns the compiled kernel for electirlca and chemical inputs
    // fn iterate_and_spike_electrochemical_kernel(&self) -> KernelFunction;
    /// Converts a grid of the neuron type to a vector of buffers
    fn convert_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts buffers back to a grid of neurons
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}
