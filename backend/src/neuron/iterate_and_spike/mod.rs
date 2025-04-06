//! The [`IterateAndSpike`] trait for encapsulating basic neuronal and spiking dynamics
//! as well as [`NeurotransmitterKinetics`] for neurotransmission and [`ReceptorKinetics`]
//! for receptor dynamics over time.

use std::{
    collections::{hash_map::{Entry, Keys, Values, ValuesMut}, HashMap, HashSet},
    fmt::Debug,
    hash::Hash, io::Error,
};
use crate::error::ReceptorNeurotransmitterError;
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BV {
    /// Concentration of extracellular magnesium (mM)
    pub mg: f32,
}

impl Default for BV {
    fn default() -> Self {
        BV { 
            mg: 0.33 // mM
        }
    }
}

impl BV {
    /// Calculates effect of magnesium and voltage on NMDA receptor,
    /// voltage should be in mV
    fn calculate_b(&self, voltage: f32) -> f32 {
        1. / (1. + ((-0.062 * voltage).exp() * self.mg / 3.57))
    }
}

/// Modifier for GABAb receptors
#[derive(Debug, Clone, PartialEq)]
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
    fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T);
    /// Returns neurotransmitter concentration
    fn get_t(&self) -> f32;
    /// Manually sets neurotransmitter concentration
    fn set_t(&mut self, t: f32);
}

#[cfg(feature = "gpu")]
/// Neurotransmitter kinetics that are compatible with the GPU
pub trait NeurotransmitterKineticsGPU: NeurotransmitterKinetics + Default {
    /// Retrieves the given attribute
    fn get_attribute(&self, attribute: &str) -> Option<BufferType>;
    /// Sets the given value
    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), Error>;
    /// Retrieves all attribute names
    fn get_attribute_names() -> HashSet<(String, AvailableBufferType)>;
    /// Retrieves all attribute names as a vector
    fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)>;
    /// Retrieves all attribute names in an ordered fashion
    fn get_attribute_names_ordered() -> BTreeSet<(String, AvailableBufferType)>;
    /// Gets update function with the associated argument names
    fn get_update_function() -> ((Vec<String>, Vec<String>), String);
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
    fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T) {
        self.t = self.t_max / (1. + (-(neuron.get_current_voltage() - self.v_p) / self.k_p).exp());
    }

    fn get_t(&self) -> f32 {
        self.t
    }

    fn set_t(&mut self, t: f32) {
        self.t = t;
    }
}

/// An approximation of neurotransmitter kinetics that sets the concentration to the 
/// maximal value when a spike is detected and
/// slowly decreases the concentration over time by a factor of `dt` times `clearance_constant`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproximateNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
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


fn bool_to_float(flag: bool) -> f32 {
    if flag {
        1.
    } else {
        0.
    }
}

impl NeurotransmitterKinetics for ApproximateNeurotransmitter {
    fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T) {
        self.t += neuron.get_dt() * -self.clearance_constant * self.t + (bool_to_float(neuron.is_spiking()) * self.t_max);
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
    fn get_attribute(&self, value: &str) -> Option<BufferType> {
        match value {
            "neurotransmitters$t" => Some(BufferType::Float(self.t)),
            "neurotransmitters$t_max" => Some(BufferType::Float(self.t_max)),
            "neurotransmitters$clearance_constant" => Some(BufferType::Float(self.clearance_constant)),
            _ => None,
        }
    }

    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), Error> {
        match attribute {
            "neurotransmitters$t" => self.t = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
            },
            "neurotransmitters$t_max" => self.t_max = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
            },
            "neurotransmitters$clearance_constant" => self.clearance_constant = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
            },
            _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute")),
        }

        Ok(())
    }

    fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)> {
        vec![
            (String::from("neurotransmitters$t"), AvailableBufferType::Float), 
            (String::from("neurotransmitters$t_max"), AvailableBufferType::Float), 
            (String::from("neurotransmitters$clearance_constant"), AvailableBufferType::Float),
        ]
    }

    fn get_attribute_names() -> HashSet<(String, AvailableBufferType)> {
        HashSet::from_iter(
            Self::get_attribute_names_as_vector()
        )
    }

    fn get_attribute_names_ordered() -> BTreeSet<(String, AvailableBufferType)> {
        Self::get_attribute_names_as_vector().into_iter().collect()
    }

    fn get_update_function() -> ((Vec<String>, Vec<String>), String) {
        (
            (
                vec![
                    String::from("current_voltage"), String::from("is_spiking"), String::from("dt"), 
                ],
                vec![
                    String::from("neurotransmitters$t"),
                    String::from("neurotransmitters$t_max"), String::from("neurotransmitters$clearance_constant"),
                ]
            ),
            String::from("
                float get_t(
                    float current_voltage,
                    float is_spiking, 
                    float dt,
                    float neurotransmitters_t,
                    float neurotransmitters_t_max,
                    float neurotransmitters_clearance_constant
                ) { 
                    float is_spiking_modifier = 0.0f;
                    if (is_spiking) {
                        is_spiking_modifier = 1.0f;
                    }
                    float new_t = neurotransmitters_t + dt * -neurotransmitters_clearance_constant * neurotransmitters_t + 
                        (is_spiking_modifier * neurotransmitters_t_max);

                    return clamp(new_t, 0.0f, neurotransmitters_t_max);
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
}

impl NeurotransmitterKinetics for DiscreteSpikeNeurotransmitter {
    fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T) {
        self.t = self.t_max * bool_to_float(neuron.is_spiking());
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
/// maximal value when a spike is detected and
/// slowly through exponential decay that scales based on the 
/// [`decay_constant`](Self::decay_constant) and [`dt`](Self::decay_constant)
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayNeurotransmitter {
    /// Maximal neurotransmitter concentration (mM)
    pub t_max: f32,
    /// Current neurotransmitter concentration (mM)
    pub t: f32,
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
    fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T) {
        let t_change = exp_decay(self.t, self.decay_constant, neuron.get_dt());
        self.t += t_change + (bool_to_float(neuron.is_spiking()) * self.t_max);
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
    fn get_attribute(&self, attribute: &str) -> Option<BufferType>;
    /// Sets the given value
    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), Error>;
    /// Retrieves all attribute names
    fn get_attribute_names() -> HashSet<(String, AvailableBufferType)>;
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn get_attribute(&self, value: &str) -> Option<BufferType> {
        match value {
            "receptor$kinetics$r" => Some(BufferType::Float(self.r)),
            _ => None,
        }
    }

    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), Error> {
        match attribute {
            "receptors$kinetics$r" => self.r = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
            },
            _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute")),
        }

        Ok(())
    }

    fn get_attribute_names() -> HashSet<(String, AvailableBufferType)> {
        HashSet::from([(String::from("receptors$kinetics$r"), AvailableBufferType::Float)])
    }

    fn get_update_function() -> (Vec<String>, String) {
        (
            vec![
                (String::from("neurotransmitters$t")),
                (String::from("dt")),
                (String::from("receptors$kinetics$r")),
            ],
            String::from("
                float get_r(
                    float t, float dt, float r
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

#[derive(Hash, Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum DefaultReceptorsNeurotransmitterType {
    X,
}

impl DefaultReceptorsNeurotransmitterType {
    fn get_associated_receptor<T: ReceptorKinetics>(&self) -> DefaultReceptorsType<T> {
        match &self {
            DefaultReceptorsNeurotransmitterType::X => {
                DefaultReceptorsType::X(XReceptor::<T>::default())
            }
        }
    }
}

impl NeurotransmitterType for DefaultReceptorsNeurotransmitterType {}

#[cfg(feature = "gpu")]
impl NeurotransmitterTypeGPU for DefaultReceptorsNeurotransmitterType {
    fn type_to_numeric(&self) -> usize {
        match &self {
            DefaultReceptorsNeurotransmitterType::X => 0,
        }
    }

    fn number_of_types() -> usize {
        1
    }

    fn get_all_types() -> BTreeSet<Self> {
        BTreeSet::from([
            DefaultReceptorsNeurotransmitterType::X,
        ])
    }

    fn to_string(&self) -> String {
        String::from("X")
    }
}

#[derive(Debug, Clone)]
pub struct XReceptor<T: ReceptorKinetics> {
    pub current: f32,
    pub g: f32,
    pub e: f32,
    pub r: T,
}

impl<T: ReceptorKinetics> Default for XReceptor<T> {
    fn default() -> Self {
        XReceptor {
            current: 0.,
            g: 1.,
            e: 0.,
            r: T::default(),
        }
    }
}

impl<T: ReceptorKinetics> XReceptor<T> {
    fn apply_r_change(&mut self, t: f32, dt: f32) {
        self.r.apply_r_change(t, dt);
    }
    fn iterate(&mut self, current_voltage: f32, _dt: f32) {
        self.current = self.g * self.r.get_r() * (current_voltage - self.e);
    }
}

impl<T: ReceptorKineticsGPU> ReceptorsGPU for DefaultReceptors<T> {
    fn get_attribute(&self, attribute: &str) -> Option<BufferType> {
        match attribute {
            "receptors$X_current" => match &self.receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
                Some(DefaultReceptorsType::X(val)) => Some(BufferType::Float(val.current)),
                _ => None
            },
            "receptors$X_g" => match &self.receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
                Some(DefaultReceptorsType::X(val)) => Some(BufferType::Float(val.g)),
                _ => None
            },
            "receptors$X_e" => match &self.receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
                Some(DefaultReceptorsType::X(val)) => Some(BufferType::Float(val.e)),
                _ => None
            },
            _ => {
                let split = attribute.split("$").collect::<Vec<&str>>();
                if split.len() != 5 { return None; }
                let (receptor, neuro, name, kinetics, attr) = (split[0], split[1], split[2], split[3], split[4]);  

                if *receptor != *"receptors".to_string() || *kinetics != *"kinetics".to_string() { return None; }  
                let stripped = format!("receptors$kinetics${}", attr);
                let to_match = format!("{}${}", neuro, name);

                match to_match.as_str() {
                    "X$r" => match self.receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
                        Some(DefaultReceptorsType::X(inner)) => inner.r.get_attribute(&stripped),
                        _ => None,
                    },
                    _ => None
                } 
            }
        }
    }

    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error> {    
        match attribute {
                "receptors$X_current" => match (self.receptors.get_mut(&DefaultReceptorsNeurotransmitterType::X), value) {     
                    (Some(DefaultReceptorsType::X(receptors)), BufferType::Float(nested_val)) => receptors.current = nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),    
                },
                "receptors$X_g" => match (self.receptors.get_mut(&DefaultReceptorsNeurotransmitterType::X), value) {
                    (Some(DefaultReceptorsType::X(receptors)), BufferType::Float(nested_val)) => receptors.g = nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),    
                    },
                "receptors$X_e" => match (self.receptors.get_mut(&DefaultReceptorsNeurotransmitterType::X), value) {
                    (Some(DefaultReceptorsType::X(receptors)), BufferType::Float(nested_val)) => receptors.e = nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),    
                },
            _ => {
                let split = attribute.split("$").collect::<Vec<&str>>();
                if split.len() != 5 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute")); }
                let (receptor, neuro, name, kinetics, attr) = (split[0], split[1], split[2], split[3], split[4]);  
                if *receptor != *"receptors".to_string() || *kinetics != *"kinetics".to_string() { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute")); }
                let stripped = format!("receptors$kinetics${}", attr);
                let to_match = format!("{}${}", neuro, name);

                match to_match.as_str() {
                "X$r" => match self.receptors.get_mut(&DefaultReceptorsNeurotransmitterType::X) {
                    Some(DefaultReceptorsType::X(inner)) => inner.r.set_attribute(&stripped, value)?,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute")),
                },
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid attribute"))} 
            }
        };

        Ok(())
    }

    fn get_all_attributes() -> HashSet<(String, AvailableBufferType)> {
        let mut attrs = HashSet::from([(String::from("receptors$X_current"), AvailableBufferType::Float), (String::from("receptors$X_g"), AvailableBufferType::Float), (String::from("receptors$X_e"), AvailableBufferType::Float)]); 
        attrs.extend(
            T::get_attribute_names().iter().map(|(i, j)|
                (
                    format!(
                        "receptors$X$r$kinetics${}",
                        i.split("$").collect::<Vec<_>>().last()
                            .expect("Invalid attribute")
                    ),
                    *j
                )
            ).collect::<Vec<(_, _)>>()
        );

        attrs
    }

    fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if grid.is_empty() || grid.iter().all(|i| i.is_empty()) {
            return Ok(HashMap::new());
        }

        let mut buffers = HashMap::new();

        let size: usize = grid.iter().map(|row| row.len()).sum();

        for (attr, current_type) in Self::get_all_attributes() {
            match current_type {
                AvailableBufferType::Float => {
                    let mut current_attrs: Vec<f32> = vec![];
                    for row in grid.iter() {
                        for i in row.iter() {
                            match i.get_attribute(&attr) {
                                Some(BufferType::Float(val)) => current_attrs.push(val),
                                Some(_) => unreachable!(),
                                None => current_attrs.push(0.),
                            };
                        }
                    }

                    write_buffer!(current_buffer, context, queue, size, &current_attrs, Float, last);      

                    buffers.insert(attr.clone(), BufferGPU::Float(current_buffer));
                },
                AvailableBufferType::UInt => {
                    let mut current_attrs: Vec<u32> = vec![];
                    for row in grid.iter() {
                        for i in row.iter() {
                            match i.get_attribute(&attr) {
                                Some(BufferType::UInt(val)) => current_attrs.push(val),
                                Some(_) => unreachable!(),
                                None => current_attrs.push(0),
                            };
                        }
                    }

                    write_buffer!(current_buffer, context, queue, size, &current_attrs, UInt, last);       

                    buffers.insert(attr.clone(), BufferGPU::UInt(current_buffer));
                },
                _ => unreachable!(),
            }
        }

        let mut receptor_flags: Vec<u32> = vec![];
        for row in grid.iter() {
            for i in row.iter() {
                for n in <Self as Receptors>::N::get_all_types() {
                    match i.receptors.get(&n) {
                        Some(_) => receptor_flags.push(1),
                        None => receptor_flags.push(0),
                    };
                }
            }
        }

        let flags_size = size * <Self as Receptors>::N::number_of_types();

        write_buffer!(flag_buffer, context, queue, flags_size, &receptor_flags, UInt, last);

        buffers.insert(String::from("receptors$flags"), BufferGPU::UInt(flag_buffer));

        Ok(buffers)
    }

    fn get_all_top_level_attributes() -> HashSet<(String, AvailableBufferType)> {
        HashSet::from([])
    }

    fn get_attributes_associated_with(neurotransmitter: &DefaultReceptorsNeurotransmitterType) -> HashSet<(String, AvailableBufferType)> {
        match neurotransmitter {
            DefaultReceptorsNeurotransmitterType::X => {
                let mut attrs = HashSet::from([(String::from("receptors$X_current"), AvailableBufferType::Float), (String::from("receptors$X_g"), AvailableBufferType::Float), (String::from("receptors$X_e"), AvailableBufferType::Float)]); 
                attrs.extend(
                    T::get_attribute_names().iter().map(|(i, j)|
                        (
                            format!(
                                "receptors$X$r$kinetics${}",
                                i.split("$").collect::<Vec<_>>().last()
                                    .expect("Invalid attribute")
                            ), 
                            *j
                        )
                    ).collect::<Vec<(_, _)>>()
                );
                attrs
            }
        }
    }

    fn convert_to_cpu(
        grid: &mut [Vec<Self>],
        buffers: &HashMap<String, BufferGPU>,
        queue: &CommandQueue,
        rows: usize,
        cols: usize,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            for inner in grid {
                inner.clear();
            }

            return Ok(());
        }

        let mut cpu_conversion: HashMap<String, Vec<BufferType>> = HashMap::new();

        for key in Self::get_all_attributes() {
            match key.1 {
                AvailableBufferType::Float => {
                    let mut current_contents = vec![0.; rows * cols];
                    read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, Float);        

                    let current_contents = current_contents.iter()
                        .map(|i| BufferType::Float(*i))
                        .collect::<Vec<BufferType>>();

                    cpu_conversion.insert(key.0.clone(), current_contents);
                },
                AvailableBufferType::UInt => {
                    let mut current_contents = vec![0; rows * cols];
                    read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                    let current_contents = current_contents.iter()
                        .map(|i| BufferType::UInt(*i))
                        .collect::<Vec<BufferType>>();

                    cpu_conversion.insert(key.0.clone(), current_contents);
                },
                _ => unreachable!(),
            }
        }

        let mut current_contents = vec![0; rows * cols * <Self as Receptors>::N::number_of_types()];   
        read_and_set_buffer!(buffers, queue, "receptors$flags", &mut current_contents, UInt);

        let flags = current_contents.iter().map(|i| *i == 1).collect::<Vec<bool>>();

        for row in 0..rows {
            for col in 0..cols {
                let current_index = row * cols + col;

                for i in Self::get_all_top_level_attributes() {
                    grid[row][col].set_attribute(
                        &i.0,
                        cpu_conversion.get(&i.0).unwrap()[current_index]
                    ).unwrap();
                }
                for i in <Self as Receptors>::N::get_all_types() {
                    if flags[current_index * <Self as Receptors>::N::number_of_types() + i.type_to_numeric()] {
                        for attr in Self::get_attributes_associated_with(&i) {
                            match grid[row][col].receptors.get_mut(&i) {
                                Some(_) => grid[row][col].set_attribute(
                                    &attr.0,
                                    cpu_conversion.get(&attr.0).unwrap()[current_index]
                                ).unwrap(),
                                None => {
                                    grid[row][col].receptors.insert(
                                        i,
                                        i.get_associated_receptor()
                                    );
                                    grid[row][col].set_attribute(
                                        &attr.0,
                                        cpu_conversion.get(&attr.0).unwrap()[current_index]
                                    ).unwrap();
                                }
                            };
                        }
                    } else {
                        let _ = grid[row][col].receptors.remove(&i);
                    }
                }
            }
        }

        Ok(())
    }

    fn get_updates() -> Vec<(String, Vec<(String, AvailableBufferType)>)> {
        vec![(String::from("__kernel void update_X(uint index, __global float *current_voltage, __global float *r, __global float *current, __global float *g, __global float *e) {
        current[index] = ((g[index] * r[index]) * (current_voltage[index] - e[index]));
        }"), 
        vec![(String::from("index"), AvailableBufferType::UInt), (String::from("current_voltage"), AvailableBufferType::Float), (String::from("receptors$X$r$kinetics$r"), AvailableBufferType::Float), (String::from("receptors$X_current"), AvailableBufferType::Float), (String::from("receptors$X_g"), AvailableBufferType::Float), (String::from("receptors$X_e"), AvailableBufferType::Float)])]
    }
}

/// An encapsulation for necessary receptor traits
pub trait Receptors: Clone {
    type T: ReceptorKinetics;
    type N: NeurotransmitterType;
    type R;

    /// Updates the receptor gating values of all the inserted receptors
    fn update_receptor_kinetics(
        &mut self,
        t_total: &NeurotransmitterConcentrations<Self::N>,
        dt: f32,
    );
    /// Retrieves a reference to a receptor for the given neurotransmitter
    fn get(&self, neurotransmitter_type: &Self::N) -> Option<&Self::R>;
    /// Retrieves a mutable reference to a receptor for the given neurotransmitter
    fn get_mut(&mut self, neurotransmitter_type: &Self::N) -> Option<&mut Self::R>;
    /// Returns how many receptors are present
    fn len(&self) -> usize;
    /// Returns whether receptors are present
    fn is_empty(&self) -> bool;
    /// Inserts a new receptor and checks that it is of the correct neurotransmitter and receptor type
    fn insert(
        &mut self,
        neurotransmitter_type: Self::N,
        receptor_type: Self::R,
    ) -> Result<(), ReceptorNeurotransmitterError>;
    /// Removes a given receptor and returns it if the receptor was present
    fn remove(&mut self, neurotransmitter_type: &Self::N) -> Option<Self::R>;
}

/// An encapsulation of behaviors for receptors with ionotropic mechanisms
pub trait IonotropicReception: Receptors {
    /// Sets the receptor currents given a voltage and timestep
    fn set_receptor_currents(&mut self, current_voltage: f32, dt: f32);
    /// Gets the currents given a timestep and capacitance
    fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32;
}

#[cfg(feature = "gpu")]
/// An encapsulation of behaviors for receptors on the GPU
pub trait ReceptorsGPU: Receptors 
where
    Self::T: ReceptorKineticsGPU,
    Self::N: NeurotransmitterTypeGPU,
{
    /// Gets a given attribute from the receptors
    fn get_attribute(&self, attribute: &str) -> Option<BufferType>;
    /// Gets a sets attribute in the receptors
    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error>;
    /// Gets all possible attributes
    fn get_all_attributes() -> HashSet<(String, AvailableBufferType)>;
    /// Gets all attributes outside of individual receptors
    fn get_all_top_level_attributes() -> HashSet<(String, AvailableBufferType)>;
    /// Gets all attributes associated with a specific neurotransmitter type
    fn get_attributes_associated_with(
        neurotransmitter: &<Self as Receptors>::N,
    ) -> HashSet<(String, AvailableBufferType)>;
    /// Gets the functions to update each receptor
    fn get_updates() -> Vec<(String, Vec<(String, AvailableBufferType)>)>;
    /// Converts the representation to one that can be used on the GPU
    fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts the GPU representation to a CPU representation
    fn convert_to_cpu(
        grid: &mut [Vec<Self>],
        buffers: &HashMap<String, BufferGPU>,
        queue: &CommandQueue,
        rows: usize,
        cols: usize,
    ) -> Result<(), GPUError>;
}

#[derive(Debug, Clone)]
pub enum DefaultReceptorsType<T: ReceptorKinetics> {
    X(XReceptor<T>),
}

#[derive(Debug, Clone)]
pub struct DefaultReceptors<T: ReceptorKinetics> {
    receptors: HashMap<DefaultReceptorsNeurotransmitterType, DefaultReceptorsType<T>>,
}

impl<T: ReceptorKinetics> Receptors for DefaultReceptors<T> {
    type T = T;
    type N = DefaultReceptorsNeurotransmitterType;
    type R = DefaultReceptorsType<T>;

    fn update_receptor_kinetics(
        &mut self,
        t_total: &NeurotransmitterConcentrations<DefaultReceptorsNeurotransmitterType>,
        dt: f32,
    ) {
        t_total.iter().for_each(|(key, value)| {
            if let Some(receptor_type) = self.receptors.get_mut(key) {
                match receptor_type {
                    DefaultReceptorsType::X(receptor) => {
                        receptor.apply_r_change(*value, dt);
                    }
                }
            }
        });
    }

    fn get(
        &self,
        neurotransmitter_type: &DefaultReceptorsNeurotransmitterType,
    ) -> Option<&DefaultReceptorsType<T>> {
        self.receptors.get(neurotransmitter_type)
    }

    fn get_mut(
        &mut self,
        neurotransmitter_type: &DefaultReceptorsNeurotransmitterType,
    ) -> Option<&mut DefaultReceptorsType<T>> {
        self.receptors.get_mut(neurotransmitter_type)
    }

    fn len(&self) -> usize {
        self.receptors.len()
    }

    fn is_empty(&self) -> bool {
        self.receptors.is_empty()
    }

    fn insert(
        &mut self,
        neurotransmitter_type: DefaultReceptorsNeurotransmitterType,
        receptor_type: DefaultReceptorsType<T>,
    ) -> Result<(), ReceptorNeurotransmitterError> {
        self.receptors.insert(neurotransmitter_type, receptor_type);

        Ok(())
    }

    fn remove(&mut self, neurotransmitter_type: &DefaultReceptorsNeurotransmitterType) -> Option<DefaultReceptorsType<T>> {
        self.receptors.remove(&neurotransmitter_type)
    }
}

impl<T: ReceptorKinetics> IonotropicReception for DefaultReceptors<T> {
    fn set_receptor_currents(&mut self, current_voltage: f32, dt: f32) {
        if let Some(DefaultReceptorsType::X(receptor)) = self.receptors.get_mut(&DefaultReceptorsNeurotransmitterType::X) {
            receptor.iterate(current_voltage, dt);
        }
    }

    fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32 {
        let mut total = 0.;
        if let Some(DefaultReceptorsType::X(receptor)) = self.receptors.get(&DefaultReceptorsNeurotransmitterType::X) {
            total += receptor.current;
        };
        total * (dt / c_m)
    }
}

impl<T: ReceptorKinetics> Default for DefaultReceptors<T> {
    fn default() -> Self {
        DefaultReceptors {
            receptors: HashMap::new(),
        }
    }
}

/// Enum containing the type of ionotropic ligand gated receptor
/// containing a modifier to use when calculating current
#[derive(Debug, Clone, PartialEq)]
pub enum IonotropicLigandGatedReceptorType {
    /// AMPA receptor
    AMPA,
    /// GABAa receptor
    GABAa,
    /// GABAb receptor with dissociation modifier
    GABAb(GABAbDissociation),
    /// NMDA receptor with magnesium and voltage modifier
    NMDA(BV),
}

/// Singular ligand gated channel 
#[derive(Debug, Clone, PartialEq)]
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
            receptor_type: IonotropicLigandGatedReceptorType::AMPA,
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
            receptor_type: IonotropicLigandGatedReceptorType::GABAa,
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
            IonotropicLigandGatedReceptorType::AMPA => 1.0,
            IonotropicLigandGatedReceptorType::GABAa => 1.0,
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

#[cfg(feature = "gpu")]
pub fn generate_unique_prefix(other_attributes: &[String], prefix: &str) -> String {
    let mut num = 0;
    let mut unique = false;

    while !unique {
        for i in other_attributes {
            if i == &format!("{}{}_", prefix, num) {
                num += 1;
                break;
            }
        }

        unique = true;
    }

    format!("{}{}_", prefix, num)
}

#[cfg(feature = "gpu")]
impl<T: ReceptorKineticsGPU> LigandGatedChannel<T> {
    /// Retrieves a given attribute from the ligand gated channel
    fn get_attribute(&self, attribute: &str) -> Option<BufferType> {
        match attribute {
            "ligand_gates$current" => Some(BufferType::Float(self.current)),
            "ligand_gates$reversal" => Some(BufferType::Float(self.reversal)),
            "ligand_gates$g" => Some(BufferType::Float(self.g)),
            "ligand_gates$nmda_mg" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::NMDA(value) => Some(BufferType::Float(value.mg)),
                _ => None
            },
            "ligand_gates$gabab_g" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(value) => Some(BufferType::Float(value.g)),
                _ => None
            },
            "ligand_gates$gabab_k3" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(value) => Some(BufferType::Float(value.k3)),
                _ => None
            },
            "ligand_gates$gabab_k4" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(value) => Some(BufferType::Float(value.k4)),
                _ => None
            },
            "ligand_gates$gabab_kd" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(value) => Some(BufferType::Float(value.kd)),
                _ => None
            },
            "ligand_gates$gabab_n" => match &self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(value) => Some(BufferType::Float(value.n)),
                _ => None
            },
            _ => {
                self.receptor.get_attribute(attribute)
            },
        }
    }

    /// Sets a given attribute to a given value
    fn set_attribute(&mut self, attribute: &str, value: BufferType) {
        match attribute {
            "ligand_gates$current" => self.current = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => unreachable!("Incorrect type passed"),
            },
            "ligand_gates$reversal" => self.reversal = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => unreachable!("Incorrect type passed"),
            },
            "ligand_gates$g" => self.g = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => unreachable!("Incorrect type passed"),
            },
            "ligand_gates$nmda_mg" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::NMDA(current_value) => current_value.mg = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set NMDA value with non NMDA receptor")
            },
            "ligand_gates$gabab_g" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(current_value) => current_value.g = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set GABAb value with non GABAb receptor")
            },
            "ligand_gates$gabab_k3" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(current_value) => current_value.k3 = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set GABAb value with non GABAb receptor")
            },
            "ligand_gates$gabab_k4" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(current_value) => current_value.k4 = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set GABAb value with non GABAb receptor")
            },
            "ligand_gates$gabab_kd" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(current_value) => current_value.kd = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set GABAb value with non GABAb receptor")
            }
            "ligand_gates$gabab_n" => match &mut self.receptor_type {
                IonotropicLigandGatedReceptorType::GABAb(current_value) => current_value.n = match value {
                    BufferType::Float(nested_val) => nested_val,
                    _ => unreachable!("Incorrect type passed"),
                },
                _ => unreachable!("Cannot set GABAb value with non GABAb receptor")
            }
            _ => {
                self.receptor.set_attribute(attribute, value).unwrap()
            },
        }
    }

    /// Gets all possible attribute names
    pub fn get_all_possible_attribute_names() -> HashSet<(String, AvailableBufferType)> {
        let mut attributes = HashSet::from([
            (String::from("ligand_gates$current"), AvailableBufferType::Float), 
            (String::from("ligand_gates$reversal"), AvailableBufferType::Float), 
            (String::from("ligand_gates$g"), AvailableBufferType::Float),
            (String::from("ligand_gates$nmda_mg"), AvailableBufferType::Float), 
            (String::from("ligand_gates$gabab_g"), AvailableBufferType::Float),
            (String::from("ligand_gates$gabab_k3"), AvailableBufferType::Float),
            (String::from("ligand_gates$gabab_k4"), AvailableBufferType::Float), 
            (String::from("ligand_gates$gabab_kd"), AvailableBufferType::Float), 
            (String::from("ligand_gates$gabab_n"), AvailableBufferType::Float),
        ]);

        attributes.extend(T::get_attribute_names());
        
        attributes
    }

    pub fn get_all_possible_attribute_names_ordered() -> BTreeSet<(String, AvailableBufferType)> {
        LigandGatedChannel::<T>::get_all_possible_attribute_names().into_iter().collect()
    }

    /// Gets all valid attribute names
    fn get_valid_attribute_names(&self) -> HashSet<(String, AvailableBufferType)> {
        let mut attributes = HashSet::from([
            (String::from("ligand_gates$current"), AvailableBufferType::Float), 
            (String::from("ligand_gates$reversal"), AvailableBufferType::Float), 
            (String::from("ligand_gates$g"), AvailableBufferType::Float),
        ]);

        attributes.extend(T::get_attribute_names());

        match &self.receptor_type {
            IonotropicLigandGatedReceptorType::AMPA => attributes,
            IonotropicLigandGatedReceptorType::NMDA(_) => {
                attributes.insert((String::from("ligand_gates$nmda_mg"), AvailableBufferType::Float));

                attributes
            },
            IonotropicLigandGatedReceptorType::GABAa => attributes,
            IonotropicLigandGatedReceptorType::GABAb(_) => {
                attributes.insert((String::from("ligand_gates_gabab$g"), AvailableBufferType::Float));
                attributes.insert((String::from("ligand_gates_gabab$k3"), AvailableBufferType::Float));
                attributes.insert((String::from("ligand_gates_gabab$k4"), AvailableBufferType::Float));
                attributes.insert((String::from("ligand_gates_gabab$kd"), AvailableBufferType::Float));
                attributes.insert((String::from("ligand_gates_gabab$n"), AvailableBufferType::Float));

                attributes
            }
        }
    }
}

/// Multiple ligand gated channels with their associated neurotransmitter type
#[derive(Clone, Debug, PartialEq)]
pub struct LigandGatedChannels<T: ReceptorKinetics> { 
    ligand_gates: HashMap<IonotropicNeurotransmitterType, LigandGatedChannel<T>> 
}

impl<T: ReceptorKinetics> Default for LigandGatedChannels<T> {
    fn default() -> Self {
        LigandGatedChannels {
            ligand_gates: HashMap::new(),
        }
    }
}

fn matching_neurotransmitter_and_receptor_type(
    neurotransmitter_type: &IonotropicNeurotransmitterType,
    receptor_type: &IonotropicLigandGatedReceptorType,
) -> bool {
    if let IonotropicLigandGatedReceptorType::AMPA = receptor_type {
        if *neurotransmitter_type == IonotropicNeurotransmitterType::AMPA {
            return true;
        }
    }
    if let IonotropicLigandGatedReceptorType::NMDA(_) = receptor_type {
        if *neurotransmitter_type == IonotropicNeurotransmitterType::NMDA {
            return true;
        }
    }
    if let IonotropicLigandGatedReceptorType::GABAa = receptor_type {
        if *neurotransmitter_type == IonotropicNeurotransmitterType::GABAa {
            return true;
        }
    }
    if let IonotropicLigandGatedReceptorType::GABAb(_) = receptor_type {
        if *neurotransmitter_type == IonotropicNeurotransmitterType::GABAb {
            return true;
        }
    }

    false
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
    ) -> Result<(), ReceptorNeurotransmitterError> {
        if !matching_neurotransmitter_and_receptor_type(&neurotransmitter_type, &ligand_gate.receptor_type) {
            return Err(ReceptorNeurotransmitterError::MismatchedTypes);
        }

        self.ligand_gates.insert(neurotransmitter_type, ligand_gate);

        Ok(())
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

#[cfg(feature = "gpu")]
fn extract_or_pad_ligand_gates<T: ReceptorKineticsGPU>(
    value: &LigandGatedChannels<T>, 
    i: IonotropicNeurotransmitterType, 
    buffers_contents: &mut HashMap<String, Vec<BufferType>>,
    flags: &mut HashMap<String, Vec<u32>>,
) {
    match value.get(&i) {
        Some(current_value) => {
            if let Some(current_flag) = flags.get_mut(&format!("ligand_gates${}", i.to_string())) {
                current_flag.push(1);
            }

            for attribute in LigandGatedChannel::<T>::get_all_possible_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute.0) {
                    if current_value.get_valid_attribute_names().contains(&attribute) {
                        retrieved_attribute.push(
                            current_value.get_attribute(&attribute.0)
                                .unwrap_or_else(|| panic!("Attribute ({}) not found", attribute.0))
                        );
                    } else {
                        match attribute.1 {
                            AvailableBufferType::Float => retrieved_attribute.push(BufferType::Float(0.)),
                            AvailableBufferType::UInt => retrieved_attribute.push(BufferType::UInt(0)),
                            AvailableBufferType::OptionalUInt => retrieved_attribute.push(BufferType::OptionalUInt(-1))
                        };
                    }
                } else {
                    unreachable!("Attribute ({}) not found", attribute.0);
                }
            }
        },
        None => {
            if let Some(current_flag) = flags.get_mut(&format!("ligand_gates${}", i.to_string())) {
                current_flag.push(0);
            }

            for attribute in LigandGatedChannel::<T>::get_all_possible_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute.0) {
                    match attribute.1 {
                        AvailableBufferType::Float => retrieved_attribute.push(BufferType::Float(0.)),
                        AvailableBufferType::UInt => retrieved_attribute.push(BufferType::UInt(0)),
                        AvailableBufferType::OptionalUInt => retrieved_attribute.push(BufferType::OptionalUInt(-1))
                    };
                } else {
                    unreachable!("Attribute ({}) not found", attribute.0)
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn get_receptor_args<T: ReceptorKineticsGPU>(indexer: &str) -> String {
    T::get_update_function().0
        .iter()
        .map(|i| format!("{}{}", i.split("$").collect::<Vec<&str>>()[1], indexer))
        .collect::<Vec<String>>()
        .join(", ")
}

#[cfg(feature = "gpu")]
impl <T: ReceptorKineticsGPU + AMPADefault + NMDADefault + GABAaDefault + GABAbDefault> LigandGatedChannels<T> {
    pub fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        // aggreate a list of all possible attributes (prefix those that are sub attributes)
        // flags that depend on
        // add to list 

        let length: usize = grid.iter().map(|row| row.len()).sum();

        if length == 0 {
            return Ok(HashMap::new());
        }

        let mut buffers_contents: HashMap<String, Vec<BufferType>> = HashMap::new();
        for i in T::get_attribute_names() {
            buffers_contents.insert(i.0.to_string(), vec![]);
        }
        for i in LigandGatedChannel::<T>::get_all_possible_attribute_names() {
            buffers_contents.insert(i.0.to_string(), vec![]);
        }

        let mut flags: HashMap<String, Vec<u32>> = HashMap::new();
        for i in IonotropicNeurotransmitterType::get_all_types() {
            flags.insert(format!("ligand_gates${}", i.to_string()), vec![]);
        }

        for row in grid.iter() {
            for value in row.iter() {
                for i in IonotropicNeurotransmitterType::get_all_types() {
                    extract_or_pad_ligand_gates(value, i, &mut buffers_contents, &mut flags);
                }
            }
        }

        let mut buffers: HashMap<String, BufferGPU> = HashMap::new();

        let size = length * IonotropicNeurotransmitterType::number_of_types();

        for (key, value) in buffers_contents.iter() {
            match value[0] {
                BufferType::Float(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::Float(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<f32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, Float, last);

                    buffers.insert(key.clone(), BufferGPU::Float(current_buffer));
                },
                BufferType::UInt(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::UInt(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<u32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, UInt, last);

                    buffers.insert(key.clone(), BufferGPU::UInt(current_buffer));
                },
                BufferType::OptionalUInt(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::OptionalUInt(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<i32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, OptionalUInt, last);

                    buffers.insert(key.clone(), BufferGPU::OptionalUInt(current_buffer));
                }
            };  
        }

        let mut flags_vec: Vec<u32> = vec![];

        for n in 0..length {
            for i in IonotropicNeurotransmitterType::get_all_types() {
                flags_vec.push(
                    flags.get(
                        &format!("ligand_gates${}", i.to_string())
                    ).unwrap()[n]
                );
            }
        }

        write_buffer!(flags_buffer, context, queue, size, &flags_vec, UInt, last);

        buffers.insert(String::from("ligand_gates$flags"), BufferGPU::UInt(flags_buffer));

        Ok(buffers)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn convert_to_cpu(
        ligand_gates_grid: &mut [Vec<Self>],
        buffers: &HashMap<String, BufferGPU>,
        queue: &CommandQueue,
        rows: usize,
        cols: usize,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            for inner in ligand_gates_grid {
                inner.clear();
            }

            return Ok(());
        }

        let mut cpu_conversion: HashMap<String, Vec<BufferType>> = HashMap::new();
        let mut flags: HashMap<String, Vec<bool>> = HashMap::new();
        let mut current_flags: Vec<bool> = vec![];

        let string_types: HashSet<(String, AvailableBufferType)> = HashSet::from([
            (String::from("ligand_gates$flags"), AvailableBufferType::UInt)
        ]);

        for key in LigandGatedChannel::<T>::get_all_possible_attribute_names().union(&string_types) {
            if !string_types.contains(key) {
                match key.1 {
                    AvailableBufferType::Float => {
                        let mut current_contents = vec![0.; rows * cols * IonotropicNeurotransmitterType::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, Float);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::Float(*i))
                            .collect::<Vec<BufferType>>();

                        cpu_conversion.insert(key.0.clone(), current_contents);
                    },
                    AvailableBufferType::UInt => {
                        let mut current_contents = vec![0; rows * cols * IonotropicNeurotransmitterType::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::UInt(*i))
                            .collect::<Vec<BufferType>>();

                        cpu_conversion.insert(key.0.clone(), current_contents);
                    },
                    AvailableBufferType::OptionalUInt => {
                        let mut current_contents = vec![-1; rows * cols * IonotropicNeurotransmitterType::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, OptionalUInt);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::OptionalUInt(*i))
                            .collect::<Vec<BufferType>>();

                        cpu_conversion.insert(key.0.clone(), current_contents);
                    }
                }
            } else {
                let mut current_contents = vec![0; rows * cols * IonotropicNeurotransmitterType::number_of_types()];
                read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                current_flags = current_contents.iter().map(|i| *i == 1).collect::<Vec<bool>>();
            }
        }

        for n in 0..(rows * cols) {
            for (n_type, i) in IonotropicNeurotransmitterType::get_all_types().iter().enumerate() {
                let current_index = n * IonotropicNeurotransmitterType::number_of_types() + n_type;
                match flags.entry(format!("ligand_gates${}", i.to_string())) {
                    Entry::Vacant(entry) => { entry.insert(vec![current_flags[current_index]]); },
                    Entry::Occupied(mut entry) => { entry.get_mut().push(current_flags[current_index]); }
                }
            }
        }

        for row in 0..rows {
            for col in 0..cols {
                let grid_value = &mut ligand_gates_grid[row][col];
                let flag_index = row * cols + col;
                for i in IonotropicNeurotransmitterType::get_all_types() {
                    let i_str = format!("ligand_gates${}", i.to_string());
                    let index = row * cols * IonotropicNeurotransmitterType::number_of_types() 
                        + col * IonotropicNeurotransmitterType::number_of_types() + i.type_to_numeric();
    
                    if let Some(flag) = flags.get(&i_str) {
                        if flag[flag_index] {
                            if !grid_value.ligand_gates.contains_key(&i) {
                                match i {
                                    IonotropicNeurotransmitterType::AMPA => grid_value.insert(
                                        i, LigandGatedChannel::ampa_default()
                                    ).expect("Should insert correct channel type"),
                                    IonotropicNeurotransmitterType::NMDA => grid_value.insert(
                                        i, LigandGatedChannel::nmda_default()
                                    ).expect("Should insert correct channel type"),
                                    IonotropicNeurotransmitterType::GABAa => grid_value.insert(
                                        i, LigandGatedChannel::gabaa_default()
                                    ).expect("Should insert correct channel type"),
                                    IonotropicNeurotransmitterType::GABAb => grid_value.insert(
                                        i, LigandGatedChannel::gabab_default()
                                    ).expect("Should insert correct channel type"),
                                };
                            }

                            let current_ligand_gated_channel = grid_value.get_mut(&i).unwrap();
    
                            for attribute in current_ligand_gated_channel.get_valid_attribute_names() {
                                if let Some(values) = cpu_conversion.get(&attribute.0) {
                                    current_ligand_gated_channel.set_attribute(&attribute.0, values[index]);
                                }
                            }
                        } else {
                            grid_value.ligand_gates.remove(&i);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn get_ligand_gated_channels_update_function() -> String {
        let mut kernel_args = vec![
            String::from("uint index"), 
            String::from("uint number_of_types"),
            String::from("__global float* t"),
            String::from("__global float* voltage"), 
            String::from("__global float* dt"), 
            String::from("__global uint* lg_flags"),
        ];
        let ligand_gates_args = LigandGatedChannel::<T>::get_all_possible_attribute_names_ordered()
            .iter()
            .map(|i| format!("__global {}* {}", i.1.to_str(), i.0.split("$").collect::<Vec<&str>>()[1]))
            .collect::<Vec<String>>();
        kernel_args.extend(ligand_gates_args);
        let kernel_args = kernel_args.join(",\n");
        format!(
            r#"
            __kernel void ligand_gates_update_function(
                {}
            ) {{
                if (lg_flags[index * number_of_types]) {{ // AMPA
                    r[index * number_of_types] = get_r({});
                    current[index * number_of_types] = g[index * number_of_types] * r[index * number_of_types] * (voltage[index] - reversal[index * number_of_types]); 
                }}
                if (lg_flags[index * number_of_types + 1]) {{ // NMDA
                    r[index * number_of_types + 1] = get_r({});
                    float modifier = 1.0 / (1.0 + (exp(-0.062 * voltage[index]) * nmda_mg[index * number_of_types + 1] / 3.57)); 
                    current[index * number_of_types + 1] = modifier * g[index * number_of_types + 1] * r[index * number_of_types + 1] * (voltage[index] - reversal[index * number_of_types + 1]);
                }}
                if (lg_flags[index * number_of_types + 2]) {{ // GABAa 
                    r[index * number_of_types + 2] = get_r({});
                    current[index * number_of_types + 2] = g[index * number_of_types + 2] * r[index * number_of_types + 2] * (voltage[index] - reversal[index * number_of_types + 2]); 
                }}
                if (lg_flags[index * number_of_types + 3]) {{ // GABAb
                    r[index * number_of_types + 3] = get_r({});
                    gabab_g[index * number_of_types + 3] += (gabab_k3[index * number_of_types + 3] * r[index * number_of_types + 3] - gabab_k4[index * number_of_types + 3] * gabab_g[index * number_of_types + 3]) * dt[index];
                    float bottom = pow(gabab_g[index * number_of_types + 3], gabab_n[index * number_of_types + 3]) * gabab_kd[index * number_of_types + 3];
                    float top = pow(gabab_g[index * number_of_types + 3], gabab_n[index * number_of_types + 3]);
                    float modifier =  top / bottom;
                    current[index * number_of_types + 3] = modifier * g[index * number_of_types + 3] * r[index * number_of_types + 3] * (voltage[index] - reversal[index * number_of_types * number_of_types + 3]);
                }}
            }}
            "#,
            kernel_args,
            get_receptor_args::<T>("[index * number_of_types]"),
            get_receptor_args::<T>("[index * number_of_types + 1]"),
            get_receptor_args::<T>("[index * number_of_types + 2]"),
            get_receptor_args::<T>("[index * number_of_types + 3]"),
        )
    }
}

/// Multiple neurotransmitters with their associated types
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Neurotransmitters<N: NeurotransmitterType, T: NeurotransmitterKinetics> {
    neurotransmitters: HashMap<N, T>
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
    pub fn apply_t_changes<U: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &U) {
        self.neurotransmitters.values_mut()
            .for_each(|value| value.apply_t_change(neuron));
    }
}


#[cfg(feature = "gpu")]
#[macro_export]
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
pub use read_and_set_buffer;

#[cfg(feature = "gpu")]
#[macro_export]
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
pub use write_buffer;

#[cfg(feature = "gpu")]
#[macro_export]
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
pub use flatten_and_retrieve_field;

#[cfg(feature = "gpu")]
#[macro_export]
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
pub use create_float_buffer;

#[cfg(feature = "gpu")]
#[macro_export]
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
pub use create_uint_buffer;

#[cfg(feature = "gpu")]
#[macro_export]
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
pub use create_optional_uint_buffer;

#[cfg(feature = "gpu")]
fn extract_or_pad_neurotransmitter<N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU>(
    value: &Neurotransmitters<N, T>, 
    i: N, 
    buffers_contents: &mut HashMap<String, Vec<BufferType>>,
    flags: &mut HashMap<String, Vec<u32>>,
) {
    match value.get(&i) {
        Some(value) => {
            if let Some(current_flag) = flags.get_mut(&format!("neurotransmitters${}", i.to_string())) {
                current_flag.push(1);
            }

            for attribute in T::get_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute.0) {
                    retrieved_attribute.push(
                        value.get_attribute(&attribute.0)
                            .unwrap_or_else(|| panic!("Attribute ({}) not found", attribute.0))
                    )
                } else {
                    unreachable!("Attribute ({}) not found", attribute.0)
                }
            }
        },
        None => {
            if let Some(current_flag) = flags.get_mut(&format!("neurotransmitters${}", i.to_string())) {
                current_flag.push(0);
            }

            for attribute in T::get_attribute_names() {
                if let Some(retrieved_attribute) = buffers_contents.get_mut(&attribute.0) {
                    match attribute.1 {
                        AvailableBufferType::Float => retrieved_attribute.push(BufferType::Float(0.)),
                        AvailableBufferType::UInt => retrieved_attribute.push(BufferType::UInt(0)),
                        AvailableBufferType::OptionalUInt => retrieved_attribute.push(BufferType::OptionalUInt(-1))
                    };
                } else {
                    unreachable!("Attribute ({}) not found", attribute.0)
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl <N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU> Neurotransmitters<N, T> {
    pub fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        let length: usize = grid.iter().map(|row| row.len()).sum();

        if length == 0 {
            return Ok(HashMap::new());
        }

        let mut buffers_contents: HashMap<String, Vec<BufferType>> = HashMap::new();
        for i in T::get_attribute_names() {
            buffers_contents.insert(i.0.to_string(), vec![]);
        }

        let mut flags: HashMap<String, Vec<u32>> = HashMap::new();
        for i in N::get_all_types() {
            flags.insert(format!("neurotransmitters${}", i.to_string()), vec![]);
        }

        for row in grid.iter() {
            for value in row.iter() {
                for i in N::get_all_types() {
                    extract_or_pad_neurotransmitter(value, i, &mut buffers_contents, &mut flags);
                }
            }
        }

        let mut buffers: HashMap<String, BufferGPU> = HashMap::new();

        let size = length * N::number_of_types();

        for (key, value) in buffers_contents.iter() {
            match value[0] {
                BufferType::Float(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::Float(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<f32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, Float, last);

                    buffers.insert(key.clone(), BufferGPU::Float(current_buffer));
                },
                BufferType::UInt(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::UInt(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<u32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, UInt, last);

                    buffers.insert(key.clone(), BufferGPU::UInt(current_buffer));
                },
                BufferType::OptionalUInt(_) => {
                    let values = value.iter()
                        .map(|i| match i {
                            BufferType::OptionalUInt(inner_value) => *inner_value,
                            _ => unreachable!("Incorrect type passed",)
                        })
                        .collect::<Vec<i32>>();

                    write_buffer!(current_buffer, context, queue, size, &values, OptionalUInt, last);

                    buffers.insert(key.clone(), BufferGPU::OptionalUInt(current_buffer));
                },
            }
        }

        let mut flags_vec: Vec<u32> = vec![];

        for n in 0..length {
            for i in N::get_all_types() {
                flags_vec.push(
                    flags.get(
                        &format!("neurotransmitters${}", i.to_string())
                    ).unwrap()[n]
                );
            }
        }

        write_buffer!(flags_buffer, context, queue, size, &flags_vec, UInt, last);

        buffers.insert(String::from("neurotransmitters$flags"), BufferGPU::UInt(flags_buffer));

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
        let mut cpu_conversion: HashMap<String, Vec<BufferType>> = HashMap::new();
        let mut current_flags: Vec<bool> = vec![];
        let mut flags: HashMap<String, Vec<bool>> = HashMap::new();

        let string_types: Vec<String> = T::get_attribute_names()
            .into_iter()
            .map(|i| i.0)
            .collect();

        if rows == 0 || cols == 0 {
            for inner in neurotransmitter_grid {
                inner.clear();
            }

            return Ok(());
        }

        let neurotransmitter_flags: HashSet<(String, AvailableBufferType)> = HashSet::from([
            (String::from("neurotransmitters$flags"), AvailableBufferType::UInt)
        ]);

        for key in T::get_attribute_names().union(&neurotransmitter_flags) {
            if string_types.contains(&key.0) {
                match key.1 {
                    AvailableBufferType::Float => {
                        let mut current_contents = vec![0.; rows * cols * N::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, Float);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::Float(*i))
                            .collect::<Vec<BufferType>>();
        
                        cpu_conversion.insert(key.0.clone(), current_contents);
                    },
                    AvailableBufferType::UInt => {
                        let mut current_contents = vec![0; rows * cols * N::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::UInt(*i))
                            .collect::<Vec<BufferType>>();
        
                        cpu_conversion.insert(key.0.clone(), current_contents);
                    },
                    AvailableBufferType::OptionalUInt => {
                        let mut current_contents = vec![-1; rows * cols * N::number_of_types()];
                        read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, OptionalUInt);

                        let current_contents = current_contents.iter()
                            .map(|i| BufferType::OptionalUInt(*i))
                            .collect::<Vec<BufferType>>();
        
                        cpu_conversion.insert(key.0.clone(), current_contents);
                    },
                }
            } else {
                let mut current_contents = vec![0; rows * cols * N::number_of_types()];
                read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                current_flags = current_contents.iter().map(|i| *i == 1).collect::<Vec<bool>>();
            }
        }

        for n in 0..(rows * cols) {
            for (n_type, i) in N::get_all_types().iter().enumerate() {
                let current_index = n * N::number_of_types() + n_type;
                match flags.entry(format!("neurotransmitters${}", i.to_string())) {
                    Entry::Vacant(entry) => { entry.insert(vec![current_flags[current_index]]); },
                    Entry::Occupied(mut entry) => { entry.get_mut().push(current_flags[current_index]); }
                }
            }
        }

        for row in 0..rows {
            for col in 0..cols {
                let grid_value = &mut neurotransmitter_grid[row][col];
                let flag_index = row * cols + col;
                for i in N::get_all_types() {
                    let i_str = format!("neurotransmitters${}", i.to_string());
                    let index = row * cols * N::number_of_types() 
                        + col * N::number_of_types() + i.type_to_numeric();
    
                    if let Some(flag) = flags.get(&i_str) {
                        if flag[flag_index] {
                            if !grid_value.neurotransmitters.contains_key(&i) {
                                grid_value.insert(i, T::default());
                            }
    
                            for attribute in T::get_attribute_names() {
                                if let Some(values) = cpu_conversion.get(&attribute.0) {
                                    grid_value.neurotransmitters
                                        .get_mut(&i)
                                        .unwrap()
                                        .set_attribute(&attribute.0, values[index])
                                        .unwrap();
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

    pub fn get_neurotransmitter_update_kernel_code() -> String {
        let mut raw_args = T::get_update_function().0.0;
        raw_args.extend(T::get_update_function().0.1);

        let kernel_args = raw_args
            .iter()
            .map(|i| {
                let split_result = i.split('$').collect::<Vec<&str>>();
                let arg_name = split_result.get(1).unwrap_or(&split_result[0]);
                if i != "is_spiking" {
                    format!("__global float* {}", arg_name)
                } else {
                    format!("__global uint* {}", arg_name)
                }
            })
            .collect::<Vec<String>>()
            .join(",\n");

        let func_args_from_neuron = T::get_update_function().0.0
            .iter()
            .map(|i| {
                let split_result = i.split('$').collect::<Vec<&str>>();
                let arg_name = split_result.get(1).unwrap_or(&split_result[0]);
                format!("{}[index]", arg_name)
            })
            .collect::<Vec<String>>()
            .join(",\n");
        let func_args = T::get_update_function().0.1
            .iter()
            .map(|i| {
                let split_result = i.split('$').collect::<Vec<&str>>();
                let arg_name = split_result.get(1).unwrap_or(&split_result[0]);
                format!("{}[index * number_of_types + i]", arg_name)
            })
            .collect::<Vec<String>>()
            .join(",\n");
        
        format!(
            r#"
                __kernel void neurotransmitters_update(
                    uint index,
                    uint number_of_types,
                    __global uint* neuro_flags,
                    {}
                ) {{
                    for (int i = 0; i < number_of_types; i++) {{
                        if (neuro_flags[index * number_of_types + i] == 1) {{
                            t[index * number_of_types + i] = get_t(
                                {},
                                {}
                            );
                        }}
                    }}
                }}
            "#,
            kernel_args,
            func_args_from_neuron,
            func_args,
        )
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
/// use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;
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
///         self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));
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
///         self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));
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
/// An encapsulation of a float or unsigned integer or optional unsigned integer buffer for the GPU
pub enum BufferGPU {
    Float(Buffer<cl_float>),
    UInt(Buffer<cl_uint>),
    OptionalUInt(Buffer<cl_int>),
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq)]
/// An encapsulation of a float or unsigned integer for converting to the GPU
pub enum BufferType {
    Float(f32),
    UInt(u32),
    OptionalUInt(i32),
}

#[cfg(feature = "gpu")]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
/// An encapsulation of the possible types for converting to the GPU
pub enum AvailableBufferType {
    Float,
    UInt,
    OptionalUInt
}

impl AvailableBufferType {
    pub fn to_str(&self) -> &str {
        match self {
            AvailableBufferType::Float => "float",
            AvailableBufferType::UInt => "uint",
            AvailableBufferType::OptionalUInt => "int",
        }
    }
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
    /// Returns the compiled kernel for chemical inputs
    fn iterate_and_spike_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
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
    /// Converts a grid of the neuron type to a vector of buffers with necessary chemical data
    fn convert_electrochemical_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts buffers back to a grid of neurons with necessary chemical data
    fn convert_electrochemical_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}
