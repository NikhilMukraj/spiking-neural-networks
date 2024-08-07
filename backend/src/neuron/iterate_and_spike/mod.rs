//! The [`IterateAndSpike`] trait for encapsulating basic neuronal and spiking dynamics
//! as well as [`NeurotransmitterKinetics`] for neurotransmission and [`ReceptorKinetics`]
//! for receptor dynamics over time.

use std::collections::{HashMap, hash_map::{Values, ValuesMut, Keys}};


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


/// Available neurotransmitter types for ionotropic receptor ligand gated channels
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub enum NeurotransmitterType {
    /// Unspecific general neurotransmitter
    Basic,
    /// Neurotransmitter type that effects only AMPA receptors
    AMPA,
    /// Neurotransmitter type that effects only NMDA receptors
    NMDA,
    /// Neurotransmitter type that effects only GABAa receptors
    GABAa,
    /// Neurotransmitter type that effects only GABAb receptors
    GABAb,
}

impl NeurotransmitterType {
    /// Converts type to string
    pub fn to_str(&self) -> &str {
        match self {
            NeurotransmitterType::Basic => "Basic",
            NeurotransmitterType::AMPA => "AMPA",
            NeurotransmitterType::GABAa => "GABAa",
            NeurotransmitterType::GABAb => "GABAb",
            NeurotransmitterType::NMDA => "NMDA",
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
#[derive(Debug, Clone, Copy)]
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
                    clearance_constant: 0.1,
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

/// An approximation of neurotransmitter kinetics that sets the concentration to the 
/// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
/// then immediately sets it to 0
#[derive(Debug, Clone, Copy)]
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
pub trait ReceptorKinetics: 
Clone + Default + AMPADefault + GABAaDefault + GABAbDefault + NMDADefault + Sync + Send {
    /// Calculates the change in receptor gating based on neurotransmitter input
    fn apply_r_change(&mut self, t: f32, dt: f32);
    /// Gets the receptor gating value
    fn get_r(&self) -> f32;
    fn set_r(&mut self, r: f32);
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
    /// Unspecified general ligand gated receptor
    Basic(f32),
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

impl<T: ReceptorKinetics> Default for LigandGatedChannel<T> {
    fn default() -> Self {
        LigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            receptor: T::default(),
            receptor_type: IonotropicLigandGatedReceptorType::Basic(1.0),
            current: 0.,
        }
    }
}

impl<T: ReceptorKinetics> AMPADefault for LigandGatedChannel<T> {
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

impl<T: ReceptorKinetics> GABAaDefault for LigandGatedChannel<T> {
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

impl<T: ReceptorKinetics> GABAbDefault for LigandGatedChannel<T> {
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

impl<T: ReceptorKinetics> NMDADefault for LigandGatedChannel<T> {
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
            IonotropicLigandGatedReceptorType::Basic(value) => *value,
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
    pub ligand_gates: HashMap<NeurotransmitterType, LigandGatedChannel<T>> 
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
    pub fn keys(&self) -> Keys<NeurotransmitterType, LigandGatedChannel<T>> {
        self.ligand_gates.keys()
    }

    /// Returns the ligand gates as a set of values
    pub fn values(&self) -> Values<NeurotransmitterType, LigandGatedChannel<T>> {
        self.ligand_gates.values()
    }

    /// Gets the ligand gate associated with the given [`NeurotransmitterType`]
    pub fn get(&self, neurotransmitter_type: &NeurotransmitterType) -> Option<&LigandGatedChannel<T>> {
        self.ligand_gates.get(neurotransmitter_type)
    }

    /// Gets a mutable reference to the ligand gate associated with the given [`NeurotransmitterType`]
    pub fn get_mut(&mut self, neurotransmitter_type: &NeurotransmitterType) -> Option<&mut LigandGatedChannel<T>> {
        self.ligand_gates.get_mut(neurotransmitter_type)
    }

    /// Inserts the given [`LigandGatedChannel`] with the associated [`NeurotransmitterType`]
    pub fn insert(
        &mut self, 
        neurotransmitter_type: NeurotransmitterType, 
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
    pub fn update_receptor_kinetics(&mut self, t_total: &NeurotransmitterConcentrations, dt: f32) {
        t_total.iter()
            .for_each(|(key, value)| {
                if let Some(gate) = self.ligand_gates.get_mut(key) {
                    gate.receptor.apply_r_change(*value, dt);
                }
            })
    }
}

/// Multiple neurotransmitters with their associated types
#[derive(Clone, Debug)]
pub struct Neurotransmitters<T: NeurotransmitterKinetics> {
    pub neurotransmitters: HashMap<NeurotransmitterType, T>
}

/// A hashmap of neurotransmitter types and their associated concentration
pub type NeurotransmitterConcentrations = HashMap<NeurotransmitterType, f32>;

impl<T: NeurotransmitterKinetics> Default for Neurotransmitters<T> {
    fn default() -> Self {
        Neurotransmitters {
            neurotransmitters: HashMap::new(),
        }
    }
}

impl <T: NeurotransmitterKinetics> Neurotransmitters<T> {
    /// Returns how many neurotransmitters there are
    pub fn len(&self) -> usize {
        self.neurotransmitters.keys().len()
    }

    /// Returns if neurotransmitters is empty
    pub fn is_empty(&self) -> bool {
        self.neurotransmitters.is_empty()
    }

    /// Returns the neurotransmitter types as a set of keys
    pub fn keys(&self) -> Keys<NeurotransmitterType, T> {
        self.neurotransmitters.keys()
    }

    // Returns the neurotransmitter dynamics as a set of values
    pub fn values(&self) -> Values<NeurotransmitterType, T> {
        self.neurotransmitters.values()
    }

    /// Returns a set of mutable neurotransmitters 
    pub fn values_mut(&mut self) -> ValuesMut<NeurotransmitterType, T> {
        self.neurotransmitters.values_mut()
    }

    /// Gets the neurotransmitter associated with the given [`NeurotransmitterType`]
    pub fn get(&self, neurotransmitter_type: &NeurotransmitterType) -> Option<&T> {
        self.neurotransmitters.get(neurotransmitter_type)
    }

    /// Gets a mutable reference to the neurotransmitter associated with the given [`NeurotransmitterType`]
    pub fn get_mut(&mut self, neurotransmitter_type: &NeurotransmitterType) -> Option<&mut T> {
        self.neurotransmitters.get_mut(neurotransmitter_type)
    }

    /// Inserts the given neurotransmitter with the associated [`NeurotransmitterType`]
    pub fn insert(
        &mut self, 
        neurotransmitter_type: NeurotransmitterType, 
        neurotransmitter: T
    ) {
        self.neurotransmitters.insert(neurotransmitter_type, neurotransmitter);
    }

    /// Returns the neurotransmitter concentration (mM) with their associated types
    pub fn get_concentrations(&self) -> NeurotransmitterConcentrations {
        self.neurotransmitters.iter()
            .map(|(neurotransmitter_type, neurotransmitter)| (*neurotransmitter_type, neurotransmitter.get_t()))
            .collect::<NeurotransmitterConcentrations>()
    }

    /// Calculates the neurotransmitter concentrations based on the given voltage (mV)
    pub fn apply_t_changes(&mut self, voltage: f32, dt: f32) {
        self.neurotransmitters.values_mut()
            .for_each(|value| value.apply_t_change(voltage, dt));
    }
}

/// Multiplies multiple neurotransmitters concentrations by a single scalar value
pub fn weight_neurotransmitter_concentration(
    neurotransmitter_hashmap: &mut NeurotransmitterConcentrations, 
    weight: f32
) {
    neurotransmitter_hashmap.values_mut().for_each(|value| *value *= weight);
}

/// Sums the neurotransmitter concentrations together
pub fn aggregate_neurotransmitter_concentrations(
    neurotransmitter_hashmaps: &Vec<NeurotransmitterConcentrations>
) -> NeurotransmitterConcentrations {
    let mut cumulative_map: NeurotransmitterConcentrations = HashMap::new();

    for map in neurotransmitter_hashmaps {
        for (key, value) in map {
            *cumulative_map.entry(*key).or_insert(0.0) += value;
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

/// Gets the noise factor for the neuron
pub trait GaussianFactor {
    fn get_gaussian_factor(&self) -> f32;
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
///     GaussianFactor, GaussianParameters, IsSpiking, Timestep,
///     CurrentVoltage, GapConductance, IterateAndSpike, 
///     LastFiringTime, NeurotransmitterConcentrations, LigandGatedChannels, 
///     ReceptorKinetics, NeurotransmitterKinetics, Neurotransmitters,
///     ApproximateNeurotransmitter, ApproximateReceptor,
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
///     /// Parameters used in generating noise
///     pub gaussian_params: GaussianParameters,
///     /// Postsynaptic neurotransmitters in cleft
///     pub synaptic_neurotransmitters: Neurotransmitters<T>,
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
///     fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
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
///         t_total: &NeurotransmitterConcentrations,
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
    CurrentVoltage + Timestep + GapConductance + GaussianFactor + IsSpiking + 
    LastFiringTime + Clone + Send + Sync 
{
    /// Takes in an input current and returns whether the model is spiking
    /// after the membrane potential is updated
    fn iterate_and_spike(&mut self, input_current: f32) -> bool;
    /// Gets the neurotransmitter concentrations of the neuron (mM)
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations;
    /// Takes in an input current and neurotransmitter input and returns whether the model
    /// is spiking after the membrane potential is updated, neurotransmitter input updates
    /// receptor currents based on the associated neurotransmitter concentration,
    /// the current from the receptors is also factored into the change in membrane potential
    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations,
    ) -> bool;
}
