use rand::Rng;
use std::{
    io::{Error, ErrorKind, Result},
    collections::{HashMap, hash_map::{Values, ValuesMut, Keys}},
};


#[derive(Debug, Clone, Copy)]
pub struct BV {
    pub mg_conc: f64,
}

impl Default for BV {
    fn default() -> Self {
        BV { mg_conc: 1.5 } // mM
    }
}

impl BV {
    fn calculate_b(&self, voltage: f64) -> f64 {
        1. / (1. + ((-0.062 * voltage).exp() * self.mg_conc / 3.57))
    }
}

#[derive(Debug, Clone)]
pub struct GABAbDissociation {
    pub g: f64,
    pub n: f64,
    pub kd: f64,
    // k1: ,
    // k2: ,
    pub k3: f64,
    pub k4: f64,
    pub dt: f64,
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
            dt: 0.1,
        }
    }
}

impl GABAbDissociation {
    fn calculate_modifer(&self) -> f64 {
        self.g.powf(self.n) / (self.g.powf(self.n) * self.kd)
    }
}

pub trait AMPADefault {
    fn ampa_default() -> Self;
}

pub trait GABAaDefault {
    fn gabaa_default() -> Self;
}

pub trait GABAbDefault {
    fn gabab_default() -> Self;
}

pub trait GABAbDefault2 {
    fn gabab_default2() -> Self;
}

pub trait NMDADefault {
    fn nmda_default() -> Self;
}

// pub trait NeurotransmitterTypeDefaults: 
// Default + AMPADefault + GABAaDefault + GABAbDefault + NMDADefault {}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[allow(dead_code)] 
// basic neurotransmitter type for unspecific general glutamate testing
pub enum NeurotransmitterType {
    Basic,
    AMPA,
    NMDA,
    GABAa,
    GABAb,
}

impl NeurotransmitterType {
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

pub trait NeurotransmitterKinetics: Clone {
    fn apply_t_change(&mut self, voltage: f64);
    fn get_t(&self) -> f64;
    fn set_t(&mut self, t: f64);
}

#[derive(Debug, Clone, Copy)]
pub struct DestexheNeurotransmitter {
    pub t_max: f64,
    pub t: f64,
    pub v_p: f64,
    pub k_p: f64,
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
impl_destexhe_neurotransmitter_default!(GABAbDefault2, gabab_default2, 0.5);

impl NeurotransmitterKinetics for DestexheNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f64) {
        self.t = self.t_max / (1. + (-(voltage - self.v_p) / self.k_p).exp());
    }

    fn get_t(&self) -> f64 {
        self.t
    }

    fn set_t(&mut self, t: f64) {
        self.t = t;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ApproximateNeurotransmitter {
    pub t_max: f64,
    pub t: f64,
    pub v_th: f64,
    pub clearance_constant: f64,
    pub dt: f64,
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
                    dt: 0.1,
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
impl_approximate_neurotransmitter_default!(GABAbDefault2, gabab_default2, 0.5);

fn heaviside(x: f64) -> f64 {
    if x > 0. {
        1.
    } else {
        0.
    }
}

impl NeurotransmitterKinetics for ApproximateNeurotransmitter {
    fn apply_t_change(&mut self, voltage: f64) {
        self.t += self.dt * -self.clearance_constant * self.t + (heaviside(voltage - self.v_th) * self.t_max);
        self.t = self.t_max.min(self.t.max(0.));
    }

    fn get_t(&self) -> f64 {
        self.t
    }

    fn set_t(&mut self, t: f64) {
        self.t = t;
    }
}

pub trait ReceptorKinetics: Default {
    fn apply_r_change(&mut self, t: f64);
    fn get_r(&self) -> f64;
    fn set_r(&mut self, r: f64);
}

#[derive(Debug, Clone, Copy)]
pub struct DestexheReceptor {
    pub r: f64,
    pub alpha: f64,
    pub beta: f64,
    pub dt: f64,
}

macro_rules! impl_destexhe_receptor_default {
    ($trait:ident, $method:ident, $alpha:expr, $beta:expr, $dt:expr) => {
        impl $trait for DestexheReceptor {
            fn $method() -> Self {
                DestexheReceptor {
                    r: 0.,
                    alpha: $alpha, // mM^-1 * ms^-1
                    beta: $beta, // ms^-1
                    dt: $dt,
                }
            }
        }
    };
}

impl ReceptorKinetics for DestexheReceptor {
    fn apply_r_change(&mut self, t: f64) {
        self.r += (self.alpha * t * (1. - self.r) - self.beta * self.r) * self.dt;
    }

    fn get_r(&self) -> f64 {
        self.r
    }

    fn set_r(&mut self, r: f64) {
        self.r = r;
    }
}

impl_destexhe_receptor_default!(Default, default, 1., 1., 0.1);
impl_destexhe_receptor_default!(AMPADefault, ampa_default, 1.1, 0.19, 0.1);
impl_destexhe_receptor_default!(GABAaDefault, gabaa_default, 5.0, 0.18, 0.1);
impl_destexhe_receptor_default!(GABAbDefault, gabab_default, 0.016, 0.0047, 0.1);
impl_destexhe_receptor_default!(GABAbDefault2, gabab_default2, 0.52, 0.0013, 0.1);
impl_destexhe_receptor_default!(NMDADefault, nmda_default, 0.072, 0.0066, 0.1);

// #[derive(Debug, Clone, Copy)]
// pub struct ApproximateReceptor {
//     pub r: f64,
// }

// impl Default for ApproximateReceptor {
//     fn default() -> Self {
//         ApproximateReceptor { r: 0. }
//     }
// }

// impl ReceptorKinetics for ApproximateReceptor {
//     fn apply_r_change(&mut self, t: f64) {
//         self.r = t;
//     }

//     fn get_r(&self) -> f64 {
//         self.r
//     }

    // fn set_r(&mut self, r: f64) {
    //     self.r = r;
    // }
// }

#[derive(Debug, Clone)]
pub enum IonotropicReceptorType {
    Basic(f64),
    AMPA(f64),
    GABAa(f64),
    GABAb(GABAbDissociation),
    NMDA(BV),
}

#[derive(Debug, Clone)]
pub struct LigandGatedChannel {
    pub g: f64,
    pub reversal: f64,
    pub receptor: DestexheReceptor,
    pub receptor_type: IonotropicReceptorType,
    pub current: f64,
}

impl Default for LigandGatedChannel {
    fn default() -> Self {
        LigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            receptor: DestexheReceptor::default(),
            receptor_type: IonotropicReceptorType::Basic(1.0),
            current: 0.,
        }
    }
}

impl AMPADefault for LigandGatedChannel {
    fn ampa_default() -> Self {
        LigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            receptor: DestexheReceptor::ampa_default(),
            receptor_type: IonotropicReceptorType::AMPA(1.0),
            current: 0.,
        }
    }
}

impl GABAaDefault for LigandGatedChannel {
    fn gabaa_default() -> Self {
        LigandGatedChannel {
            g: 1.2, // 1.2 nS
            reversal: -80., // -80 mV
            receptor: DestexheReceptor::gabaa_default(),
            receptor_type: IonotropicReceptorType::GABAa(1.0),
            current: 0.,
        }
    }
}

impl GABAbDefault for LigandGatedChannel {
    fn gabab_default() -> Self {
        LigandGatedChannel {
            g: 0.06, // 0.06 nS
            reversal: -95., // -95 mV
            receptor: DestexheReceptor::gabab_default(),
            receptor_type: IonotropicReceptorType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl GABAbDefault2 for LigandGatedChannel {
    fn gabab_default2() -> Self {
        LigandGatedChannel {
            g: 0.06, // 0.06 nS
            reversal: -95., // -95 mV
            receptor: DestexheReceptor::gabab_default2(),
            receptor_type: IonotropicReceptorType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl NMDADefault for LigandGatedChannel {
    fn nmda_default() -> Self {
        LigandGatedChannel {
            g: 0.6, // 0.6 nS
            reversal: 0., // 0.0 mV
            receptor: DestexheReceptor::nmda_default(),
            receptor_type: IonotropicReceptorType::NMDA(BV::default()),
            current: 0.,
        }
    }
}

pub trait NMDAWithBV {
    fn nmda_with_bv(bv: BV) -> Self;
}

impl NMDAWithBV for LigandGatedChannel {
    fn nmda_with_bv(bv: BV) -> Self {
        LigandGatedChannel {
            g: 0.6, // 0.6 nS
            reversal: 0., // 0.0 mV
            receptor: DestexheReceptor::nmda_default(),
            receptor_type: IonotropicReceptorType::NMDA(bv),
            current: 0.,
        }
    }
}

impl LigandGatedChannel {
    fn get_modifier(&mut self, voltage: f64) -> f64 {
        match &mut self.receptor_type {
            IonotropicReceptorType::AMPA(value) => *value,
            IonotropicReceptorType::GABAa(value) => *value,
            IonotropicReceptorType::GABAb(value) => {
                value.g += (value.k3 * self.receptor.get_r() - value.k4 * value.g) * value.dt;
                value.calculate_modifer() // G^N / (G^N + Kd)
            }, 
            IonotropicReceptorType::NMDA(value) => value.calculate_b(voltage),
            IonotropicReceptorType::Basic(value) => *value,
        }
    }

    pub fn calculate_current(&mut self, voltage: f64) -> f64 {
        let modifier = self.get_modifier(voltage);

        self.current = modifier * self.receptor.get_r() * self.g * (voltage - self.reversal);

        self.current
    }

    pub fn to_str(&self) -> &str {
        match self.receptor_type {
            IonotropicReceptorType::Basic(_) => "Basic",
            IonotropicReceptorType::AMPA(_) => "AMPA",
            IonotropicReceptorType::GABAa(_) => "GABAa",
            IonotropicReceptorType::GABAb(_) => "GABAb",
            IonotropicReceptorType::NMDA(_) => "NMDA",
        }
    }
}

#[derive(Clone, Debug)]
pub struct LigandGatedChannels { 
    pub ligand_gates: HashMap<NeurotransmitterType, LigandGatedChannel> 
}

impl Default for LigandGatedChannels {
    fn default() -> Self {
        LigandGatedChannels {
            ligand_gates: HashMap::new(),
        }
    }
}

impl LigandGatedChannels {
    pub fn len(&self) -> usize {
        self.ligand_gates.len()
    }

    // pub fn keys(&self) -> Keys<NeurotransmitterType, LigandGatedChannel> {
    //     self.ligand_gates.keys()
    // }

    pub fn values(&self) -> Values<NeurotransmitterType, LigandGatedChannel> {
        self.ligand_gates.values()
    }

    pub fn set_receptor_currents(&mut self, voltage: f64) {
        self.ligand_gates
            .values_mut()
            .for_each(|i| {
                i.calculate_current(voltage);
        });
    }

    pub fn get_receptor_currents(&self, dt: f64, c_m: f64) -> f64 {
        self.ligand_gates
            .values()
            .map(|i| i.current)
            .sum::<f64>() * (dt / c_m)
    }

    pub fn update_receptor_kinetics(&mut self, t_total: Option<&HashMap<NeurotransmitterType, f64>>) {
        match t_total {
            Some(t_hashmap) => {
                t_hashmap.iter()
                    .for_each(|(key, value)| {
                        if let Some(gate) = self.ligand_gates.get_mut(key) {
                            gate.receptor.apply_r_change(*value);
                        }
                    })
            },
            None => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct Neurotransmitters<T: NeurotransmitterKinetics> {
    pub neurotransmitters: HashMap<NeurotransmitterType, T>
}

pub type NeurotransmitterConcentrations = HashMap<NeurotransmitterType, f64>;

impl<T: NeurotransmitterKinetics> Default for Neurotransmitters<T> {
    fn default() -> Self {
        Neurotransmitters {
            neurotransmitters: HashMap::new(),
        }
    }
}

impl <T: NeurotransmitterKinetics> Neurotransmitters<T> {
    pub fn len(&self) -> usize {
        self.neurotransmitters.keys().len()
    }

    pub fn keys(&self) -> Keys<NeurotransmitterType, T> {
        self.neurotransmitters.keys()
    }

    pub fn values(&self) -> Values<NeurotransmitterType, T> {
        self.neurotransmitters.values()
    }

    pub fn values_mut(&mut self) -> ValuesMut<NeurotransmitterType, T> {
        self.neurotransmitters.values_mut()
    }

    pub fn get_concentrations(&self) -> NeurotransmitterConcentrations {
        self.neurotransmitters.iter()
            .map(|(neurotransmitter_type, neurotransmitter)| (*neurotransmitter_type, neurotransmitter.get_t()))
            .collect::<NeurotransmitterConcentrations>()
    }

    pub fn apply_t_changes(&mut self, voltage: f64) {
        self.neurotransmitters.values_mut()
            .for_each(|value| value.apply_t_change(voltage));
    }
}

pub fn weight_neurotransmitter_concentration(
    neurotransmitter_hashmap: &mut NeurotransmitterConcentrations, 
    weight: f64
) {
    neurotransmitter_hashmap.values_mut().for_each(|value| *value *= weight);
}

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
// must find way to modify based on glutmate binding
// channel activated only by glutamate binding
// maybe multiply by a weighting of open receptors
// should plot how it changes over time without any weighting

// percent of open receptors fraction is r, T is neurotrasnmitter concentration
// could vary Tmax to vary amount of nt conc
// dr/dt = alpha * T * (1 - r) - beta * r
// T = Tmax / (1 + (-(Vpre - Vp) / Kp).exp())

// I AMPA (or GABAa) = G AMPA (or GABAa) * (Vm - E AMPA (or GABAa))
// can also be modified with r

#[derive(Clone, Copy, Debug)]
pub enum PotentiationType {
    Excitatory,
    Inhibitory,
}

impl PotentiationType {
    pub fn from_str(string: &str) -> Result<PotentiationType> {
        match string.to_ascii_lowercase().as_str() {
            "excitatory" => Ok(PotentiationType::Excitatory),
            "inhibitory" => Ok(PotentiationType::Inhibitory),
            _ => Err(Error::new(ErrorKind::InvalidInput, "Unknown potentiation type")),
        }
    }

    pub fn weighted_random_type(prob: f64) -> PotentiationType {
        if rand::thread_rng().gen_range(0.0..=1.0) <= prob {
            PotentiationType::Excitatory
        } else {
            PotentiationType::Inhibitory
        }
    }
}


#[derive(Debug, Clone)]
pub struct BayesianParameters {
    pub mean: f64,
    pub std: f64,
    pub max: f64,
    pub min: f64,
}

impl Default for BayesianParameters {
    fn default() -> Self {
        BayesianParameters { 
            mean: 1.0, // center of norm distr
            std: 0.0, // std of norm distr
            max: 2.0, // maximum cutoff for norm distr
            min: 0.0, // minimum cutoff for norm distr
        }
    }
}

#[derive(Clone, Debug)]
pub struct STDPParameters {
    pub a_plus: f64, // postitive stdp modifier 
    pub a_minus: f64, // negative stdp modifier 
    pub tau_plus: f64, // postitive stdp decay modifier 
    pub tau_minus: f64, // negative stdp decay modifier 
}

impl Default for STDPParameters {
    fn default() -> Self {
        STDPParameters { 
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 45., 
            tau_minus: 45., 
        }
    }
}

pub trait CurrentVoltage {
    fn get_current_voltage(&self) -> f64;
}

pub trait GapConductance {
    fn get_gap_conductance(&self) -> f64;
}

pub trait Potentiation {
    fn get_potentiation_type(&self) -> PotentiationType;
}

pub trait BayesianFactor {
    fn get_bayesian_factor(&self) -> f64;
}

pub trait LastFiringTime {
    fn get_last_firing_time(&self) -> Option<usize>;
    fn set_last_firing_time(&mut self, timestep: Option<usize>);
}

pub trait STDP: LastFiringTime {
    fn get_stdp_params(&self) -> &STDPParameters;
}

macro_rules! impl_current_voltage_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> CurrentVoltage for $struct<T> {
            fn get_current_voltage(&self) -> f64 {
                self.current_voltage
            }
        }
    };
}

pub(crate) use impl_current_voltage_with_neurotransmitter;

macro_rules! impl_gap_conductance_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> GapConductance for $struct<T> {
            fn get_gap_conductance(&self) -> f64 {
                self.gap_conductance
            }
        }
    };
}

pub(crate) use impl_gap_conductance_with_neurotransmitter;

macro_rules! impl_potentiation_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> Potentiation for $struct<T> {
            fn get_potentiation_type(&self) -> PotentiationType {
                self.potentiation_type
            }
        }
    };
}

pub(crate) use impl_potentiation_with_neurotransmitter;

macro_rules! impl_bayesian_factor_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> BayesianFactor for $struct<T> {
            fn get_bayesian_factor(&self) -> f64 {
                crate::distribution::limited_distr(
                    self.bayesian_params.mean, 
                    self.bayesian_params.std, 
                    self.bayesian_params.min, 
                    self.bayesian_params.max,
                )
            }
        }
    };
}

pub(crate) use impl_bayesian_factor_with_neurotransmitter;

macro_rules! impl_last_firing_time_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> LastFiringTime for $struct<T> {
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.last_firing_time = timestep;
            }
        
            fn get_last_firing_time(&self) -> Option<usize> {
                self.last_firing_time
            }
        }
    }
}

pub(crate) use impl_last_firing_time_with_neurotransmitter;

macro_rules! impl_stdp_with_neurotransmitter {
    ($struct:ident) => {
        impl<T: NeurotransmitterKinetics> STDP for $struct<T> {        
            fn get_stdp_params(&self) -> &STDPParameters {
                &self.stdp_params
            }
        }
    };
}

pub(crate) use impl_stdp_with_neurotransmitter;

macro_rules! impl_necessary_iterate_and_spike_traits {
    ($name:ident) => {
        impl_current_voltage_with_neurotransmitter!($name);
        impl_gap_conductance_with_neurotransmitter!($name);
        impl_potentiation_with_neurotransmitter!($name);
        impl_bayesian_factor_with_neurotransmitter!($name);
        impl_last_firing_time_with_neurotransmitter!($name);
        impl_stdp_with_neurotransmitter!($name);
    }
}

pub(crate) use impl_necessary_iterate_and_spike_traits;

pub trait IterateAndSpike: 
Clone + CurrentVoltage + GapConductance + Potentiation + BayesianFactor + STDP {
    type T: NeurotransmitterKinetics;
    fn iterate_and_spike(&mut self, input_current: f64) -> bool;
    fn get_ligand_gates(&self) -> &LigandGatedChannels;
    fn get_neurotransmitters(&self) -> &Neurotransmitters<Self::T>;
    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64>;
    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>,
    ) -> bool;
}
