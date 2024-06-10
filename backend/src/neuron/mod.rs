use std::{
    f64::consts::E, 
    fs::File, 
    io::{BufWriter, Error, ErrorKind, Result, Write},
    collections::{HashMap, HashSet},
    ops::Sub,
};
use rand::Rng;
use crate::distribution;
use distribution::limited_distr;
// pub mod integrate_and_fire;
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, Potentiation, BayesianFactor, LastFiringTime, STDP,
    IterateAndSpike, BayesianParameters, STDPParameters, PotentiationType,
    Neurotransmitters, NeurotransmitterType, NeurotransmitterKinetics, 
    ApproximateNeurotransmitter, weight_neurotransmitter_concentration,
    LigandGatedChannels,
    impl_current_voltage_with_neurotransmitter,
    impl_gap_conductance_with_neurotransmitter,
    impl_potentiation_with_neurotransmitter,
    impl_bayesian_factor_with_neurotransmitter,
    impl_last_firing_time_with_neurotransmitter,
    impl_stdp_with_neurotransmitter,
};
use crate::graph;
use graph::GraphFunctionality;


pub trait IzhikevichDefault {
    fn izhikevich_default() -> Self;
}

// weight_bayesian_params: BayesianParameters {
//     mean: 3.5,
//     std: 1.0,
//     min: 1.75,
//     max: 1.75,
// },

#[derive(Clone, Copy, Debug)]
pub enum IFType {
    Basic,
    Adaptive,
    AdaptiveExponential,
    Izhikevich,
    IzhikevichLeaky,
}

impl IFType {
    pub fn from_str(string: &str) -> Result<IFType> {
        let output = match string.to_ascii_lowercase().as_str() {
            "basic" => { IFType::Basic },
            "adaptive" => { IFType::Adaptive },
            "adaptive exponential" => { IFType::AdaptiveExponential },
            "izhikevich" | "adaptive quadratic" => { IFType::Izhikevich },
            "leaky izhikevich" | "leaky adaptive quadratic" => { IFType::IzhikevichLeaky }
            _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown string")); },
        };

        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub struct IntegrateAndFireCell<T: NeurotransmitterKinetics> {
    pub if_type: IFType,
    pub current_voltage: f64, // membrane potential
    pub refractory_count: f64, // keeping track of refractory period
    pub leak_constant: f64, // leak constant gene
    pub integration_constant: f64, // integration constant gene
    pub gap_conductance: f64, // condutance between synapses
    pub potentiation_type: PotentiationType,
    pub w_value: f64, // adaptive value 
    pub last_firing_time: Option<usize>,
    pub alpha: f64, // arbitrary value (controls speed in izhikevich)
    pub beta: f64, // arbitrary value (controls sensitivity to w in izhikevich)
    pub c: f64, // after spike reset value for voltage
    pub d: f64, // after spike reset value for w
    pub v_th: f64, // voltage threshold
    pub v_reset: f64, // voltage reset
    pub tau_m: f64, // membrane time constant
    pub c_m: f64, // membrane capacitance
    pub g_l: f64, // leak constant
    pub v_init: f64, // initial voltage
    pub e_l: f64, // leak reversal potential
    pub tref: f64, // total refractory period
    pub w_init: f64, // initial adaptive value
    pub slope_factor: f64, // slope factor in exponential adaptive neuron
    pub dt: f64, // time step
    pub bayesian_params: BayesianParameters, // bayesian parameters
    pub stdp_params: STDPParameters, // stdp parameters
    pub synaptic_neurotransmitters: Neurotransmitters<T>, // neurotransmitters
    pub ligand_gates: LigandGatedChannels, // ligand gates
}

impl<T: NeurotransmitterKinetics> Default for IntegrateAndFireCell<T> {
    fn default() -> Self {
        IntegrateAndFireCell {
            if_type: IFType::Basic,
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            potentiation_type: PotentiationType::Excitatory,
            w_value: 0.,
            last_firing_time: None,
            alpha: 6.0,
            beta: 10.0,
            c: -55.0,
            d: 8.0,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            dt: 0.1, // simulation time step (ms)
            slope_factor: 1., // exponential time step (ms)
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics> IzhikevichDefault for IntegrateAndFireCell<T> {
    fn izhikevich_default() -> Self {
        IntegrateAndFireCell {
            if_type: IFType::Izhikevich,
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            potentiation_type: PotentiationType::Excitatory,
            w_value: 0.,
            last_firing_time: None,
            alpha: 0.02,
            beta: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            v_reset: -65., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -65., // initial potential (mV)
            e_l: -65., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            dt: 0.1, // simulation time step (ms)
            slope_factor: 1., // exponential time step (ms)
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl_current_voltage_with_neurotransmitter!(IntegrateAndFireCell);
impl_gap_conductance_with_neurotransmitter!(IntegrateAndFireCell);
impl_potentiation_with_neurotransmitter!(IntegrateAndFireCell);
impl_bayesian_factor_with_neurotransmitter!(IntegrateAndFireCell);
impl_last_firing_time_with_neurotransmitter!(IntegrateAndFireCell);
impl_stdp_with_neurotransmitter!(IntegrateAndFireCell);

impl<T: NeurotransmitterKinetics> IntegrateAndFireCell<T> {
    pub fn get_basic_dv_change(&self, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.integration_constant * (i / self.g_l))
        ) * (self.dt / self.tau_m);

        dv
    }

    pub fn basic_handle_spiking(&mut self) -> bool {
        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = self.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= self.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = self.v_reset;
            self.refractory_count = self.tref / self.dt
        }

        is_spiking
    }

    fn basic_iterate_and_spike(&mut self, i: f64) -> bool {
        let dv = self.get_basic_dv_change(i);
        self.current_voltage += dv;

        self.basic_handle_spiking()
    }

    pub fn adaptive_get_dw_change(&self) -> f64 {
        let dw = (
            self.alpha * (self.current_voltage - self.e_l) -
            self.w_value
        ) * (self.dt / self.tau_m);

        dw
    }

    pub fn adaptive_handle_spiking(&mut self) -> bool {
        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = self.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= self.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = self.v_reset;
            self.w_value += self.beta;
            self.refractory_count = self.tref / self.dt
        }

        is_spiking
    }

    pub fn adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m);

        dv
    }

    pub fn adaptive_iterate_and_spike(&mut self, i: f64) -> bool {
        let dv = self.adaptive_get_dv_change(i);
        let dw = self.adaptive_get_dw_change();

        self.current_voltage += dv;
        self.w_value += dw;

        self.adaptive_handle_spiking()
    }

    pub fn exp_adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.slope_factor * ((self.current_voltage - self.v_th) / self.slope_factor).exp()) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m);

        dv
    }

    pub fn exp_adaptive_iterate_and_spike(&mut self, i: f64) -> bool {
        let dv = self.exp_adaptive_get_dv_change(i);
        let dw = self.adaptive_get_dw_change();

        self.current_voltage += dv;
        self.w_value += dw;

        self.adaptive_handle_spiking()
    }

    pub fn izhikevich_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (self.dt / self.c_m);

        dv
    }

    pub fn izhikevich_get_dw_change(&self) -> f64 {
        let dw = (
            self.alpha * (self.beta * self.current_voltage - self.w_value)
        ) * self.dt;

        dw
    }

    pub fn izhikevich_handle_spiking(&mut self) -> bool {
        let mut is_spiking = false;

        if self.current_voltage >= self.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = self.c;
            self.w_value += self.d;
        }

        is_spiking
    }

    pub fn izhikevich_iterate_and_spike(&mut self, i: f64) -> bool {
        let dv = self.izhikevich_get_dv_change(i);
        let dw = self.izhikevich_get_dw_change();

        self.current_voltage += dv;
        self.w_value += dw;

        self.izhikevich_handle_spiking()
    }

    pub fn izhikevich_leaky_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - 
            self.w_value * (self.current_voltage - self.e_l) + i
        ) * (self.dt / self.c_m);

        dv
    }

    pub fn izhikevich_leaky_iterate_and_spike(&mut self, i: f64) -> bool {
        let dv = self.izhikevich_leaky_get_dv_change(i);
        let dw = self.izhikevich_get_dw_change();

        self.current_voltage += dv;
        self.w_value += dw;

        self.izhikevich_handle_spiking()
    }

    pub fn bayesian_iterate_and_spike(&mut self, i: f64, bayesian: bool) -> bool {
        if bayesian {
            self.iterate_and_spike(i * limited_distr(self.bayesian_params.mean, self.bayesian_params.std, 0., 1.))
        } else {
            self.iterate_and_spike(i)
        }
    }

    pub fn run_static_input(
        &mut self, 
        i: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        
        match self.if_type {
            IFType::Basic => {
                writeln!(file, "voltage").expect("Unable to write to file");
                writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");

                for _ in 0..iterations {
                    let _is_spiking = self.bayesian_iterate_and_spike(i, bayesian);
        
                    writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
                }
            },
            IFType::Adaptive | IFType::AdaptiveExponential | IFType::Izhikevich | IFType::IzhikevichLeaky => {
                writeln!(file, "voltage,w").expect("Unable to write to file");
                writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");

                for _ in 0..iterations {
                    let _is_spiking = self.bayesian_iterate_and_spike(i, bayesian);
        
                    writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
                }
            }
        }
    }
}

impl<T: NeurotransmitterKinetics> IterateAndSpike for IntegrateAndFireCell<T> {
    type T = T;

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let is_spiking = match self.if_type {
            IFType::Basic => {
                self.basic_iterate_and_spike(input_current)
            },
            IFType::Adaptive => {
                self.adaptive_iterate_and_spike(input_current)
            },
            IFType::AdaptiveExponential => {
                self.exp_adaptive_iterate_and_spike(input_current)
            },
            IFType::Izhikevich => {
                self.izhikevich_iterate_and_spike(input_current)
            },
            IFType::IzhikevichLeaky => {
                self.izhikevich_leaky_iterate_and_spike(input_current)
            }
        };

        // t changes should be applied after dv is added and before spiking is handled
        // in if split
        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        is_spiking
    }

    fn get_ligand_gates(&self) -> &LigandGatedChannels {
        &self.ligand_gates
    }

    fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
        &self.synaptic_neurotransmitters
    }

    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);

        self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.iterate_and_spike(input_current)
    }
}

pub type CellGrid<T> = Vec<Vec<T>>;

// ** IMPLEMENT TESTING FOR THIS **
// ** IMPLEMENT MULTICOMPARTMENT MODEL BASED ON THIS **
// REFER TO destexhe model of neuronal modeling

// https://webpages.uidaho.edu/rwells/techdocs/Biological%20Signal%20Processing/Chapter%2004%20The%20Biological%20Neuron.pdf

// https://www.nature.com/articles/356441a0.pdf : calcium currents paper
// https://github.com/ModelDBRepository/151460/blob/master/CaT.mod // low threshold calcium current
// https://modeldb.science/279?tab=1 // low threshold calcium current (thalamic)
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod // high threshold calcium current (l type)
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9373714/ // assume [Ca2+]in,inf is initial [Ca2+] value

// l-type ca2+ channel, ca1.2
#[derive(Clone, Copy)]
pub struct HighThresholdCalciumChannel {
    current: f64,
    z: f64,
    f: f64,
    // r: f64,
    // temp: f64,
    ca_in: f64,
    ca_in_equilibrium: f64,
    ca_out: f64,
    // permeability: f64,
    max_permeability: f64,
    d: f64,
    kt: f64,
    kd: f64,
    tr: f64,
    k: f64,
    p: f64,
    v_th: f64,
    s: f64,
    m_ca: f64,
    alpha: f64,
    beta: f64, 
}

impl Default for HighThresholdCalciumChannel {
    fn default() -> Self {
        HighThresholdCalciumChannel {
            current: 0.,
            z: 2.,
            f: 96489., // C/mol
            // r: 8.31, // J/Kmol
            // temp: 35., // degrees c
            ca_in: 0.001, // mM
            ca_in_equilibrium: 0.001, // mM
            ca_out: 5., // mM
            // permeability: 0.,
            max_permeability: 5.36e-6,
            d: 0.1, // um
            kt: 1e-4, // mM / ms
            kd: 1e-4, // mM
            tr: 43., // ms
            k: 1000.,
            p: 0.02,
            v_th: (9. * 297.) / (2. * E),
            s: 1.,
            m_ca: 0.,
            alpha: 0.,
            beta: 0.,
        }
    }
}

impl HighThresholdCalciumChannel {
    // m^x * n^y
    // x and y here probably refer to 3 and 4
    // fn update_permeability(&mut self, m_state: f64, n_state: f64) {
    //     self.permeability = self.max_permeability * m_state * n_state;
    // }

    fn update_ca_in(&mut self, dt: f64) {
        let term1 = self.k * (-self.current / (2. * self.f * self.d));
        let term2 = self.p * ((self.kt * self.ca_in) / (self.ca_in + self.kd));
        let term3 = (self.ca_in_equilibrium - self.ca_in) / self.tr;
        self.ca_in += (term1 + term2 + term3) * dt;
    }

    fn update_m_ca(&mut self, voltage: f64) {
        self.alpha += 1.6 / (1. + (-0.072 * (voltage - 5.)).exp());
        self.beta += (0.02 * (voltage - 1.31)) / (((voltage - 1.31) / 5.36).exp() - 1.);
        self.m_ca = self.alpha / (self.alpha + self.beta);
    }

    fn get_ca_current(&self, voltage: f64) -> f64 {
        let term1 = self.m_ca.powf(2.) * self.max_permeability * self.s;
        let term2 = (self.z * self.f) / self.v_th;
        let term3 = voltage / self.v_th.exp();
        let term4 = self.ca_in_equilibrium * term3 - self.ca_out;
        let term5 = term3 - 1.;

        term1 * term2 * (term4 / term5) * voltage
    }

    fn get_ca_current_and_update(&mut self, voltage: f64, dt: f64) -> f64 {
        self.update_ca_in(dt);
        self.update_m_ca(voltage);
        self.current = self.get_ca_current(voltage);

        self.current
    }
}

#[derive(Clone, Copy)]
pub struct HighVoltageActivatedCalciumChannel {
    m: f64,
    m_a: f64,
    m_b: f64,
    h: f64,
    h_a: f64,
    h_b: f64,
    // ca_in: f64,
    // ca_out: f64,
    gca_bar: f64,
    ca_rev: f64,
    current: f64,
}

impl Default for HighVoltageActivatedCalciumChannel {
    fn default() -> Self {
        let r: f64 = 8.314; // joules * kelvin ^ -1 * mol ^ -1 // universal gas constant
        let faraday: f64 = 96485.; // coulombs per mole // faraday constant
        let celsius: f64 = 36.; // degrees c
        let ca_in: f64 = 0.00024; // mM
        let ca_out: f64 = 2.; // mM
        let ca_rev: f64 = 1e3 * ((r * (celsius + 273.15)) / (2. * faraday)) * (ca_out / ca_in).ln(); // nernst equation

        HighVoltageActivatedCalciumChannel {
            m: 0.,
            m_a: 0.,
            m_b: 0.,
            h: 0.,
            h_a: 0.,
            h_b: 0.,
            // ca_in: ca_in,
            // ca_out: ca_out,
            gca_bar: 1e-4,
            ca_rev: ca_rev,
            current: 0.,
        }
    }
}

// https://github.com/ModelDBRepository/121060/blob/master/chan_CaL12.mod
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod
impl HighVoltageActivatedCalciumChannel {
    fn update_m(&mut self, voltage: f64) {
        self.m_a = 0.055 * (-27. - voltage) / (((-27. - voltage) / 3.8).exp() - 1.);
        self.m_b = 0.94 * ((-75. - voltage) / 17.).exp();
    }

    fn update_h(&mut self, voltage: f64) {
        self.h_a = 0.000457 * ((-13. - voltage) / 50.).exp();
        self.h_b = 0.0065 / (((-15. - voltage) / 28.).exp() + 1.);
    }

    fn initialize_m_and_h(&mut self, voltage: f64) {
        self.update_m(voltage);
        self.update_h(voltage);

        self.m = self.m_a / (self.m_a + self.m_b);
        self.h = self.h_a / (self.h_a + self.h_b);
    } 

    fn update_m_and_h_states(&mut self, voltage: f64, dt: f64) {
        self.update_m(voltage);
        self.update_h(voltage);

        self.m += (self.m_a * (1. - self.m) - (self.m_b * self.m)) * dt;
        self.h += (self.h_a * (1. - self.h) - (self.h_b * self.h)) * dt;
    }

    fn get_ca_and_update_current(&mut self, voltage: f64, dt: f64) -> f64 {
        self.update_m_and_h_states(voltage, dt);
        self.current = self.gca_bar * self.m.powf(2.) * self.h * (voltage - self.ca_rev);

        // if this isnt working gas constant might be wrong
        // try to determine where it is reaching inf
        // println!("m: {}, h: {}", self.m, self.h);

        self.current
    }
}

// can look at this
// https://github.com/JoErNanO/brianmodel/blob/master/brianmodel/neuron/ioniccurrent/ioniccurrentcal.py
#[derive(Clone, Copy)]
pub enum AdditionalGates {
    LTypeCa(HighThresholdCalciumChannel),
    HVACa(HighVoltageActivatedCalciumChannel), // https://neuronaldynamics.epfl.ch/online/Ch2.S3.html // https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/8229187/
    // OscillatingCa(OscillatingCalciumChannel),
    // PotassiumRectifying(KRectifierChannel),
}

impl AdditionalGates {
    pub fn initialize(&mut self, voltage: f64) {
        match self {
            AdditionalGates::LTypeCa(_) => {},
            AdditionalGates::HVACa(channel) => channel.initialize_m_and_h(voltage),
        }
    }

    pub fn get_and_update_current(&mut self, voltage: f64, dt: f64) -> f64 {
        match self {
            AdditionalGates::LTypeCa(channel) => channel.get_ca_current_and_update(voltage, dt),
            AdditionalGates::HVACa(channel) => channel.get_ca_and_update_current(voltage, dt),
        }
    }

    pub fn get_current(&self) -> f64 {
        match &self {
            AdditionalGates::LTypeCa(channel) => channel.current,
            AdditionalGates::HVACa(channel) => channel.current,
        }
    }

    pub fn to_str(&self) -> &str {
        match &self {
            AdditionalGates::LTypeCa(_) => "LTypeCa",
            AdditionalGates::HVACa(_) => "HVA LTypeCa",
        }
    }
}

// multicomparment stuff, refer to dopamine modeling paper as well
// https://github.com/antgon/msn-model/blob/main/msn/cell.py 
// https://github.com/jrieke/NeuroSim
// MULTICOMPARTMENT EXPLAINED
// https://neuronaldynamics.epfl.ch/online/Ch3.S2.html
// pub struct Soma {

// }

// pub struct Dendrite {

// }

#[derive(Clone, Copy)]
pub struct Gate {
    pub alpha: f64,
    pub beta: f64,
    pub state: f64,
}

impl Gate {
    pub fn init_state(&mut self) {
        self.state = self.alpha / (self.alpha + self.beta);
    }

    pub fn update(&mut self, dt: f64) {
        let alpha_state: f64 = self.alpha * (1. - self.state);
        let beta_state: f64 = self.beta * self.state;
        self.state += dt * (alpha_state - beta_state);
    }
}

#[derive(Clone)]
pub struct HodgkinHuxleyCell<T: NeurotransmitterKinetics> {
    pub current_voltage: f64,
    pub gap_conductance: f64,
    pub potentiation_type: PotentiationType,
    pub dt: f64,
    pub c_m: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_k_leak: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_k_leak: f64,
    pub m: Gate,
    pub n: Gate,
    pub h: Gate,
    pub v_th: f64,
    pub last_firing_time: Option<usize>,
    pub was_increasing: bool,
    pub is_spiking: bool,
    pub additional_gates: Vec<AdditionalGates>,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels,
    pub bayesian_params: BayesianParameters,
    pub stdp_params: STDPParameters,
}

impl_current_voltage_with_neurotransmitter!(HodgkinHuxleyCell);
impl_gap_conductance_with_neurotransmitter!(HodgkinHuxleyCell);
impl_potentiation_with_neurotransmitter!(HodgkinHuxleyCell);
impl_bayesian_factor_with_neurotransmitter!(HodgkinHuxleyCell);
impl_last_firing_time_with_neurotransmitter!(HodgkinHuxleyCell);
impl_stdp_with_neurotransmitter!(HodgkinHuxleyCell);

impl<T: NeurotransmitterKinetics> Default for HodgkinHuxleyCell<T> {
    fn default() -> Self {
        let default_gate = Gate {
            alpha: 0.,
            beta: 0.,
            state: 0.,
        };

        HodgkinHuxleyCell { 
            current_voltage: 0.,
            gap_conductance: 7.,
            potentiation_type: PotentiationType::Excitatory,
            dt: 0.1,
            c_m: 1., 
            e_na: 115., 
            e_k: -12., 
            e_k_leak: 10.6, 
            g_na: 120., 
            g_k: 36., 
            g_k_leak: 0.3, 
            m: default_gate.clone(), 
            n: default_gate.clone(), 
            h: default_gate,  
            v_th: 60.,
            last_firing_time: None,
            is_spiking: false,
            was_increasing: false,
            synaptic_neurotransmitters: Neurotransmitters::default(), 
            ligand_gates: LigandGatedChannels::default(),
            additional_gates: vec![],
            bayesian_params: BayesianParameters::default(),
            stdp_params: STDPParameters::default(),
        }
    }
}

// find peaks of hodgkin huxley
// result starts at index 1 of input list
pub fn diff<T: Sub<Output = T> + Copy>(x: &Vec<T>) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i-1])
        .collect()
}

pub fn find_peaks(voltages: &Vec<f64>, tolerance: f64) -> Vec<usize> {
    let first_diff: Vec<f64> = diff(&voltages);
    let second_diff: Vec<f64> = diff(&first_diff);

    let local_optima = first_diff.iter()
        .enumerate()
        .filter(|(_, i)| i.abs() <= tolerance)
        .map(|(n, i)| (n, *i))
        .collect::<Vec<(usize, f64)>>();

    let local_maxima = local_optima.iter()
        .map(|(n, i)| (*n, *i))
        .filter(|(n, _)| *n < second_diff.len() - 1 && second_diff[n+1] < 0.)
        .collect::<Vec<(usize, f64)>>();

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
impl<T: NeurotransmitterKinetics> HodgkinHuxleyCell<T> {
    pub fn update_gate_time_constants(&mut self, voltage: f64) {
        self.n.alpha = 0.01 * (10. - voltage) / (((10. - voltage) / 10.).exp() - 1.);
        self.n.beta = 0.125 * (-voltage / 80.).exp();
        self.m.alpha = 0.1 * ((25. - voltage) / (((25. - voltage) / 10.).exp() - 1.));
        self.m.beta = 4. * (-voltage / 18.).exp();
        self.h.alpha = 0.07 * (-voltage / 20.).exp();
        self.h.beta = 1. / (((30. - voltage) / 10.).exp() + 1.);
    }

    pub fn initialize_parameters(&mut self, starting_voltage: f64) {
        self.current_voltage = starting_voltage;
        self.update_gate_time_constants(starting_voltage);
        self.m.init_state();
        self.n.init_state();
        self.h.init_state();

        self.additional_gates.iter_mut()
            .for_each(|i| i.initialize(starting_voltage));
    }

    pub fn update_cell_voltage(&mut self, input_current: f64) {
        let i_na = self.m.state.powf(3.) * self.g_na * self.h.state * (self.current_voltage - self.e_na);
        let i_k = self.n.state.powf(4.) * self.g_k * (self.current_voltage - self.e_k);
        let i_k_leak = self.g_k_leak * (self.current_voltage - self.e_k_leak);

        let i_ligand_gates = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        let i_additional_gates = self.additional_gates
            .iter_mut()
            .map(|i| 
                i.get_and_update_current(self.current_voltage, self.dt)
            ) 
            .sum::<f64>();

        let i_sum = input_current - (i_na + i_k + i_k_leak) + i_ligand_gates + i_additional_gates;
        self.current_voltage += self.dt * i_sum / self.c_m;
    }

    pub fn update_neurotransmitters(&mut self) {
        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
    }

    pub fn update_receptors(
        &mut self, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>
    ) {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);
    }

    pub fn update_gate_states(&mut self) {
        self.m.update(self.dt);
        self.n.update(self.dt);
        self.h.update(self.dt);
    }

    pub fn iterate(&mut self, input: f64) {
        self.update_gate_time_constants(self.current_voltage);
        self.update_cell_voltage(input);
        self.update_gate_states();
        self.update_neurotransmitters();
    }

    pub fn iterate_with_neurotransmitter(
        &mut self, 
        input: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>
    ) {
        self.update_receptors(t_total);
        self.iterate(input);
    }

    pub fn run_static_input(
        &mut self, 
        input: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str, 
        full: bool,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        if !full {
            writeln!(file, "voltage").expect("Unable to write to file");
            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        } else {
            write!(file, "voltage,m,n,h").expect("Unable to write to file");
            writeln!(
                file, 
                ",{}",
                self.additional_gates.iter()
                    .map(|x| x.to_str())
                    .collect::<Vec<&str>>()
                    .join(",")
            ).expect("Unable to write to file");
            write!(file, "{}, {}, {}, {}", 
                self.current_voltage, 
                self.m.state, 
                self.n.state, 
                self.h.state,
            ).expect("Unable to write to file");
            writeln!(
                file, 
                ", {}",
                self.additional_gates.iter()
                    .map(|x| x.get_current().to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            ).expect("Unable to write to file");
        }

        self.initialize_parameters(self.current_voltage);
        
        for _ in 0..iterations {
            if bayesian {
                self.iterate(
                    input * limited_distr(
                        self.bayesian_params.mean, 
                        self.bayesian_params.std, 
                        self.bayesian_params.min, 
                        self.bayesian_params.max,
                    )
                );
            } else {
                self.iterate(input);
            }

            if !full {
                writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
            } else {
                write!(file, "{}, {}, {}, {}", 
                    self.current_voltage, 
                    self.m.state, 
                    self.n.state, 
                    self.h.state,
                ).expect("Unable to write to file");
                writeln!(
                    file, 
                    ", {}",
                    self.additional_gates.iter()
                        .map(|x| x.get_current().to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                ).expect("Unable to write to file");
            }
        }
    }

    pub fn peaks_test(
        &mut self, 
        input: f64, 
        bayesian: bool, 
        iterations: usize, 
        tolerance: f64,
        filename: &str, 
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        
        let mut voltages: Vec<f64> = vec![self.current_voltage];

        for _ in 0..iterations {
            if bayesian {
                let bayesian_factor = limited_distr(
                    self.bayesian_params.mean, 
                    self.bayesian_params.std, 
                    self.bayesian_params.min, 
                    self.bayesian_params.max,
                );
                let bayesian_input = input * bayesian_factor;

                self.iterate(bayesian_input);
            } else {
                self.iterate(input);
            }

            voltages.push(self.current_voltage);
        }

        let peaks = find_peaks(&voltages, tolerance);

        writeln!(file, "voltages,peak").expect("Could not write to file");
        for (n, i) in voltages.iter().enumerate() {
            let is_peak: &str = if peaks.contains(&n) {
                "true"
            } else {
                "false"
            };

            writeln!(file, "{},{}", i, is_peak).expect("Could not write to file");
        }
    }
}

impl<T: NeurotransmitterKinetics> IterateAndSpike for HodgkinHuxleyCell<T> {
    type T = T;

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate(input_current);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed  && self.was_increasing && !increasing_right_now;
        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }

    fn get_ligand_gates(&self) -> &LigandGatedChannels {
        &self.ligand_gates
    }

    fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
        &self.synaptic_neurotransmitters
    }

    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>,
    ) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate_with_neurotransmitter(input_current, t_total);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed  && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }
}

pub fn gap_junction<T: CurrentVoltage, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    postsynaptic_neuron.get_gap_conductance() * 
    (presynaptic_neuron.get_current_voltage() - postsynaptic_neuron.get_current_voltage())
}

pub fn signed_gap_junction<T: CurrentVoltage + Potentiation, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    sign * gap_junction(presynaptic_neuron, postsynaptic_neuron)
}

pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    do_receptor_kinetics: bool,
    bayesian: bool,
    input_current: f64,
) {
    let (t_total, post_current, input_current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let input_current = input_current * pre_bayesian_factor;

        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    } else {
        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    };

    let _pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);

    let _post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        t_total.as_ref(),
    );
}

pub enum DiscreteNeuronState {
    Active,
    Inactive,
}

pub struct DiscreteNeuron {
    pub state: DiscreteNeuronState
}

impl Default for DiscreteNeuron {
    fn default() -> Self {
        DiscreteNeuron { state: DiscreteNeuronState::Inactive }
    }
}

impl DiscreteNeuron {
    fn update(&mut self, input: f64) {
        match input > 0. {
            true => self.state = DiscreteNeuronState::Active,
            false => self.state = DiscreteNeuronState::Inactive,
        }
    }

    fn state_to_numeric(&self) -> f64 {
        match &self.state {
            DiscreteNeuronState::Active => 1.,
            DiscreteNeuronState::Inactive => -1.,
        }
    }
}

fn outer_product(a: &Vec<isize>, b: &Vec<isize>) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in a {
        let mut vector: Vec<isize> = Vec::new();
        for j in b {
            vector.push(i * j);
        }

        output.push(vector);
    }

    output
}

fn first_dimensional_index_to_position(i: usize, num_cols: usize) -> (usize, usize) {
    ((i / num_cols), (i % num_cols))
}

pub fn generate_hopfield_network<T: GraphFunctionality + Default>(
    num_rows: usize, 
    num_cols: usize, 
    data: &Vec<Vec<Vec<isize>>>
) -> Result<T> {
    let mut weights = T::default();

    for i in 0..num_rows {
        for j in 0..num_cols {
            weights.add_vertex((i, j));
        }
    }

    for pattern in data {
        let flattened_pattern: Vec<isize> = pattern.iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        let weight_changes = outer_product(&flattened_pattern, &flattened_pattern);

        for (i, weight_vec) in weight_changes.iter().enumerate() {
            for (j, value) in weight_vec.iter().enumerate() {
                let coming = first_dimensional_index_to_position(i, num_cols);
                let going = first_dimensional_index_to_position(j, num_cols);

                //   1 2 3 ...
                // 1 . . .
                // 2 . . .
                // 3 . . .
                // ...
                
                //       (0, 0) (0, 1) (0, 2) ...
                // (0, 0)   .      .      .
                // (0, 1)   .      .      .
                // (0, 2)   .      .      .
                // ...

                if coming == going {
                    weights.edit_weight(&coming, &going, None)?;
                    continue;
                }

                let current_weight = match weights.lookup_weight(&coming, &going)? {
                    Some(w) => w,
                    None => 0.
                };

                weights.edit_weight(&coming, &going, Some(current_weight + *value as f64))?;
            }
        }  
    } 

    Ok(weights)
}

pub fn input_pattern_into_grid(cell_grid: &mut Vec<Vec<DiscreteNeuron>>, pattern: Vec<Vec<isize>>) {
    for (i, pattern_vec) in pattern.iter().enumerate() {
        for (j, value) in pattern_vec.iter().enumerate() {
            cell_grid[i][j].update(*value as f64);
        }
    }
}

pub fn iterate_hopfield_network<T: GraphFunctionality>(
    cell_grid: &mut Vec<Vec<DiscreteNeuron>>, 
    weights: &T, 
) -> Result<()> {
    for i in 0..cell_grid.len() {
        for j in 0..cell_grid[0].len() {
            let input_positions = weights.get_incoming_connections(&(i, j))?;

            // if there is problem with convergence it is likely this calculation
            let input_value: f64 = input_positions.iter()
                .map(|(pos_i, pos_j)| 
                    weights.lookup_weight(&(*pos_i, *pos_j), &(i, j)).unwrap().unwrap() 
                    * cell_grid[*pos_i][*pos_j].state_to_numeric()
                )
                .sum();

            cell_grid[i][j].update(input_value);
        }
    }

    Ok(())
}

pub fn convert_hopfield_network(cell_grid: &Vec<Vec<DiscreteNeuron>>) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in cell_grid.iter() {
        let mut output_vec: Vec<isize> = Vec::new();
        for j in i.iter() {
            output_vec.push(j.state_to_numeric() as isize);
        }

        output.push(output_vec);
    }

    output
}

pub fn distort_pattern(pattern: &Vec<Vec<isize>>, noise_level: f64) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in pattern.iter() {
        let mut output_vec: Vec<isize> = Vec::new();
        for j in i.iter() {
            if rand::thread_rng().gen_range(0.0..=1.0) <= noise_level {
                if *j > 0 {
                    output_vec.push(-1);
                } else {
                    output_vec.push(1);
                }
            } else {
                output_vec.push(*j)
            }
        }

        output.push(output_vec);
    }

    output
}

// could try random turing patterns as well
// pub fn generate_random_patterns(
//     num_rows: usize, 
//     num_cols: usize, 
//     num_patterns: usize, 
//     noise_level: f64
// ) -> Vec<Vec<Vec<isize>>> {
//     let base_pattern = (0..num_rows).map(|_| {
//         (0..num_cols)
//             .map(|_| {
//                 -1
//             })
//             .collect::<Vec<isize>>()
//     })
//     .collect::<Vec<Vec<isize>>>();

//     (0..num_patterns).map(|_| {
//         distort_pattern(&base_pattern, noise_level)
//     })
//     .collect()
// }

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

pub fn spike_train_gap_juncton<T: SpikeTrain + Potentiation, U: GapConductance>(
    presynaptic_neuron: &T,
    postsynaptic_neuron: &U,
    timestep: usize,
) -> f64 {
    let (v_max, v_resting) = presynaptic_neuron.get_height();

    if let None = presynaptic_neuron.get_last_firing_time() {
        return v_resting;
    }

    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    let last_firing_time = presynaptic_neuron.get_last_firing_time().unwrap();
    let refractoriness_function = presynaptic_neuron.get_refractoriness_function();
    let dt = presynaptic_neuron.get_refractoriness_timestep();
    let conductance = postsynaptic_neuron.get_gap_conductance();

    sign * conductance * refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
}

pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
    spike_train: &mut T,
    presynaptic_neuron: &mut U, 
    postsynaptic_neuron: &mut U,
    timestep: usize,
    do_receptor_kinetics: bool,
    bayesian: bool,
) {
    let input_current = spike_train_gap_juncton(spike_train, presynaptic_neuron, timestep);

    let (pre_t_total, post_t_total, current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let pre_t_total = if do_receptor_kinetics {
            let mut t = spike_train.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, pre_bayesian_factor);

            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    } else {
        let pre_t_total = if do_receptor_kinetics {
            let t = spike_train.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    };

    let spike_train_spiking = spike_train.iterate();   
    if spike_train_spiking {
        spike_train.set_last_firing_time(Some(timestep));
    }
    
    let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        input_current,
        pre_t_total.as_ref(),
    );
    if pre_spiking {
        presynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        current,
        post_t_total.as_ref(),
    ); 
    if post_spiking {
        postsynaptic_neuron.set_last_firing_time(Some(timestep));
    }
}
