use std::{
    fs::File, 
    io::{Result, Error, ErrorKind, Write, BufWriter}, 
};
use rand::Rng;
#[path = "../distribution/mod.rs"]
mod distribution;
use distribution::limited_distr;


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

#[derive(Debug, Clone)]
pub struct IFParameters {
    pub v_th: f64,
    pub v_reset: f64,
    pub tau_m: f64,
    pub g_l: f64,
    pub v_init: f64,
    pub e_l: f64,
    pub tref: f64,
    pub w_init: f64,
    pub alpha_init: f64,
    pub beta_init: f64,
    pub d_init: f64,
    pub dt: f64,
    pub exp_dt: f64,
    pub bayesian_params: BayesianParameters,
    // total_time: f64,
}

impl Default for IFParameters {
    fn default() -> Self {
        IFParameters { 
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            alpha_init: 6., // arbitrary a value
            beta_init: 10., // arbitrary b value
            d_init: 2., // arbitrary d value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_params: BayesianParameters::default(), // default bayesian parameters
        }
    }
}

pub trait ScaledDefault {
    fn scaled_default() -> Self;
}

impl ScaledDefault for IFParameters {
    fn scaled_default() -> Self {
        IFParameters { 
            v_th: 1., // spike threshold (mV)
            v_reset: 0., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            g_l: 4.25, // leak conductance (nS) ((10 - (-75)) / ((-55) - (-75))) * (1 - 0)) + 1
            v_init: 0., // initial potential (mV)
            e_l: 0., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            alpha_init: 6., // arbitrary a value
            beta_init: 10., // arbitrary b value
            d_init: 2., // arbitrary d value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_params: BayesianParameters::default(), // default bayesian parameters
        }
    }
}

pub trait IzhikevichDefault {
    fn izhikevich_default() -> Self;
}

impl IzhikevichDefault for IFParameters {
    fn izhikevich_default() -> Self {
        IFParameters { 
            v_th: 30., // spike threshold (mV)
            v_reset: -65., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            g_l: 10., // leak conductance (nS)
            v_init: -65., // initial potential (mV)
            e_l: -65., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 30., // initial w value
            alpha_init: 0.02, // arbitrary a value
            beta_init: 0.2, // arbitrary b value
            d_init: 8.0, // arbitrary d value
            dt: 0.5, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_params: BayesianParameters::default(), // default bayesian parameters
        }
    }
}

#[derive(Clone)]
pub struct STDPParameters {
    pub a_plus: f64, // postitive stdp modifier 
    pub a_minus: f64, // negative stdp modifier 
    pub tau_plus: f64, // postitive stdp decay modifier 
    pub tau_minus: f64, // negative stdp decay modifier 
    pub weight_bayesian_params: BayesianParameters, // weight initialization parameters
}

impl Default for STDPParameters {
    fn default() -> Self {
        STDPParameters { 
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 45., 
            tau_minus: 45., 
            weight_bayesian_params: BayesianParameters {
                mean: 3.5,
                std: 1.0,
                min: 1.75,
                max: 1.75,
            },
        }
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone)]
pub enum PotentiationType {
    Excitatory,
    Inhibitory,
}

impl PotentiationType {
    pub fn weighted_random_type(prob: f64) -> PotentiationType {
        if rand::thread_rng().gen_range(0.0..=1.0) <= prob {
            PotentiationType::Excitatory
        } else {
            PotentiationType::Inhibitory
        }
    }
}

#[derive(Clone)]
pub struct Cell {
    pub current_voltage: f64, // membrane potential
    pub refractory_count: f64, // keeping track of refractory period
    pub leak_constant: f64, // leak constant gene
    pub integration_constant: f64, // integration constant gene
    pub potentiation_type: PotentiationType,
    pub neurotransmission_concentration: f64, // concentration of neurotransmitter in synapse
    pub neurotransmission_release: f64, // concentration of neurotransmitter released at spiking
    pub receptor_density: f64, // factor of how many receiving receptors for a given neurotransmitter
    pub chance_of_releasing: f64, // chance cell can produce neurotransmitter
    pub dissipation_rate: f64, // how quickly neurotransmitter concentration decreases
    pub chance_of_random_release: f64, // likelyhood of neuron randomly releasing neurotransmitter
    pub random_release_concentration: f64, // how much neurotransmitter is randomly released
    pub w_value: f64, // adaptive value 
    pub stdp_params: STDPParameters, // stdp parameters
    pub last_firing_time: Option<usize>,
    pub alpha: f64, // arbitrary value (controls speed in izhikevich)
    pub beta: f64, // arbitrary value (controls sensitivity to w in izhikevich)
    pub c: f64, // after spike reset value for voltage
    pub d: f64, // after spike reset value for w
    pub last_dv: f64, // last change in voltage
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            current_voltage: IFParameters::default().v_init, 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            potentiation_type: PotentiationType::Excitatory,
            neurotransmission_concentration: 0., 
            neurotransmission_release: 0.,
            receptor_density: 0.,
            chance_of_releasing: 0., 
            dissipation_rate: 0., 
            chance_of_random_release: 0.,
            random_release_concentration: 0.,
            w_value: IFParameters::default().w_init,
            stdp_params: STDPParameters::default(),
            last_firing_time: None,
            alpha: IFParameters::default().alpha_init,
            beta: IFParameters::default().beta_init,
            c: IFParameters::default().v_reset,
            d: IFParameters::default().d_init,
            last_dv: 0.,
        }
    }
}

impl IzhikevichDefault for Cell {
    fn izhikevich_default() -> Self {
        Cell {
            current_voltage: IFParameters::izhikevich_default().v_init, 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            potentiation_type: PotentiationType::Excitatory,
            neurotransmission_concentration: 0., 
            neurotransmission_release: 0.,
            receptor_density: 0.,
            chance_of_releasing: 0., 
            dissipation_rate: 0., 
            chance_of_random_release: 0.,
            random_release_concentration: 0.,
            w_value: IFParameters::izhikevich_default().w_init,
            stdp_params: STDPParameters::default(),
            last_firing_time: None,
            alpha: IFParameters::izhikevich_default().alpha_init,
            beta: IFParameters::izhikevich_default().beta_init,
            c: IFParameters::izhikevich_default().v_reset,
            d: IFParameters::izhikevich_default().d_init,
            last_dv: 0.,
        }
    }
}

impl Cell {
    pub fn get_dv_change_and_spike(&mut self, lif: &IFParameters, i: f64) -> (f64, bool) {
        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = lif.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = lif.v_reset;
            self.refractory_count = lif.tref / lif.dt
        }

        // let dv = (-1. * (self.current_voltage - lif.e_l) + i / lif.g_l) * (lif.dt / lif.tau_m);
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (self.integration_constant * (i / lif.g_l))
        ) * (lif.dt / lif.tau_m);
        // could be varied with a leak constant instead of -1 *
        // input could be varied with a integration constant times the input

        return (dv, is_spiking);
    }

    pub fn apply_dw_change_and_get_spike(&mut self, lif: &IFParameters) -> bool {
        // dw = (self.a * (v[it]-self.V_L) - w[it]) * (self.dt/self.tau_m)
        let dw = (
            lif.alpha_init * (self.current_voltage - lif.e_l) -
            self.w_value
        ) * (lif.dt / lif.tau_m);

        self.w_value += dw;

        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = lif.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = lif.v_reset;
            self.w_value += lif.beta_init;
            self.refractory_count = lif.tref / lif.dt
        }

        return is_spiking;
    }

    pub fn adaptive_get_dv_change(&mut self, lif: &IFParameters, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (self.integration_constant * (i / lif.g_l)) - 
            (self.w_value / lif.g_l)
        ) * (lif.dt / lif.tau_m);

        dv
    }

    pub fn exp_adaptive_get_dv_change(&mut self, lif: &IFParameters, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (lif.exp_dt * ((self.current_voltage - lif.v_th) / lif.exp_dt).exp()) +
            (self.integration_constant * (i / lif.g_l)) - 
            (self.w_value / lif.g_l)
        ) * (lif.dt / lif.tau_m);

        dv
    }

    pub fn izhikevich_apply_dw_and_get_spike(&mut self, lif: &IFParameters) -> bool {
        let dw = (
            self.alpha * (self.beta * self.current_voltage - self.w_value)
        ) * (lif.dt / lif.tau_m);

        self.w_value += dw;

        let mut is_spiking = false;

        if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = self.c;
            self.w_value += self.d;
        }

        return is_spiking;
    }

    pub fn izhikevich_get_dv_change(&mut self, lif: &IFParameters, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (lif.dt / lif.tau_m);

        dv
    }

    pub fn izhikevich_leaky_get_dv_change(&mut self, lif: &IFParameters, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - 
            self.w_value * (self.current_voltage - lif.e_l) + i
        ) * (lif.dt / lif.tau_m);

        dv
    }

    pub fn determine_neurotransmitter_concentration(&mut self, is_spiking: bool) {
        // (excitatory should increase voltage)
        // (inhibitory should decrease voltage)
        // (may also depend on kind of receptor)
        let prob = rand::thread_rng().gen_range(0.0..=1.0);
        if is_spiking && (prob <= self.chance_of_releasing) {
            self.neurotransmission_concentration += self.neurotransmission_release;
        } else if self.neurotransmission_concentration > 0. {
            let concentration = (
                    self.neurotransmission_concentration - self.dissipation_rate
                )
                .max(0.0); // reduce concentration until 0
            self.neurotransmission_concentration = concentration;
        }
        
        let prob = rand::thread_rng().gen_range(0.0..=1.0);
        if self.refractory_count <= 0. && prob <= self.chance_of_random_release {
            self.neurotransmission_concentration += self.random_release_concentration;
        }
    }

    // voltage of cell should be initial voltage + this change
    pub fn run_static_input(
        &mut self, 
        lif: &IFParameters, 
        i: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        writeln!(file, "voltage").expect("Unable to write to file");
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");

        for _ in 0..iterations {
            let (dv, _is_spiking) = if bayesian {
                self.get_dv_change_and_spike(lif, i * limited_distr(lif.bayesian_params.mean, lif.bayesian_params.std, 0., 1.))
            } else {
                self.get_dv_change_and_spike(lif, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        }
    }

    pub fn run_adaptive_static_input(
        &mut self, 
        lif: &IFParameters, 
        i: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.apply_dw_change_and_get_spike(lif);
            let dv = if bayesian {
                self.adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_params.mean, lif.bayesian_params.std, 0., 1.))
            } else {
                self.adaptive_get_dv_change(lif, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        }
    }

    pub fn run_exp_adaptive_static_input(
        &mut self, 
        lif: &IFParameters, 
        i: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.apply_dw_change_and_get_spike(lif);
            let dv = if bayesian {
                self.exp_adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_params.mean, lif.bayesian_params.std, 0., 1.))
            } else {
                self.exp_adaptive_get_dv_change(lif, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        }
    }

    pub fn run_izhikevich_static_input(
        &mut self, 
        if_params: &IFParameters, 
        i: f64, 
        bayesian: bool, 
        iterations: usize,
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.izhikevich_apply_dw_and_get_spike(if_params);
            let dv = if bayesian {
                self.izhikevich_get_dv_change(if_params, i * limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.))
            } else {
                self.izhikevich_get_dv_change(if_params, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        }
    }

    pub fn run_izhikevich_leaky_static_input(
        &mut self, 
        if_params: &IFParameters, 
        i: f64, 
        bayesian: bool, 
        iterations: usize,
        filename: &str,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.izhikevich_apply_dw_and_get_spike(if_params);
            let dv = if bayesian {
                self.izhikevich_leaky_get_dv_change(if_params, i * limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.))
            } else {
                self.izhikevich_leaky_get_dv_change(if_params, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        }
    }
}

pub type CellGrid = Vec<Vec<Cell>>;

// fn heaviside(x: f64) -> f64 {
//     if (x > 0) {
//         1.
//     } else {
//         0.
//     }
// }

// struct NMDA {
//     g_nmda: f64, // 1.0 nS
//     tau1: f64,
//     tau2: f64,
//     mg_conc: f64,
//     alpha: f64,
//     beta: f64,
//     e_syn: f64,
// }

// impl NMDA {
//     fn caulcate_tpk(&self) -> f64 {
//         (self.tau1 * self.tau2) / (self.tau2 - self.tau1) * (self.tau2 / self.tau1).ln()
//     }

//     fn calculate_gamma(&self) -> f64 {
//         let tpk = calculate_tpk();
//         1. / ((-tpk / self.tau2).exp() - (-tpk / self.tau1).exp())
//     }

//     fn calculate_g_syn(&self, t: f64, voltage: f64) -> f64 {
//         let gamma = self.calculate_gamma();
//         let term1 =  gamma / (1 + self.mg_conc * (-self.alpha * voltage).exp() / self.beta);
//         let term2 = ((-t / self.tau2).exp() - (-t / self.tau1).exp()) * heaviside(t);
//         self.g_nmda * term1 * term2
//     }

    // fn calculate_b(&self, voltage: f64) -> f64 {
    //     1. / (1. + ((-0.062 * voltage).exp() * self.mg_conc / 3.57))
    // }
// }

#[derive(Clone, Copy)]
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

pub struct GABAbDissociation {
    g: f64,
    n: f64,
    kd: f64,
    // k1: ,
    // k2: ,
    k3: f64,
    k4: f64,
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

pub struct Neurotransmitter {
    pub t_max: f64,
    pub alpha: f64,
    pub beta: f64,
    pub t: f64,
    pub r: f64,
    pub v_p: f64,
    pub k_p: f64,
}

impl Default for Neurotransmitter {
    fn default() -> Self {
        Neurotransmitter {
            t_max: 1.,
            alpha: 1.,
            beta: 1.,
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl AMPADefault for Neurotransmitter {
    fn ampa_default() -> Self {
        Neurotransmitter {
            t_max: 1.,
            alpha: 1.1, // mM^-1 * ms^-1
            beta: 0.19, // ms^-1
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl GABAaDefault for Neurotransmitter {
    fn gabaa_default() -> Self {
        Neurotransmitter {
            t_max: 1.,
            alpha: 5.0, // mM^-1 * ms^-1
            beta: 0.18, // ms^-1
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl GABAbDefault for Neurotransmitter {
    fn gabab_default() -> Self {
        Neurotransmitter {
            t_max: 0.5,
            alpha: 0.016, // mM^-1 * ms^-1
            beta: 0.0047, // ms^-1
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl GABAbDefault2 for Neurotransmitter {
    fn gabab_default2() -> Self {
        Neurotransmitter {
            t_max: 0.5,
            alpha: 0.52, // mM^-1 * ms^-1 // k1
            beta: 0.0013, // ms^-1 // k2
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl NMDADefault for Neurotransmitter {
    fn nmda_default() -> Self {
        Neurotransmitter {
            t_max: 1.,
            alpha: 0.072, // mM^-1 * ms^-1
            beta: 0.0066, // ms^-1
            t: 0.,
            r: 0.,
            v_p: 2., // 2 mV
            k_p: 5., // 5 mV
        }
    }
}

impl Neurotransmitter {
    fn apply_r_change(&mut self, dt: f64) {
        self.r += (self.alpha * self.t * (1. - self.r) - self.beta * self.r) * dt;
    }

    fn apply_t_change(&mut self, voltage: f64) {
        self.t = self.t_max / (1. + (-(voltage - self.v_p) / self.k_p).exp());
    }
}

pub enum NeurotransmitterType {
    AMPA,
    GABAa,
    GABAb(GABAbDissociation),
    NMDA(BV),
    Basic,
}

pub struct GeneralLigandGatedChannel {
    pub g: f64,
    pub reversal: f64,
    pub neurotransmitter: Neurotransmitter,
    pub neurotransmitter_type: NeurotransmitterType,
    pub current: f64,
}

impl Default for GeneralLigandGatedChannel {
    fn default() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            neurotransmitter: Neurotransmitter::default(),
            neurotransmitter_type: NeurotransmitterType::Basic,
            current: 0.,
        }
    }
}

impl AMPADefault for GeneralLigandGatedChannel {
    fn ampa_default() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            neurotransmitter: Neurotransmitter::ampa_default(),
            neurotransmitter_type: NeurotransmitterType::AMPA,
            current: 0.,
        }
    }
}

impl GABAaDefault for GeneralLigandGatedChannel {
    fn gabaa_default() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: -80., // 0.0 mV
            neurotransmitter: Neurotransmitter::gabaa_default(),
            neurotransmitter_type: NeurotransmitterType::GABAa,
            current: 0.,
        }
    }
}

impl GABAbDefault for GeneralLigandGatedChannel {
    fn gabab_default() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: -95., // 0.0 mV
            neurotransmitter: Neurotransmitter::gabab_default(),
            neurotransmitter_type: NeurotransmitterType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl GABAbDefault2 for GeneralLigandGatedChannel {
    fn gabab_default2() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: -95., // 0.0 mV
            neurotransmitter: Neurotransmitter::gabab_default2(),
            neurotransmitter_type: NeurotransmitterType::GABAb(GABAbDissociation::default()),
            current: 0.,
        }
    }
}

impl NMDADefault for GeneralLigandGatedChannel {
    fn nmda_default() -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            neurotransmitter: Neurotransmitter::nmda_default(),
            neurotransmitter_type: NeurotransmitterType::NMDA(BV::default()),
            current: 0.,
        }
    }
}

pub trait NMDAWithBV {
    fn nmda_with_bv(bv: BV) -> Self;
}

impl NMDAWithBV for GeneralLigandGatedChannel {
    fn nmda_with_bv(bv: BV) -> Self {
        GeneralLigandGatedChannel {
            g: 1.0, // 1.0 nS
            reversal: 0., // 0.0 mV
            neurotransmitter: Neurotransmitter::nmda_default(),
            neurotransmitter_type: NeurotransmitterType::NMDA(bv),
            current: 0.,
        }
    }
}

impl GeneralLigandGatedChannel {
    pub fn calculate_g(&mut self, voltage: f64, r: f64, dt: f64) -> f64 {
        let modifier = match &mut self.neurotransmitter_type {
            NeurotransmitterType::AMPA => 1.0,
            NeurotransmitterType::GABAa => 1.0,
            NeurotransmitterType::GABAb(value) => {
                value.g += (value.k3 * r - value.k4 * value.g) * dt;
                value.calculate_modifer()
            }, // G^N / (G^N + Kd)
            NeurotransmitterType::NMDA(value) => value.calculate_b(voltage),
            NeurotransmitterType::Basic => 1.0,
        };

        self.current = modifier * self.g * (voltage - self.reversal);

        self.current
    }

    pub fn to_str(&self) -> &str {
        match self.neurotransmitter_type {
            NeurotransmitterType::AMPA => "AMPA",
            NeurotransmitterType::GABAa => "GABAa",
            NeurotransmitterType::GABAb(_) => "GABAb",
            NeurotransmitterType::NMDA(_) => "NMDA",
            NeurotransmitterType::Basic => "Basic",
        }
    }
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
pub struct HighThresholdCalciumChannel {
    current: f64,
    z: f64,
    f: f64,
    r: f64,
    temp: f64,
    ca_in: f64,
    ca_in_equilibrium: f64,
    ca_out: f64,
    permeability: f64,
    max_permeability: f64,
    d: f64,
    kt: f64,
    kd: f64,
    tr: f64,
    k: f64,
    p: f64,
}

impl Default for HighThresholdCalciumChannel {
    fn default() -> Self {
        HighThresholdCalciumChannel {
            current: 0.,
            z: 2.,
            f: 96489., // C/mol
            r: 8.31, // J/Kmol
            temp: 35., // degrees c
            ca_in: 0.001, // mM
            ca_in_equilibrium: 0.001, // mM
            ca_out: 5., // mM
            permeability: 0.,
            max_permeability: 5.36e-6,
            d: 0.1, // um
            kt: 1e-4, // mM / ms
            kd: 1e-4, // mM
            tr: 43., // ms
            k: 1000.,
            p: 0.02,
        }
    }
}

impl HighThresholdCalciumChannel {
    // m^x * n^y
    // x and y here probably refer to 3 and 4
    fn update_permeability(&mut self, m_state: f64, n_state: f64) {
        self.permeability = self.max_permeability * m_state * n_state;
    }

    fn update_ca_in(&mut self, dt: f64) {
        let term1 = self.k * (-self.current / (2. * self.f * self.d));
        let term2 = self.p * ((self.kt * self.ca_in) / (self.ca_in + self.kd));
        let term3 = (self.ca_in_equilibrium - self.ca_in) / self.tr;
        self.ca_in +=  (term1 + term2 + term3) * dt;
    }

    fn get_ca_current(&self, voltage: f64) -> f64 {
        let r_by_temp = self.r * self.temp;
        let term1 = self.permeability * self.z.powf(2.) * ((voltage * self.f.powf(2.)) / r_by_temp);
        let term2 = self.ca_in - (self.ca_out * ((-self.z * self.f * voltage) / r_by_temp)).exp();
        let term3 = 1. - ((-self.z * self.f * voltage) / r_by_temp).exp();

        term1 * (term2 / term3)
    }

    fn get_ca_current_and_update(&mut self, m: f64, n: f64, dt: f64, voltage: f64) -> f64 {
        self.update_permeability(m.powf(3.), n.powf(4.));
        self.update_ca_in(dt);
        self.current = self.get_ca_current(voltage);

        self.current
    }
}

pub enum AdditionalGates {
    LTypeCa(HighThresholdCalciumChannel),
}

impl AdditionalGates {
    fn get_and_update_current(&mut self, m: f64, n: f64, dt: f64, voltage: f64) -> f64 {
        match self {
            AdditionalGates::LTypeCa(channel) => channel.get_ca_current_and_update(m, n, dt, voltage),
        }
    }

    fn get_current(&self) -> f64 {
        match &self {
            AdditionalGates::LTypeCa(channel) => channel.current
        }
    }

    fn to_str(&self) -> &str {
        match &self {
            AdditionalGates::LTypeCa(_) => "LTypeCa",
        }
    }
}

// multicomparment stuff, refer to dopamine modeling paper as well
// https://github.com/antgon/msn-model/blob/main/msn/cell.py 
// pub struct Soma {

// }

// pub struct Dendrite {

// }

#[derive(Clone)]
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

pub struct HodgkinHuxleyCell {
    pub current_voltage: f64,
    pub dt: f64,
    pub cm: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_k_leak: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_k_leak: f64,
    pub m: Gate,
    pub n: Gate,
    pub h: Gate,
    pub ligand_gates: Vec<GeneralLigandGatedChannel>,
    pub additional_gates: Vec<AdditionalGates>,
    pub bayesian_params: BayesianParameters,
}

impl Default for HodgkinHuxleyCell {
    fn default() -> Self {
        let default_gate = Gate {
            alpha: 0.,
            beta: 0.,
            state: 0.,
        };

        HodgkinHuxleyCell { 
            current_voltage: 0.,
            dt: 0.1,
            cm: 1., 
            e_na: 115., 
            e_k: -12., 
            e_k_leak: 10.6, 
            g_na: 120., 
            g_k: 36., 
            g_k_leak: 0.3, 
            m: default_gate.clone(), 
            n: default_gate.clone(), 
            h: default_gate,  
            ligand_gates: vec![],
            additional_gates: vec![],
            bayesian_params: BayesianParameters::default() 
        }
    }
}

// https://github.com/swharden/pyHH/blob/master/src/pyhh/models.py
impl HodgkinHuxleyCell {
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
    }

    pub fn update_cell_voltage(&mut self, input_current: f64) {
        let i_na = self.m.state.powf(3.) * self.g_na * self.h.state * (self.current_voltage - self.e_na);
        let i_k = self.n.state.powf(4.) * self.g_k * (self.current_voltage - self.e_k);
        let i_k_leak = self.g_k_leak * (self.current_voltage - self.e_k_leak);

        let i_ligand_gates = self.ligand_gates
            .iter_mut()
            .map(|i| 
                i.calculate_g(self.current_voltage, i.neurotransmitter.r, self.dt) * i.neurotransmitter.r
            )
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>();

        let i_additional_gates = self.additional_gates
            .iter_mut()
            .map(|i| 
                i.get_and_update_current(self.m.state, self.n.state, self.dt, self.current_voltage)
            ) 
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>();

        let i_sum = input_current - (i_na + i_k + i_k_leak) + i_ligand_gates + i_additional_gates;
        self.current_voltage += self.dt * i_sum / self.cm;
    }

    pub fn update_neurotransmitter(&mut self, presynaptic_voltage: f64) {
        self.ligand_gates
            .iter_mut()
            .for_each(|i| {
                i.neurotransmitter.apply_t_change(presynaptic_voltage);
                i.neurotransmitter.apply_r_change(self.dt);
            });
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
    }

    pub fn run_static_input(
        &mut self, 
        input: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str, 
        full: bool
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        if !full {
            writeln!(file, "voltage").expect("Unable to write to file");
            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        } else {
            // writeln!(file, "voltage,m,n,h").expect("Unable to write to file");
            // writeln!(file, "{}, {}, {}, {}", 
            //     self.current_voltage, 
            //     self.m.state, 
            //     self.n.state, 
            //     self.h.state,
            // ).expect("Unable to write to file");

            write!(file, "voltage,m,n,h").expect("Unable to write to file");
            writeln!(
                file, 
                "{}",
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
                "{}",
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
                // writeln!(file, "{}, {}, {}, {}", 
                //     self.current_voltage, 
                //     self.m.state, 
                //     self.n.state, 
                //     self.h.state,
                // ).expect("Unable to write to file");

                write!(file, "{}, {}, {}, {}", 
                    self.current_voltage, 
                    self.m.state, 
                    self.n.state, 
                    self.h.state,
                ).expect("Unable to write to file");
                writeln!(
                    file, 
                    "{}",
                    self.additional_gates.iter()
                        .map(|x| x.get_current().to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                ).expect("Unable to write to file");
            }
        }
    }
}
