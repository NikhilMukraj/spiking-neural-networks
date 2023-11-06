use std::{
    fs::File, 
    io::{Result, Error, ErrorKind, Write}, 
};
use rand::Rng;
use rand_distr::{Normal, Distribution};


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
    pub alpha: f64,
    pub beta: f64,
    pub d: f64,
    pub dt: f64,
    pub exp_dt: f64,
    pub bayesian_mean: f64,
    pub bayesian_std: f64,
    pub bayesian_max: f64,
    pub bayesian_min: f64,
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
            alpha: 6., // arbitrary a value
            beta: 10., // arbitrary b value
            d: 2., // arbitrary d value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: BayesianParameters::default().mean, // center of norm distr
            bayesian_std: BayesianParameters::default().std, // std of norm distr
            bayesian_max: BayesianParameters::default().max, // maximum cutoff for norm distr
            bayesian_min: BayesianParameters::default().min, // minimum cutoff for norm distr
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
            alpha: 6., // arbitrary a value
            beta: 10., // arbitrary b value
            d: 2., // arbitrary d value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: BayesianParameters::default().mean, // center of norm distr
            bayesian_std: BayesianParameters::default().std, // std of norm distr
            bayesian_max: BayesianParameters::default().max, // maximum cutoff for norm distr
            bayesian_min: BayesianParameters::default().min, // minimum cutoff for norm distr
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
            alpha: 0.02, // arbitrary a value
            beta: 0.2, // arbitrary b value
            d: 8.0, // arbitrary d value
            dt: 0.5, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: BayesianParameters::default().mean, // center of norm distr
            bayesian_std: BayesianParameters::default().std, // std of norm distr
            bayesian_max: BayesianParameters::default().max, // maximum cutoff for norm distr
            bayesian_min: BayesianParameters::default().min, // minimum cutoff for norm distr
        }
    }
}

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
    pub a_plus: f64, // postitive stdp modifier 
    pub a_minus: f64, // negative stdp modifier 
    pub tau_plus: f64, // postitive stdp decay modifier 
    pub tau_minus: f64, // negative stdp decay modifier 
    pub last_firing_time: Option<usize>,
}

pub fn limited_distr(mean: f64, std_dev: f64, minimum: f64, maximum: f64) -> f64 {
    if std_dev == 0.0 {
        return mean;
    }

    let normal = Normal::new(mean, std_dev).unwrap();
    let output: f64 = normal.sample(&mut rand::thread_rng());
   
    output.max(minimum).min(maximum)
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
            lif.alpha * (self.current_voltage - lif.e_l) -
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
            self.w_value += lif.beta;
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
            lif.alpha * (lif.beta * self.current_voltage - self.w_value)
        ) * (lif.dt / lif.tau_m);

        self.w_value += dw;

        let mut is_spiking = false;

        if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = lif.v_reset;
            self.w_value += lif.d;
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
        let mut file = File::create(filename)
            .expect("Unable to create file");
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");

        for _ in 0..iterations {
            let (dv, _is_spiking) = if bayesian {
                self.get_dv_change_and_spike(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
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
        let mut file = File::create(filename)
            .expect("Unable to create file");
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.apply_dw_change_and_get_spike(lif);
            let dv = if bayesian {
                self.adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
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
        let mut file = File::create(filename)
            .expect("Unable to create file");
        writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.apply_dw_change_and_get_spike(lif);
            let dv = if bayesian {
                self.exp_adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
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
        let mut file = File::create(filename)
            .expect("Unable to create file");
        writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.izhikevich_apply_dw_and_get_spike(if_params);
            let dv = if bayesian {
                self.izhikevich_get_dv_change(if_params, i * limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.))
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
        let mut file = File::create(filename)
            .expect("Unable to create file");
        writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        
        for _ in 0..iterations {
            let _is_spiking = self.izhikevich_apply_dw_and_get_spike(if_params);
            let dv = if bayesian {
                self.izhikevich_leaky_get_dv_change(if_params, i * limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.))
            } else {
                self.izhikevich_leaky_get_dv_change(if_params, i)
            };
            self.current_voltage += dv;

            writeln!(file, "{}, {}", self.current_voltage, self.w_value).expect("Unable to write to file");
        }
    }
}

pub type CellGrid = Vec<Vec<Cell>>;
