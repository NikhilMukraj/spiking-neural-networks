use std::{
    fs::File, 
    io::{Result, Error, ErrorKind, Write, BufWriter}, 
};
use rand::Rng;
#[path = "../distribution/mod.rs"]
mod distribution;
use distribution::limited_distr;


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
            alpha_init: 6., // arbitrary a value
            beta_init: 10., // arbitrary b value
            d_init: 2., // arbitrary d value
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
            alpha_init: 6., // arbitrary a value
            beta_init: 10., // arbitrary b value
            d_init: 2., // arbitrary d value
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
            alpha_init: 0.02, // arbitrary a value
            beta_init: 0.2, // arbitrary b value
            d_init: 8.0, // arbitrary d value
            dt: 0.5, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: BayesianParameters::default().mean, // center of norm distr
            bayesian_std: BayesianParameters::default().std, // std of norm distr
            bayesian_max: BayesianParameters::default().max, // maximum cutoff for norm distr
            bayesian_min: BayesianParameters::default().min, // minimum cutoff for norm distr
        }
    }
}

#[derive(Clone)]
pub struct STDPParameters {
    pub a_plus: f64, // postitive stdp modifier 
    pub a_minus: f64, // negative stdp modifier 
    pub tau_plus: f64, // postitive stdp decay modifier 
    pub tau_minus: f64, // negative stdp decay modifier 
    pub weight_init: f64, // initial weight value
    pub weight_std: f64, // weight deviation
    pub weight_min: f64, // minimum weight value
    pub weight_max: f64, // maximum weight value
}

impl Default for STDPParameters {
    fn default() -> Self {
        STDPParameters { 
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 45., 
            tau_minus: 45., 
            weight_init: 3.5,
            weight_std: 1.0,
            weight_min: 1.75,
            weight_max: 5.25,
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
    pub alpha: f64, // arbitrary value (controls speed in izhikevich)
    pub beta: f64, // arbitrary value (controls sensitivity to w in izhikevich)
    pub c: f64, // after spike reset value for voltage
    pub d: f64, // after spike reset value for w
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
            a_plus: STDPParameters::default().a_plus,
            a_minus: STDPParameters::default().a_minus,
            tau_plus: STDPParameters::default().tau_plus,
            tau_minus: STDPParameters::default().tau_minus,
            last_firing_time: None,
            alpha: IFParameters::default().alpha_init,
            beta: IFParameters::default().beta_init,
            c: IFParameters::default().v_reset,
            d: IFParameters::default().d_init,
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
            a_plus: STDPParameters::default().a_plus,
            a_minus: STDPParameters::default().a_minus,
            tau_plus: STDPParameters::default().tau_plus,
            tau_minus: STDPParameters::default().tau_minus,
            last_firing_time: None,
            alpha: IFParameters::izhikevich_default().alpha_init,
            beta: IFParameters::izhikevich_default().beta_init,
            c: IFParameters::izhikevich_default().v_reset,
            d: IFParameters::izhikevich_default().d_init,
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
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
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
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
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
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
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
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
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

// #[derive(Clone)]
// struct Gate {
//     alpha: f64,
//     beta: f64,
//     state: f64,
// }

// impl Gate {
//     fn init_state(&mut self) {
//         self.state = self.alpha / (self.alpha + self.beta);
//     }

//     fn update(&mut self, dt: f64) {
//         let alpha_state: f64 = self.alpha * (1. - self.state);
//         let beta_state: f64 = self.beta * self.state;
//         self.state += dt * (alpha_state - beta_state);
//     }
// }

// struct HodgkinHuxleyCell {
//     voltage: f64,
//     dt: f64,
//     cm: f64,
//     e_na: f64,
//     e_k: f64,
//     e_k_leak: f64,
//     g_na: f64,
//     g_k: f64,
//     g_k_leak: f64,
//     m: Gate,
//     n: Gate,
//     h: Gate,
// }

// impl Default for HodgkinHuxleyCell {
//     fn default() -> Self {
//         let default_gate = Gate {
//             alpha: 0.,
//             beta: 0.,
//             state: 0.,
//         };

//         ConductanceBasedCell { 
//             voltage: 0.,
//             dt: 0.1,
//             cm: 1., 
//             e_na: 115., 
//             e_k: -12., 
//             e_k_leak: 10.6, 
//             g_na: 120., 
//             g_k: 36., 
//             g_k_leak: 0.3, 
//             m: default_gate.clone(), 
//             n: default_gate.clone(), 
//             h: default_gate,   
//         }
//     }
// }

// // https://github.com/swharden/pyHH/blob/master/src/pyhh/models.py
// // https://github.com/openworm/hodgkin_huxley_tutorial/blob/71aaa509021d8c9c55dd7d3238eaaf7b5bd14893/Tutorial/Source/HodgkinHuxley.py#L4
// impl HodgkinHuxleyCell {
//     fn update_gate_time_constants(&mut self, voltage: f64) {
//         self.n.alpha = 0.01 * ((10. - voltage) / (((10. - voltage) / 10.)-1.).exp());
//         self.n.beta = 0.125 * (-voltage / 80.).exp();
//         self.m.alpha = 0.1 * ((25. - voltage) / ((25. - voltage) / 10.) - 1.).exp();
//         self.m.beta = 4. * (-voltage / 18.).exp();
//         self.h.alpha = 0.07 * (-voltage / 20.).exp();
//         self.h.beta = 1. / (((30. - voltage) / 10.).exp() + 1.);
//     }

//     fn initialize_parameters(&mut self, starting_voltage: f64) {
//         self.voltage = starting_voltage;
//         self.update_gate_time_constants(starting_voltage);
//         self.m.init_state();
//         self.n.init_state();
//         self.h.init_state();
//     }

//     fn update_cell_voltage(&mut self, input: f64) {
//         let i_na = self.m.state.powf(3.) * self.g_na * self.h.state * (self.voltage - self.e_na);
//         let i_k = self.n.state.powf(4.) * self.g_k * (self.voltage - self.e_k);
//         let i_k_leak = self.g_k_leak * (self.voltage - self.e_k_leak);
//         let i_sum = input - i_na - i_k - i_k_leak;
//         self.voltage += self.dt * i_sum / self.cm;
//     }

//     fn update_gate_states(&mut self) {
//         self.m.update(self.dt);
//         self.n.update(self.dt);
//         self.h.update(self.dt);
//     }

//     fn iterate(&mut self, input: f64) {
//         self.update_gate_time_constants(self.voltage);
//         self.update_cell_voltage(input);
//         self.update_gate_states();
//     }
// }
