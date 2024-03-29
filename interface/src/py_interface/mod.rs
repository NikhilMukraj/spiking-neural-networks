// use std::io::{Result, Error, ErrorKind};
use pyo3::prelude::*;
use pyo3::exceptions::PyLookupError;
use crate::neuron::{IFParameters, IFType, Cell, PotentiationType, HodgkinHuxleyCell};
use crate::distribution::limited_distr;


// pub current_voltage: f64, // membrane potential
// pub refractory_count: f64, // keeping track of refractory period
// pub leak_constant: f64, // leak constant gene
// pub integration_constant: f64, // integration constant gene
// pub potentiation_type: PotentiationType,
// pub neurotransmission_concentration: f64, // concentration of neurotransmitter in synapse
// pub neurotransmission_release: f64, // concentration of neurotransmitter released at spiking
// pub receptor_density: f64, // factor of how many receiving receptors for a given neurotransmitter
// pub chance_of_releasing: f64, // chance cell can produce neurotransmitter
// pub dissipation_rate: f64, // how quickly neurotransmitter concentration decreases
// pub chance_of_random_release: f64, // likelyhood of neuron randomly releasing neurotransmitter
// pub random_release_concentration: f64, // how much neurotransmitter is randomly released
// pub w_value: f64, // adaptive value 
// pub stdp_params: STDPParameters, // stdp parameters
// pub last_firing_time: Option<usize>,
// pub alpha: f64, // arbitrary value (controls speed in izhikevich)
// pub beta: f64, // arbitrary value (controls sensitivity to w in izhikevich)
// pub c: f64, // after spike reset value for voltage
// pub d: f64, // after spike reset value for w

// v_th: 30., // spike threshold (mV)
// v_reset: -65., // reset potential (mV)
// tau_m: 10., // membrane time constant (ms)
// g_l: 10., // leak conductance (nS)
// v_init: -65., // initial potential (mV)
// e_l: -65., // leak reversal potential (mV)
// tref: 10., // refractory time (ms), could rename to refract_time
// w_init: 30., // initial w value
// alpha_init: 0.02, // arbitrary a value
// beta_init: 0.2, // arbitrary b value
// d_init: 8.0, // arbitrary d value
// dt: 0.5, // simulation time step (ms)
// exp_dt: 1., // exponential time step (ms)

#[pyclass]
#[derive(Clone)]
pub struct IFCell {
    pub mode: IFType,
    pub cell_backend: Cell,
    pub if_params: IFParameters,
}

#[pymethods]
impl IFCell {
    #[new]
    #[pyo3(signature=(mode="basic"))]
    fn new(mode: &str) -> PyResult<Self> {
        Ok(
            IFCell { 
                mode: IFType::from_str(mode)?,
                cell_backend: Cell::default(),
                if_params: IFParameters::default(),
            }
        )
    }

    fn get_dv_change_and_spike(&mut self, i: f64) -> (f64, bool) {
        self.cell_backend.get_dv_change_and_spike(&self.if_params, i)
    }
    fn apply_dw_change_and_get_spike(&mut self) -> bool {
        self.cell_backend.apply_dw_change_and_get_spike(&self.if_params)
    }

    fn adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.adaptive_get_dv_change(&self.if_params, i)
    }

    fn exp_adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.exp_adaptive_get_dv_change(&self.if_params, i)
    }
    fn izhikevich_apply_dw_and_get_spike(&mut self) -> bool {
        self.cell_backend.izhikevich_apply_dw_and_get_spike(&self.if_params)
    }

    fn izhikevich_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.izhikevich_get_dv_change(&self.if_params, i)
    }

    fn izhikevich_leaky_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.izhikevich_leaky_get_dv_change(&self.if_params, i)
    }

    #[pyo3(signature = (i, bayesian=false))]
    pub fn iterate_and_return_spike(&mut self, i: f64, bayesian: bool) -> bool {
        let input_voltage = if bayesian {
            i * limited_distr(
                self.if_params.bayesian_params.mean, 
                self.if_params.bayesian_params.std,
                self.if_params.bayesian_params.min,
                self.if_params.bayesian_params.max,
            )
        } else {
            i
        };

        match self.mode {
            IFType::Basic => {
                let (dv, is_spiking) = self.get_dv_change_and_spike(input_voltage);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::Adaptive => {
                let is_spiking = self.apply_dw_change_and_get_spike();
                let dv = self.adaptive_get_dv_change(input_voltage);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::AdaptiveExponential => {
                let is_spiking = self.apply_dw_change_and_get_spike();
                let dv = self.exp_adaptive_get_dv_change(input_voltage);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::Izhikevich => {
                let is_spiking = self.izhikevich_apply_dw_and_get_spike();
                let dv = self.izhikevich_get_dv_change(input_voltage);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::IzhikevichLeaky => {
                let is_spiking = self.izhikevich_apply_dw_and_get_spike();
                let dv = self.izhikevich_leaky_get_dv_change(input_voltage);
                self.cell_backend.current_voltage += dv;

                is_spiking
            }
        }
    }

    #[pyo3(signature = (is_spiking))]
    pub fn determine_neurotransmitter_concentration(&mut self, is_spiking: bool) {
        self.cell_backend.determine_neurotransmitter_concentration(is_spiking);
    }

    #[pyo3(signature = (i, iterations, bayesian=false))]
    pub fn run_static_input(&mut self, i: f64, iterations: usize, bayesian: bool) -> (Vec<f64>, Vec<bool>) {
        let mut voltages: Vec<f64> = vec![];
        let mut is_spikings: Vec<bool> = vec![];

        for _ in 0..iterations {
            let is_spiking = self.iterate_and_return_spike(i, bayesian);
            is_spikings.push(is_spiking);
            voltages.push(self.cell_backend.current_voltage);
        }

        (voltages, is_spikings)
    }
    
    #[pyo3(signature = (param_name, value))]
    pub fn change_param(&mut self, param_name: &str, value: &PyAny) -> PyResult<()> {
        match param_name {
            "v_th" => { self.if_params.v_th = value.extract::<f64>()?; },
            "v_reset" => { self.if_params.v_reset = value.extract::<f64>()?; },
            "tau_m" => { self.if_params.tau_m = value.extract::<f64>()?; },
            "g_l" => { self.if_params.g_l = value.extract::<f64>()?; },
            "v_init" => { self.if_params.v_init = value.extract::<f64>()?; },
            "e_l" => { self.if_params.e_l = value.extract::<f64>()?; },
            "tref" => { self.if_params.tref = value.extract::<f64>()?; },
            "w_init" => { self.if_params.w_init = value.extract::<f64>()?; },
            "alpha_init" => { self.if_params.alpha_init = value.extract::<f64>()?; },
            "beta_init" => { self.if_params.beta_init = value.extract::<f64>()?; },
            "d_init" => { self.if_params.d_init = value.extract::<f64>()?; },
            "dt" => { self.if_params.dt = value.extract::<f64>()?; },
            "exp_dt" => { self.if_params.exp_dt = value.extract::<f64>()?; },
            "current_voltage" => { self.cell_backend.current_voltage = value.extract::<f64>()?; },
            "refractory_count" => { self.cell_backend.refractory_count = value.extract::<f64>()?; },
            "leak_constant" => { self.cell_backend.leak_constant = value.extract::<f64>()?; },
            "integration_constant" => { self.cell_backend.integration_constant = value.extract::<f64>()?; },
            "is_excitatory" => { 
                self.cell_backend.potentiation_type = match value.extract::<bool>()? {
                    true => PotentiationType::Excitatory,
                    false => PotentiationType::Inhibitory,
                }; 
            },
            "neurotransmission_concentration" => { self.cell_backend.neurotransmission_concentration = value.extract::<f64>()?; },
            "neurotransmission_release" => { self.cell_backend.neurotransmission_release = value.extract::<f64>()?; },
            "receptor_density" => { self.cell_backend.receptor_density = value.extract::<f64>()?; },
            "chance_of_releasing" => { self.cell_backend.chance_of_releasing = value.extract::<f64>()?; },
            "dissipation_rate" => { self.cell_backend.dissipation_rate = value.extract::<f64>()?; },
            "chance_of_random_release" => { self.cell_backend.chance_of_random_release = value.extract::<f64>()?; },
            "random_release_concentration" => { self.cell_backend.random_release_concentration = value.extract::<f64>()?; },
            "w_value" => { self.cell_backend.w_value = value.extract::<f64>()?; },
            // "stdp_params" => { self.cell_backend.stdp_params = value.extract::<f64>()?; },
            // "last_firing_time" => { self.cell_backend.last_firing_time = value.extract::<f64>()?; },
            "alpha" => { self.cell_backend.alpha = value.extract::<f64>()?; },
            "beta" => { self.cell_backend.beta = value.extract::<f64>()?; },
            "c" => { self.cell_backend.c = value.extract::<f64>()?; },
            "d" => { self.cell_backend.d = value.extract::<f64>()?; },
            "bayesian_mean" => { self.if_params.bayesian_params.mean = value.extract::<f64>()?; },
            "bayesian_std" => { self.if_params.bayesian_params.std = value.extract::<f64>()?; },
            "bayesian_min" => { self.if_params.bayesian_params.min = value.extract::<f64>()?; },
            "bayesian_max" => { self.if_params.bayesian_params.max = value.extract::<f64>()?; },
            "a_plus" => { self.cell_backend.stdp_params.a_plus = value.extract::<f64>()?; },
            "a_minus" => { self.cell_backend.stdp_params.a_minus = value.extract::<f64>()?; },
            "tau_plus" => { self.cell_backend.stdp_params.tau_plus = value.extract::<f64>()?; },
            "tau_minus" => { self.cell_backend.stdp_params.tau_minus = value.extract::<f64>()?; },
            "stdp_weight_mean" => { self.cell_backend.stdp_params.weight_bayesian_params.mean = value.extract::<f64>()?; },
            "stdp_weight_std" => { self.cell_backend.stdp_params.weight_bayesian_params.std = value.extract::<f64>()?; },
            "stdp_weight_min" => { self.cell_backend.stdp_params.weight_bayesian_params.min = value.extract::<f64>()?; },
            "stdp_weight_max" => { self.cell_backend.stdp_params.weight_bayesian_params.max = value.extract::<f64>()?; },
            "last_firing_time" => { 
                match value.is_none() {
                    false => { self.cell_backend.last_firing_time = None; },
                    true => { self.cell_backend.last_firing_time = Some(value.extract::<usize>()?); }
                }
            }
            _ => { return Err(PyLookupError::new_err("Unknown paramter")) }
        };

        Ok(())
    }

    #[pyo3(signature = (param_name))]
    pub fn get_param(&mut self, py: Python, param_name: &str) -> PyResult<PyObject> {
        let result = match param_name {
            "v_th" => self.if_params.v_th.to_object(py),
            "v_reset" => self.if_params.v_reset.to_object(py),
            "tau_m" => self.if_params.tau_m.to_object(py),
            "g_l" => self.if_params.g_l.to_object(py),
            "v_init" => self.if_params.v_init.to_object(py),
            "e_l" => self.if_params.e_l.to_object(py),
            "tref" => self.if_params.tref.to_object(py),
            "w_init" => self.if_params.w_init.to_object(py),
            "alpha_init" => self.if_params.alpha_init.to_object(py),
            "beta_init" => self.if_params.beta_init.to_object(py),
            "d_init" => self.if_params.d_init.to_object(py),
            "dt" => self.if_params.dt.to_object(py),
            "exp_dt" => self.if_params.exp_dt.to_object(py),
            "current_voltage" => self.cell_backend.current_voltage.to_object(py),
            "refractory_count" => self.cell_backend.refractory_count.to_object(py),
            "leak_constant" => self.cell_backend.leak_constant.to_object(py),
            "integration_constant" => self.cell_backend.integration_constant.to_object(py),
            "is_excitatory" => { 
                match self.cell_backend.potentiation_type {
                    PotentiationType::Excitatory => true,
                    PotentiationType::Inhibitory => false,
                }.to_object(py)
            },
            "neurotransmission_concentration" => self.cell_backend.neurotransmission_concentration.to_object(py),
            "neurotransmission_release" => self.cell_backend.neurotransmission_release.to_object(py),
            "receptor_density" => self.cell_backend.receptor_density.to_object(py),
            "chance_of_releasing" => self.cell_backend.chance_of_releasing.to_object(py),
            "dissipation_rate" => self.cell_backend.dissipation_rate.to_object(py),
            "chance_of_random_release" => self.cell_backend.chance_of_random_release.to_object(py),
            "random_release_concentration" => self.cell_backend.random_release_concentration.to_object(py),
            "w_value" => self.cell_backend.w_value.to_object(py),
            // "stdp_params" => self.cell_backend.stdp_params,
            // "last_firing_time" => self.cell_backend.last_firing_time,
            "alpha" => self.cell_backend.alpha.to_object(py),
            "beta" => self.cell_backend.beta.to_object(py),
            "c" => self.cell_backend.c.to_object(py),
            "d" => self.cell_backend.d.to_object(py),
            "bayesian_mean" => self.if_params.bayesian_params.mean.to_object(py),
            "bayesian_std" => self.if_params.bayesian_params.std.to_object(py),
            "bayesian_min" => self.if_params.bayesian_params.min.to_object(py),
            "bayesian_max" => self.if_params.bayesian_params.max.to_object(py),
            "a_plus" => self.cell_backend.stdp_params.a_plus.to_object(py),
            "a_minus" => self.cell_backend.stdp_params.a_minus.to_object(py),
            "tau_plus" => self.cell_backend.stdp_params.tau_plus.to_object(py),
            "tau_minus" => self.cell_backend.stdp_params.tau_minus.to_object(py),
            "stdp_weight_mean" => self.cell_backend.stdp_params.weight_bayesian_params.mean.to_object(py),
            "stdp_weight_std" => self.cell_backend.stdp_params.weight_bayesian_params.std.to_object(py),
            "stdp_weight_min" => self.cell_backend.stdp_params.weight_bayesian_params.min.to_object(py),
            "stdp_weight_max" => self.cell_backend.stdp_params.weight_bayesian_params.max.to_object(py),
            "last_firing_time" => self.cell_backend.last_firing_time.to_object(py),
            _ => { return Err(PyLookupError::new_err("Unknown paramter")) },
        };

        Ok(result)
    }

    #[getter]
    fn mode(&self) -> IFType {
        self.mode.clone()
    }

    #[setter]
    fn set_mode(&mut self, new_mode: IFType) {
        self.mode = new_mode;
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HodgkinHuxleyModel {
    pub cell_backend: HodgkinHuxleyCell,
}

#[pymethods]
impl HodgkinHuxleyModel {
    #[pyo3(signature = (i))]
    fn iterate(&mut self, i: f64) {
        self.cell_backend.iterate(i);
    }

    #[pyo3(signature = (i, iterations, bayesian=false))]
    pub fn run_static_input(
        &mut self, 
        i: f64, 
        iterations: usize, 
        bayesian: bool
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut voltages: Vec<f64> = vec![];
        let mut m_states: Vec<f64> = vec![];
        let mut n_states: Vec<f64> = vec![];
        let mut h_states: Vec<f64> = vec![];

        for _ in 0..iterations {
            if bayesian {
                self.iterate(
                    i * limited_distr(
                        self.cell_backend.bayesian_params.mean, 
                        self.cell_backend.bayesian_params.std, 
                        self.cell_backend.bayesian_params.min, 
                        self.cell_backend.bayesian_params.max,
                    )
                );
            } else {
                self.iterate(i);
            }

            voltages.push(self.cell_backend.current_voltage);
            m_states.push(self.cell_backend.m.state);
            n_states.push(self.cell_backend.n.state);
            h_states.push(self.cell_backend.h.state);
        }

        (voltages, m_states, n_states, h_states)
    }

    #[getter]
    fn current_voltage(&self) -> f64 {
        self.cell_backend.current_voltage
    }

    #[setter]
    fn set_current_voltage(&mut self, new_current_voltage: f64) {
        self.cell_backend.current_voltage = new_current_voltage;
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.cell_backend.dt
    }

    #[setter]
    fn set_dt(&mut self, new_dt: f64) {
        self.cell_backend.dt = new_dt;
    }

    #[getter]
    fn cm(&self) -> f64 {
        self.cell_backend.cm
    }

    #[setter]
    fn set_cm(&mut self, new_cm: f64) {
        self.cell_backend.cm = new_cm;
    }

    #[getter]
    fn e_na(&self) -> f64 {
        self.cell_backend.e_na
    }

    #[setter]
    fn set_e_na(&mut self, new_e_na: f64) {
        self.cell_backend.e_na = new_e_na;
    }

    #[getter]
    fn e_k(&self) -> f64 {
        self.cell_backend.e_k
    }

    #[setter]
    fn set_e_k(&mut self, new_e_k: f64) {
        self.cell_backend.e_k = new_e_k;
    }

    #[getter]
    fn e_k_leak(&self) -> f64 {
        self.cell_backend.e_k_leak
    }

    #[setter]
    fn set_e_k_leak(&mut self, new_e_k_leak: f64) {
        self.cell_backend.e_k_leak = new_e_k_leak;
    }

    #[getter]
    fn g_na(&self) -> f64 {
        self.cell_backend.g_na
    }

    #[setter]
    fn set_g_na(&mut self, new_g_na: f64) {
        self.cell_backend.g_na = new_g_na;
    }

    #[getter]
    fn g_k(&self) -> f64 {
        self.cell_backend.g_k
    }

    #[setter]
    fn set_g_k(&mut self, new_g_k: f64) {
        self.cell_backend.g_k = new_g_k;
    }

    #[getter]
    fn g_k_leak(&self) -> f64 {
        self.cell_backend.g_k_leak
    }

    #[setter]
    fn set_g_k_leak(&mut self, new_g_k_leak: f64) {
        self.cell_backend.g_k_leak = new_g_k_leak;
    }

    #[pyo3(signature = (gate, gate_parameter))]
    fn get_gates_params(&mut self, gate: &str, gate_parameter: &str) -> PyResult<f64> {
        let result = match gate.to_ascii_lowercase().as_str() {
            "m" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => self.cell_backend.m.alpha,
                    "beta" => self.cell_backend.m.beta,
                    "state" => self.cell_backend.m.state,
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            },
            "n" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => self.cell_backend.n.alpha,
                    "beta" => self.cell_backend.n.beta,
                    "state" => self.cell_backend.n.state,
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            },
            "h" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => self.cell_backend.h.alpha,
                    "beta" => self.cell_backend.h.beta,
                    "state" => self.cell_backend.h.state,
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            }
            _ => { return Err(PyLookupError::new_err("Unknown gate")) }
        };

        Ok(result)
    }

    #[pyo3(signature = (gate, gate_parameter, value))]
    fn change_gates_params(&mut self, gate: &str, gate_parameter: &str, value: f64) -> PyResult<()> {
        match gate.to_ascii_lowercase().as_str() {
            "m" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => { self.cell_backend.m.alpha = value; },
                    "beta" => { self.cell_backend.m.beta = value; },
                    "state" => { self.cell_backend.m.state = value; },
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            },
            "n" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => { self.cell_backend.n.alpha = value; },
                    "beta" => { self.cell_backend.n.beta = value; },
                    "state" => { self.cell_backend.n.state = value; },
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            },
            "h" => {
                match gate_parameter.to_ascii_lowercase().as_str() {
                    "alpha" => { self.cell_backend.h.alpha = value; },
                    "beta" => { self.cell_backend.h.beta = value; },
                    "state" => { self.cell_backend.h.state = value; },
                    _ => { return Err(PyLookupError::new_err("Unknown gate paramter")) }
                }
            }
            _ => { return Err(PyLookupError::new_err("Unknown gate")) }
        };

        Ok(())
    }
}
