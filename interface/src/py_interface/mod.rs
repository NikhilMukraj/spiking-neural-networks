use std::io::{Result, Error, ErrorKind};
use pyo3::prelude::*;
use pyo3::exceptions::PyLookupError;
use crate::neuron::{IFParameters, IFType, Cell};

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
pub struct IFCell {
    mode: IFType,
    cell_backend: Cell,
    if_params: IFParameters,
}

#[pymethods]
impl IFCell {
    #[new]
    fn new() -> Self {
        IFCell { 
            mode: IFType::Basic,
            cell_backend: Cell::default(),
            if_params: IFParameters::default(),
        }
    }

    #[pyo3(signature = (i))]
    pub fn get_dv_change_and_spike(&mut self, i: f64) -> (f64, bool) {
        self.cell_backend.get_dv_change_and_spike(&self.if_params, i)
    }

    #[pyo3(signature = ())]
    pub fn apply_dw_change_and_get_spike(&mut self) -> bool {
        self.cell_backend.apply_dw_change_and_get_spike(&self.if_params)
    }

    #[pyo3(signature = (i))]
    pub fn adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.adaptive_get_dv_change(&self.if_params, i)
    }

    #[pyo3(signature = (i))]
    pub fn exp_adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.exp_adaptive_get_dv_change(&self.if_params, i)
    }

    #[pyo3(signature = ())]
    pub fn izhikevich_apply_dw_and_get_spike(&mut self) -> bool {
        self.cell_backend.izhikevich_apply_dw_and_get_spike(&self.if_params)
    }

    #[pyo3(signature = (i))]
    pub fn izhikevich_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.izhikevich_get_dv_change(&self.if_params, i)
    }

    #[pyo3(signature = (i))]
    pub fn izhikevich_leaky_get_dv_change(&mut self, i: f64) -> f64 {
        self.cell_backend.izhikevich_leaky_get_dv_change(&self.if_params, i)
    }

    #[pyo3(signature = (i))]
    pub fn iterate_and_return_spike(&mut self, i: f64) -> bool {
        match self.mode {
            IFType::Basic => {
                let (dv, is_spiking) = self.get_dv_change_and_spike(i);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::Adaptive => {
                let is_spiking = self.apply_dw_change_and_get_spike();
                let dv = self.adaptive_get_dv_change(i);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::AdaptiveExponential => {
                let is_spiking = self.apply_dw_change_and_get_spike();
                let dv = self.exp_adaptive_get_dv_change(i);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::Izhikevich => {
                let is_spiking = self.izhikevich_apply_dw_and_get_spike();
                let dv = self.izhikevich_get_dv_change(i);
                self.cell_backend.current_voltage += dv;

                is_spiking
            },
            IFType::IzhikevichLeaky => {
                let is_spiking = self.izhikevich_apply_dw_and_get_spike();
                let dv = self.izhikevich_leaky_get_dv_change(i);
                self.cell_backend.current_voltage += dv;

                is_spiking
            }
        }
    }

    #[pyo3(signature = (is_spiking))]
    pub fn determine_neurotransmitter_concentration(&mut self, is_spiking: bool) {
        self.cell_backend.determine_neurotransmitter_concentration(is_spiking);
    }
    
    #[pyo3(signature = (param_name, value=None))]
    pub fn param(&mut self, param_name: &str, value: Option<f64>) -> PyResult<Option<f64>> {
        match value {
            Some(new_value) => {
                match param_name {
                    "v_th" => { self.if_params.v_th = new_value; },
                    "v_reset" => { self.if_params.v_reset = new_value; },
                    "tau_m" => { self.if_params.tau_m = new_value; },
                    "g_l" => { self.if_params.g_l = new_value; },
                    "v_init" => { self.if_params.v_init = new_value; },
                    "e_l" => { self.if_params.e_l = new_value; },
                    "tref" => { self.if_params.tref = new_value; },
                    "w_init" => { self.if_params.w_init = new_value; },
                    "alpha_init" => { self.if_params.alpha_init = new_value; },
                    "beta_init" => { self.if_params.beta_init = new_value; },
                    "d_init" => { self.if_params.d_init = new_value; },
                    "dt" => { self.if_params.dt = new_value; },
                    "exp_dt" => { self.if_params.exp_dt = new_value; },
                    "current_voltage" => { self.cell_backend.current_voltage = new_value; },
                    "refractory_count" => { self.cell_backend.refractory_count = new_value; },
                    "leak_constant" => { self.cell_backend.leak_constant = new_value; },
                    "integration_constant" => { self.cell_backend.integration_constant = new_value; },
                    // "potentiation_type" => { self.cell_backend.potentiation_type = new_value; },
                    "neurotransmission_concentration" => { self.cell_backend.neurotransmission_concentration = new_value; },
                    "neurotransmission_release" => { self.cell_backend.neurotransmission_release = new_value; },
                    "receptor_density" => { self.cell_backend.receptor_density = new_value; },
                    "chance_of_releasing" => { self.cell_backend.chance_of_releasing = new_value; },
                    "dissipation_rate" => { self.cell_backend.dissipation_rate = new_value; },
                    "chance_of_random_release" => { self.cell_backend.chance_of_random_release = new_value; },
                    "random_release_concentration" => { self.cell_backend.random_release_concentration = new_value; },
                    "w_value" => { self.cell_backend.w_value = new_value; },
                    // "stdp_params" => { self.cell_backend.stdp_params = new_value; },
                    // "last_firing_time" => { self.cell_backend.last_firing_time = new_value; },
                    "alpha" => { self.cell_backend.alpha = new_value; },
                    "beta" => { self.cell_backend.beta = new_value; },
                    "c" => { self.cell_backend.c = new_value; },
                    "d" => { self.cell_backend.d = new_value; },
                    _ => { return Err(PyLookupError::new_err("Unknown paramter")) }
                };

                Ok(None)
            },
            None => {
                let return_value = match param_name {
                    "v_th" => self.if_params.v_th,
                    "v_reset" => self.if_params.v_reset,
                    "tau_m" => self.if_params.tau_m,
                    "g_l" => self.if_params.g_l,
                    "v_init" => self.if_params.v_init,
                    "e_l" => self.if_params.e_l,
                    "tref" => self.if_params.tref,
                    "w_init" => self.if_params.w_init,
                    "alpha_init" => self.if_params.alpha_init,
                    "beta_init" => self.if_params.beta_init,
                    "d_init" => self.if_params.d_init,
                    "dt" => self.if_params.dt,
                    "exp_dt" => self.if_params.exp_dt,
                    "current_voltage" => self.cell_backend.current_voltage,
                    "refractory_count" => self.cell_backend.refractory_count,
                    "leak_constant" => self.cell_backend.leak_constant,
                    "integration_constant" => self.cell_backend.integration_constant,
                    // "potentiation_type" => self.cell_backend.potentiation_type,
                    "neurotransmission_concentration" => self.cell_backend.neurotransmission_concentration,
                    "neurotransmission_release" => self.cell_backend.neurotransmission_release,
                    "receptor_density" => self.cell_backend.receptor_density,
                    "chance_of_releasing" => self.cell_backend.chance_of_releasing,
                    "dissipation_rate" => self.cell_backend.dissipation_rate,
                    "chance_of_random_release" => self.cell_backend.chance_of_random_release,
                    "random_release_concentration" => self.cell_backend.random_release_concentration,
                    "w_value" => self.cell_backend.w_value,
                    // "stdp_params" => self.cell_backend.stdp_params,
                    // "last_firing_time" => self.cell_backend.last_firing_time,
                    "alpha" => self.cell_backend.alpha,
                    "beta" => self.cell_backend.beta,
                    "c" => self.cell_backend.c,
                    "d" => self.cell_backend.d,
                    _ => { return Err(PyLookupError::new_err("Unknown paramter")) }
                };

                Ok(Some(return_value))
            }
        }
    }
}
