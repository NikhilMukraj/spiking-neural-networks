use pyo3::prelude::*;
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
    
    #[getter]
    fn v_th(&self) -> f64 {
        self.if_params.v_th
    }

    #[setter]
    fn set_v_th(&mut self, new_v_th: f64) {
        self.if_params.v_th = new_v_th;
    }

    #[getter]
    fn v_reset(&self) -> f64 {
        self.if_params.v_reset
    }

    #[setter]
    fn set_v_reset(&mut self, new_v_reset: f64) {
        self.if_params.v_reset = new_v_reset;
    }

    #[getter]
    fn tau_m(&self) -> f64 {
        self.if_params.tau_m
    }

    #[setter]
    fn set_tau_m(&mut self, new_tau_m: f64) {
        self.if_params.tau_m = new_tau_m;
    }

    #[getter]
    fn g_l(&self) -> f64 {
        self.if_params.g_l
    }

    #[setter]
    fn set_g_l(&mut self, new_g_l: f64) {
        self.if_params.g_l = new_g_l;
    }

    #[getter]
    fn v_init(&self) -> f64 {
        self.if_params.v_init
    }

    #[setter]
    fn set_v_init(&mut self, new_v_init: f64) {
        self.if_params.v_init = new_v_init;
    }

    #[getter]
    fn e_l(&self) -> f64 {
        self.if_params.e_l
    }

    #[setter]
    fn set_e_l(&mut self, new_e_l: f64) {
        self.if_params.e_l = new_e_l;
    }

    #[getter]
    fn tref(&self) -> f64 {
        self.if_params.tref
    }

    #[setter]
    fn set_tref(&mut self, new_tref: f64) {
        self.if_params.tref = new_tref;
    }

    #[getter]
    fn w_init(&self) -> f64 {
        self.if_params.w_init
    }

    #[setter]
    fn set_w_init(&mut self, new_w_init: f64) {
        self.if_params.w_init = new_w_init;
    }

    #[getter]
    fn alpha_init(&self) -> f64 {
        self.if_params.alpha_init
    }

    #[setter]
    fn set_alpha_init(&mut self, new_alpha_init: f64) {
        self.if_params.alpha_init = new_alpha_init;
    }

    #[getter]
    fn beta_init(&self) -> f64 {
        self.if_params.beta_init
    }

    #[setter]
    fn set_beta_init(&mut self, new_beta_init: f64) {
        self.if_params.beta_init = new_beta_init;
    }

    #[getter]
    fn d_init(&self) -> f64 {
        self.if_params.d_init
    }

    #[setter]
    fn set_d_init(&mut self, new_d_init: f64) {
        self.if_params.d_init = new_d_init;
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.if_params.dt
    }

    #[setter]
    fn set_dt(&mut self, new_dt: f64) {
        self.if_params.dt = new_dt;
    }

    #[getter]
    fn exp_dt(&self) -> f64 {
        self.if_params.exp_dt
    }

    #[setter]
    fn set_exp_dt(&mut self, new_exp_dt: f64) {
        self.if_params.exp_dt = new_exp_dt;
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
    fn refractory_count(&self) -> f64 {
        self.cell_backend.refractory_count
    }

    #[setter]
    fn set_refractory_count(&mut self, new_refractory_count: f64) {
        self.cell_backend.refractory_count = new_refractory_count;
    }

    #[getter]
    fn leak_constant(&self) -> f64 {
        self.cell_backend.leak_constant
    }

    #[setter]
    fn set_leak_constant(&mut self, new_leak_constant: f64) {
        self.cell_backend.leak_constant = new_leak_constant;
    }

    #[getter]
    fn integration_constant(&self) -> f64 {
        self.cell_backend.integration_constant
    }

    #[setter]
    fn set_integration_constant(&mut self, new_integration_constant: f64) {
        self.cell_backend.integration_constant = new_integration_constant;
    }

    // #[getter]
    // fn potentiation_type(&self) -> f64 {
    //     self.cell_backend.potentiation_type
    // }

    // #[setter]
    // fn set_potentiation_type(&mut self, new_potentiation_type: f64) {
    //     self.cell_backend.potentiation_type = new_potentiation_type;
    // }

    #[getter]
    fn neurotransmission_concentration(&self) -> f64 {
        self.cell_backend.neurotransmission_concentration
    }

    #[setter]
    fn set_neurotransmission_concentration(&mut self, new_neurotransmission_concentration: f64) {
        self.cell_backend.neurotransmission_concentration = new_neurotransmission_concentration;
    }

    #[getter]
    fn neurotransmission_release(&self) -> f64 {
        self.cell_backend.neurotransmission_release
    }

    #[setter]
    fn set_neurotransmission_release(&mut self, new_neurotransmission_release: f64) {
        self.cell_backend.neurotransmission_release = new_neurotransmission_release;
    }

    #[getter]
    fn receptor_density(&self) -> f64 {
        self.cell_backend.receptor_density
    }

    #[setter]
    fn set_receptor_density(&mut self, new_receptor_density: f64) {
        self.cell_backend.receptor_density = new_receptor_density;
    }

    #[getter]
    fn chance_of_releasing(&self) -> f64 {
        self.cell_backend.chance_of_releasing
    }

    #[setter]
    fn set_chance_of_releasing(&mut self, new_chance_of_releasing: f64) {
        self.cell_backend.chance_of_releasing = new_chance_of_releasing;
    }

    #[getter]
    fn dissipation_rate(&self) -> f64 {
        self.cell_backend.dissipation_rate
    }

    #[setter]
    fn set_dissipation_rate(&mut self, new_dissipation_rate: f64) {
        self.cell_backend.dissipation_rate = new_dissipation_rate;
    }

    #[getter]
    fn chance_of_random_release(&self) -> f64 {
        self.cell_backend.chance_of_random_release
    }

    #[setter]
    fn set_chance_of_random_release(&mut self, new_chance_of_random_release: f64) {
        self.cell_backend.chance_of_random_release = new_chance_of_random_release;
    }

    #[getter]
    fn random_release_concentration(&self) -> f64 {
        self.cell_backend.random_release_concentration
    }

    #[setter]
    fn set_random_release_concentration(&mut self, new_random_release_concentration: f64) {
        self.cell_backend.random_release_concentration = new_random_release_concentration;
    }

    #[getter]
    fn w_value(&self) -> f64 {
        self.cell_backend.w_value
    }

    #[setter]
    fn set_w_value(&mut self, new_w_value: f64) {
        self.cell_backend.w_value = new_w_value;
    }

    // #[getter]
    // fn stdp_params(&self) -> f64 {
    //     self.cell_backend.stdp_params
    // }

    // #[setter]
    // fn set_stdp_params(&mut self, new_stdp_params: f64) {
    //     self.cell_backend.stdp_params = new_stdp_params;
    // }

    // #[getter]
    // fn last_firing_time(&self) -> f64 {
    //     self.cell_backend.last_firing_time
    // }

    // #[setter]
    // fn set_last_firing_time(&mut self, new_last_firing_time: f64) {
    //     self.cell_backend.last_firing_time = new_last_firing_time;
    // }

    #[getter]
    fn alpha(&self) -> f64 {
        self.cell_backend.alpha
    }

    #[setter]
    fn set_alpha(&mut self, new_alpha: f64) {
        self.cell_backend.alpha = new_alpha;
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.cell_backend.beta
    }

    #[setter]
    fn set_beta(&mut self, new_beta: f64) {
        self.cell_backend.beta = new_beta;
    }

    #[getter]
    fn c(&self) -> f64 {
        self.cell_backend.c
    }

    #[setter]
    fn set_c(&mut self, new_c: f64) {
        self.cell_backend.c = new_c;
    }

    #[getter]
    fn d(&self) -> f64 {
        self.cell_backend.d
    }

    #[setter]
    fn set_d(&mut self, new_d: f64) {
        self.cell_backend.d = new_d;
    }
}
