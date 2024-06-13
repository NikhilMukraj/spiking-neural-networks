use std::{
    fs::File, 
    io::{BufWriter, Error, ErrorKind, Result, Write},
    collections::HashMap,
};
use crate::distribution;
use distribution::limited_distr;
pub mod integrate_and_fire;
pub mod hodgkin_huxley;
pub mod attractors;
pub mod spike_train;
use spike_train::{SpikeTrain, NeuralRefractoriness};
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
    impl_necessary_iterate_and_spike_traits,
};


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
