use std::collections::HashMap;
use super::{ 
    iterate_and_spike::ReceptorKinetics, BayesianFactor, BayesianParameters, 
    CurrentVoltage, GapConductance, IterateAndSpike, LastFiringTime, 
    LigandGatedChannels, NeurotransmitterKinetics, NeurotransmitterType, 
    Neurotransmitters, Potentiation, PotentiationType, STDPParameters, STDP,
    impl_bayesian_factor_with_neurotransmitter, 
    impl_current_voltage_with_neurotransmitter, 
    impl_gap_conductance_with_neurotransmitter, 
    impl_last_firing_time_with_neurotransmitter, 
    impl_necessary_iterate_and_spike_traits, 
    impl_potentiation_with_neurotransmitter, 
    impl_stdp_with_neurotransmitter, 
};


pub fn run_static_input_integrate_and_fire<T: IterateAndSpike>(
    cell: &mut T, 
    input: f64, 
    bayesian: bool, 
    iterations: usize
) -> Vec<f64> {
    let mut voltages: Vec<f64> = vec![cell.get_current_voltage()];

    for _ in 0..iterations {
        let _is_spiking = if bayesian {
            cell.iterate_and_spike(cell.get_bayesian_factor() * input)
        } else {
            cell.iterate_and_spike(input)
        };

        voltages.push(cell.get_current_voltage());
    }

    voltages
}

macro_rules! impl_default_neurotransmitter_methods {
    () => {
        type T = T;
        type R = R;

        fn get_ligand_gates(&self) -> &LigandGatedChannels<R> {
            &self.ligand_gates
        }
    
        fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
            &self.synaptic_neurotransmitters
        }
    
        fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    }
}

pub(crate) use impl_default_neurotransmitter_methods;

#[derive(Debug, Clone)]
pub struct LeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64, // membrane potential (mV)
    pub v_th: f64, // voltage threshold (mV)
    pub v_reset: f64, // voltage reset value (mV)
    pub v_init: f64, // voltage initialization value (mV)
    pub refractory_count: f64, // counter for refractory period
    pub tref: f64, // total refractory period
    pub leak_constant: f64, // leak constant
    pub integration_constant: f64, // input value modifier
    pub gap_conductance: f64, // controls conductance of input gap junctions
    pub e_l: f64, // leak reversal potential (mV)
    pub g_l: f64, // leak conductance (nS)
    pub tau_m: f64, // membrane time constant (ms)
    pub c_m: f64, // membrane capacitance (nF)
    pub dt: f64, // time step (ms)
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_necessary_iterate_and_spike_traits!(LeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for LeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        LeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            dt: 0.1, // simulation time step (ms)
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> LeakyIntegrateAndFireNeuron<T, R> {
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
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for LeakyIntegrateAndFireNeuron<T, R> {
    impl_default_neurotransmitter_methods!();

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let dv = self.get_basic_dv_change(input_current);
        self.current_voltage += dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.basic_handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);

        let dv = self.get_basic_dv_change(input_current);
        let neurotransmitter_dv = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.basic_handle_spiking()
    }
}

macro_rules! impl_iterate_and_spike {
    ($name:ident, $dv_method:ident, $dw_method:ident, $handle_spiking:ident) => {
        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for $name<T, R> {
            impl_default_neurotransmitter_methods!();

            fn iterate_and_spike(&mut self, input_current: f64) -> bool {
                let dv = self.$dv_method(input_current);
                let dw = self.$dw_method();

                self.current_voltage += dv;
                self.w_value += dw;

                self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

                self.$handle_spiking()
            }

            fn iterate_with_neurotransmitter_and_spike(
                &mut self, 
                input_current: f64, 
                t_total: Option<&HashMap<NeurotransmitterType, f64>>,
            ) -> bool {
                self.ligand_gates.update_receptor_kinetics(t_total);
                self.ligand_gates.set_receptor_currents(self.current_voltage);

                let dv = self.$dv_method(input_current);
                let dw = self.$dw_method();
                let neurotransmitter_dv = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

                self.current_voltage += dv + neurotransmitter_dv;
                self.w_value += dw;

                self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

                self.$handle_spiking()
            }
        }
    };
}

#[derive(Debug, Clone)]
pub struct AdaptiveLeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64, // membrane potential (mV)
    pub v_th: f64, // voltage threshold (mV)
    pub v_reset: f64, // voltage reset value (mV)
    pub v_init: f64, // voltage initialization value (mV)
    pub refractory_count: f64, // counter for refractory period
    pub tref: f64, // total refractory period
    pub alpha: f64, // arbitrary paramterization value
    pub beta: f64, // arbitrary paramterization value
    pub w_value: f64, // adaptive value
    pub w_init: f64, // adaptive value initialization
    pub leak_constant: f64, // leak constant
    pub integration_constant: f64, // input value modifier
    pub gap_conductance: f64, // controls conductance of input gap junctions
    pub e_l: f64, // leak reversal potential (mV)
    pub g_l: f64, // leak conductance (nS)
    pub tau_m: f64, // membrane time constant (ms)
    pub c_m: f64, // membrane capacitance (nF)
    pub dt: f64, // time step (ms)
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_necessary_iterate_and_spike_traits!(AdaptiveLeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for AdaptiveLeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        AdaptiveLeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 6.0,
            beta: 10.0,
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
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

macro_rules! impl_adaptive_default_methods {
    () => {
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
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> AdaptiveLeakyIntegrateAndFireNeuron<T, R> {
    pub fn adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m);

        dv
    }

    impl_adaptive_default_methods!();
}

impl_iterate_and_spike!(
    AdaptiveLeakyIntegrateAndFireNeuron, 
    adaptive_get_dv_change, 
    adaptive_get_dw_change,
    adaptive_handle_spiking
);

#[derive(Debug, Clone)]
pub struct AdaptiveExpLeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64, // membrane potential (mV)
    pub v_th: f64, // voltage threshold (mV)
    pub v_reset: f64, // voltage reset value (mV)
    pub v_init: f64, // voltage initialization value (mV)
    pub refractory_count: f64, // counter for refractory period
    pub tref: f64, // total refractory period
    pub alpha: f64, // arbitrary paramterization value
    pub beta: f64, // arbitrary paramterization value
    pub slope_factor: f64, // controls steepness
    pub w_value: f64, // adaptive value
    pub w_init: f64, // adaptive value initialization
    pub leak_constant: f64, // leak constant
    pub integration_constant: f64, // input value modifier
    pub gap_conductance: f64, // controls conductance of input gap junctions
    pub e_l: f64, // leak reversal potential (mV)
    pub g_l: f64, // leak conductance (nS)
    pub tau_m: f64, // membrane time constant (ms)
    pub c_m: f64, // membrane capacitance (nF)
    pub dt: f64, // time step (ms)
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_necessary_iterate_and_spike_traits!(AdaptiveExpLeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for AdaptiveExpLeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        AdaptiveExpLeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 6.0,
            beta: 10.0,
            slope_factor: 1.,
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
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> AdaptiveExpLeakyIntegrateAndFireNeuron<T, R> {
    pub fn exp_adaptive_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.slope_factor * ((self.current_voltage - self.v_th) / self.slope_factor).exp()) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m);

        dv
    }

    impl_adaptive_default_methods!();
}

impl_iterate_and_spike!(
    AdaptiveExpLeakyIntegrateAndFireNeuron, 
    exp_adaptive_get_dv_change, 
    adaptive_get_dw_change,
    adaptive_handle_spiking
);

#[derive(Debug, Clone)]
pub struct IzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64, // membrane potential (mV)
    pub v_th: f64, // voltage threshold (mV)
    pub v_init: f64, // voltage initialization value (mV)
    pub alpha: f64, // controls speed
    pub beta: f64, // controls sensitivity to adaptive value
    pub c: f64, // after spike reset value for voltage
    pub d: f64, // after spike reset value for adaptive value
    pub w_value: f64, // adaptive value
    pub w_init: f64, // adaptive value initialization
    pub gap_conductance: f64, // controls conductance of input gap junctions
    pub tau_m: f64, // membrane time constant (ms)
    pub c_m: f64, // membrane capacitance (nF)
    pub dt: f64, // time step (ms)
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_necessary_iterate_and_spike_traits!(IzhikevichNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for IzhikevichNeuron<T, R> {
    fn default() -> Self {
        IzhikevichNeuron {
            current_voltage: -75., 
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 0.02,
            beta: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            tau_m: 1., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -65., // initial potential (mV)
            w_init: 30., // initial w value
            dt: 0.1, // simulation time step (ms)
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

macro_rules! impl_izhikevich_default_methods {
    () => {
        pub fn izhikevich_get_dw_change(&self) -> f64 {
            let dw = (
                self.alpha * (self.beta * self.current_voltage - self.w_value)
            ) * (self.dt / self.tau_m);
    
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
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IzhikevichNeuron<T, R> {
    impl_izhikevich_default_methods!();

    pub fn izhikevich_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (self.dt / self.c_m);

        dv
    }
}

impl_iterate_and_spike!(
    IzhikevichNeuron, 
    izhikevich_get_dv_change, 
    izhikevich_get_dw_change,
    izhikevich_handle_spiking
);

#[derive(Debug, Clone)]
pub struct LeakyIzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64, // membrane potential (mV)
    pub v_th: f64, // voltage threshold (mV)
    pub v_init: f64, // voltage initialization value (mV)
    pub alpha: f64, // controls speed
    pub beta: f64, // controls sensitivity to adaptive value
    pub c: f64, // after spike reset value for voltage
    pub d: f64, // after spike reset value for adaptive value
    pub w_value: f64, // adaptive value
    pub w_init: f64, // adaptive value initialization
    pub e_l: f64, // leak reversal potential (mV)
    pub gap_conductance: f64, // controls conductance of input gap junctions
    pub tau_m: f64, // membrane time constant (ms)
    pub c_m: f64, // membrane capacitance (nF)
    pub dt: f64, // time step (ms)
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_necessary_iterate_and_spike_traits!(LeakyIzhikevichNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for LeakyIzhikevichNeuron<T, R> {
    fn default() -> Self {
        LeakyIzhikevichNeuron {
            current_voltage: -75., 
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 0.02,
            beta: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -65., // initial potential (mV)
            e_l: -65., // leak reversal potential (mV)
            w_init: 30., // initial w value
            dt: 0.1, // simulation time step (ms)
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> LeakyIzhikevichNeuron<T, R> {
    impl_izhikevich_default_methods!();

    pub fn izhikevich_leaky_get_dv_change(&mut self, i: f64) -> f64 {
        let dv = (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - 
            self.w_value * (self.current_voltage - self.e_l) + i
        ) * (self.dt / self.c_m);

        dv
    }
}

impl_iterate_and_spike!(
    LeakyIzhikevichNeuron, 
    izhikevich_leaky_get_dv_change, 
    izhikevich_get_dw_change,
    izhikevich_handle_spiking
);
