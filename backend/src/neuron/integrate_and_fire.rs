use std::collections::HashMap;
#[path ="iterate_and_spike/mod.rs"]
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, Potentiation, BayesianFactor, LastFiringTime, STDP,
    IterateAndSpike, BayesianParameters, STDPParameters, PotentiationType,
    Neurotransmitters, NeurotransmitterType, NeurotransmitterKinetics, LigandGatedChannels,
    impl_current_voltage_with_neurotransmitter,
    impl_gap_conductance_with_neurotransmitter,
    impl_potentiation_with_neurotransmitter,
    impl_bayesian_factor_with_neurotransmitter,
    impl_last_firing_time_with_neurotransmitter,
    impl_stdp_with_neurotransmitter,
};

#[derive(Debug, Clone)]
pub struct LeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics> {
    pub current_voltage: f64,
    pub v_th: f64,
    pub v_reset: f64,
    pub v_init: f64,
    pub refractory_count: f64,
    pub tref: f64,
    pub leak_constant: f64,
    pub integration_constant: f64,
    pub gap_conductance: f64,
    pub e_l: f64,
    pub g_l: f64,
    pub tau_m: f64,
    pub c_m: f64,
    pub dt: f64,
    pub last_firing_time: Option<usize>,
    pub potentiation_type: PotentiationType,
    pub stdp_params: STDPParameters,
    pub bayesian_params: BayesianParameters,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels,
}

impl_current_voltage_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);
impl_gap_conductance_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);
impl_potentiation_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);
impl_bayesian_factor_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);
impl_last_firing_time_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);
impl_stdp_with_neurotransmitter!(LeakyIntegrateAndFireNeuron);

macro_rules! impl_default_neurotransmitter_methods {
    () => {
        type T = T;

        fn get_ligand_gates(&self) -> &LigandGatedChannels {
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

impl<T: NeurotransmitterKinetics> Default for LeakyIntegrateAndFireNeuron<T> {
    fn default() -> Self {
        LeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            last_firing_time: None,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            dt: 0.1, // simulation time step (ms)
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            bayesian_params: BayesianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics> LeakyIntegrateAndFireNeuron<T> {
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

    pub fn run_static_input(&mut self, input: f64, bayesian: bool, iterations: usize, ) -> Vec<f64> {
        let mut voltages: Vec<f64> = vec![self.current_voltage];

        for _ in 0..iterations {
            let _is_spiking = if bayesian {
                self.iterate_and_spike(self.get_bayesian_factor() * input)
            } else {
                self.iterate_and_spike(input)
            };

            voltages.push(self.current_voltage);
        }

        voltages
    }
}

impl<T: NeurotransmitterKinetics> IterateAndSpike for LeakyIntegrateAndFireNeuron<T> {
    impl_default_neurotransmitter_methods!();

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let dv = self.get_basic_dv_change(input_current);
        self.current_voltage += dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        let is_spiking = self.basic_handle_spiking();

        is_spiking
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

        let is_spiking = self.basic_handle_spiking();

        is_spiking
    }
}
