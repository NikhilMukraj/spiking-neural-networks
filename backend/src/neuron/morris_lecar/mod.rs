//! An implementation of the Morris Lecar neuron.

use iterate_and_spike_traits::IterateAndSpikeBase;
use super::iterate_and_spike::{
    CurrentVoltage, GapConductance, GaussianFactor, LastFiringTime, STDP, IsSpiking,
    Potentiation, IterateAndSpike, PotentiationType, GaussianParameters, STDPParameters,
    LigandGatedChannels, Neurotransmitters, NeurotransmitterKinetics, ReceptorKinetics,
    NeurotransmitterConcentrations, DestexheNeurotransmitter, DestexheReceptor,
};


#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct MorrisLecarNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Initial voltage value (mV)
    pub v_init: f32,
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32,
    /// Conductance of leak channel (nS)
    pub g_l: f32,
    /// Conductance of calcium channel (nS)
    pub g_ca: f32,
    /// Conductance of potassium channel (nS)
    pub g_k: f32,
    /// Leak channel reversal potential (mV)
    pub v_l: f32,
    /// Calcium channel reversal potential (mV)
    pub v_ca: f32,
    /// Potassium channel reversal potential (mV)
    pub v_k: f32,
    /// Calcium gating variable
    pub m_ss: f32,
    /// Potassium gating variable
    pub n: f32,
    /// Potassium gating variable modifier
    pub n_ss: f32,
    /// Decay of potassium gating variable
    pub t_n: f32,
    /// Tuning parameter for gating variable
    pub v_1: f32,
    /// Tuning parameter for gating variable
    pub v_2: f32,
    /// Tuning parameter for gating variable
    pub v_3: f32,
    /// Tuning parameter for gating variable
    pub v_4: f32,
    /// Reference frequency
    pub phi: f32,
    /// Membrane capacitance (nF)
    pub c_m: f32,
    /// Timestep in (ms)
    pub dt: f32,
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Whether the voltage was increasing in the last step
    pub was_increasing: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Potentiation type of neuron
    pub potentiation_type: PotentiationType,
    /// STDP parameters
    pub stdp_params: STDPParameters,
    /// Parameters used in generating noise
    pub gaussian_params: GaussianParameters,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for MorrisLecarNeuron<T, R> {
    fn default() -> Self {
        MorrisLecarNeuron {
            current_voltage: -70.,
            v_init: -70.,
            v_th: 25.,
            gap_conductance: 10.,
            g_l: 2.,
            g_ca: 4.,
            g_k: 8.,
            v_l: -60.,
            v_ca: 120.,
            v_k: -84.,
            m_ss: 0.,
            n_ss: 0.,
            t_n: 0.,
            n: 0.,
            v_1: -1.2,
            v_2: 18.,
            v_3: 12.,
            v_4: 17.4,
            phi: 0.067,
            c_m: 20.,
            dt: 0.01,
            is_spiking: false,
            was_increasing: false,
            last_firing_time: None,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            gaussian_params: GaussianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::<R>::default(),
        }
    }
}

impl MorrisLecarNeuron<DestexheNeurotransmitter, DestexheReceptor> {
    /// Returns the default implementation of the Morris Lecar Neuron
    pub fn default_impl() -> Self {
        MorrisLecarNeuron::default()
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> MorrisLecarNeuron<T, R> {
    fn update_m_ss(&mut self) {
        self.m_ss = 0.5 * (1. + ((self.current_voltage - self.v_1) / self.v_2).tanh())
    }

    fn update_n_ss(&mut self) {
        self.n_ss = 0.5 * (1. + ((self.current_voltage - self.v_3) / self.v_4).tanh())
    }

    fn update_t_n(&mut self) {
        self.t_n = 1. / (self.phi * ((self.current_voltage - self.v_3) / (2. * self.v_4)).cosh())
    }

    fn get_n_change(&mut self) -> f32 {
        ((self.n_ss - self.n) / self.t_n) * self.dt
    }

    fn update_gating_variables(&mut self) {
        self.update_m_ss();
        self.update_n_ss();
        self.update_t_n();

        self.n += self.get_n_change();
    }

    fn get_dv_change(&mut self, i: f32) -> f32 {
        (i - (self.g_l * (self.current_voltage - self.v_l)) - 
        (self.g_ca * self.m_ss * (self.current_voltage - self.v_ca)) - 
        (self.g_k * self.n * (self.current_voltage - self.v_k)))
        * (self.dt / self.c_m)
    }

    fn handle_spiking(&mut self, last_voltage: f32) -> bool {
        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for MorrisLecarNeuron<T, R> {
    type T = T;
    type R = R;

    fn get_ligand_gates(&self) -> &LigandGatedChannels<Self::R> {
        &self.ligand_gates
    }

    fn get_neurotransmitters(&self) -> &Neurotransmitters<Self::T> {
        &self.synaptic_neurotransmitters
    }

    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        self.update_gating_variables();

        let last_voltage = self.current_voltage;
        self.current_voltage += self.get_dv_change(input_current);

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.handle_spiking(last_voltage)
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: Option<&NeurotransmitterConcentrations>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);
        
        self.update_gating_variables();

        let last_voltage = self.current_voltage;
        let receptor_current = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
        self.current_voltage += self.get_dv_change(input_current) + receptor_current;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.handle_spiking(last_voltage)
    }
}
