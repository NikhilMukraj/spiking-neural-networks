//! An implementation of the Morris Lecar neuron.

use iterate_and_spike_traits::IterateAndSpikeBase;
use super::intermediate_delegate::NeurotransmittersIntermediate;
use super::iterate_and_spike::{
    CurrentVoltage, DestexheNeurotransmitter, DestexheReceptor, GapConductance, 
    IonotropicReceptorNeurotransmitterType, IsSpiking, IterateAndSpike, LastFiringTime, 
    LigandGatedChannels, NeurotransmitterConcentrations, NeurotransmitterKinetics, Neurotransmitters, 
    ReceptorKinetics, Timestep
};
use super::ion_channels::{
    TimestepIndependentIonChannel, IonChannel, ReducedCalciumChannel, 
    KSteadyStateChannel, LeakChannel,
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
    /// Calcium channel
    pub ca_channel: ReducedCalciumChannel,
    /// Potassium channel
    pub k_channel: KSteadyStateChannel,
    /// Leak channel
    pub leak_channel: LeakChannel,
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
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicReceptorNeurotransmitterType, T>,
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
            ca_channel: ReducedCalciumChannel::default(),
            k_channel: KSteadyStateChannel::default(),
            leak_channel: LeakChannel::default(),
            c_m: 6.6,
            dt: 0.01,
            is_spiking: false,
            was_increasing: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicReceptorNeurotransmitterType, T>::default(),
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
    /// Updates channel states based on current voltage
    pub fn update_channels(&mut self) {
        self.ca_channel.update_current(self.current_voltage);
        self.k_channel.update_current(self.current_voltage, self.dt);
        self.leak_channel.update_current(self.current_voltage);
    }

    /// Calculates change in voltage given an input current
    pub fn get_dv_change(&self, i: f32) -> f32 {
        (i - self.leak_channel.current - self.ca_channel.current - self.k_channel.current)
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
    type N = IonotropicReceptorNeurotransmitterType;

    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        self.update_channels();

        let last_voltage = self.current_voltage;
        self.current_voltage += self.get_dv_change(input_current);

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        self.handle_spiking(last_voltage)
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);
        
        self.update_channels();

        let last_voltage = self.current_voltage;
        let receptor_current = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
        self.current_voltage += self.get_dv_change(input_current) + receptor_current;

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        self.handle_spiking(last_voltage)
    }
}
