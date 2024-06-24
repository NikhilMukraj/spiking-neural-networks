//! An implementation of the FitzHugh-Nagumo neuron model.

use iterate_and_spike_traits::IterateAndSpikeBase;
use super::iterate_and_spike::{
    GaussianFactor, GaussianParameters, Potentiation, PotentiationType, IsSpiking,
    STDPParameters, STDP, CurrentVoltage, GapConductance, IterateAndSpike, 
    LastFiringTime, NeurotransmitterConcentrations, LigandGatedChannels, 
    ReceptorKinetics, NeurotransmitterKinetics, Neurotransmitters,
    ApproximateNeurotransmitter, ApproximateReceptor,
};


// A FitzHugh-Nagumo neuron 
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct FitzHughNagumoNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential
    pub current_voltage: f64,
    /// Initial voltage
    pub v_init: f64,
    /// Voltage threshold for spike calculation (mV)
    pub v_th: f64,
    /// Adaptive value
    pub w: f64,
    // Initial adaptive value
    pub w_init: f64,
    /// Resistance value
    pub resistance: f64,
    /// Adaptive value modifier
    pub a: f64,
    /// Adaptive value integration constant
    pub b: f64,
    /// Controls conductance of input gap junctions
    pub gap_conductance: f64, 
    /// Membrane time constant (ms)
    pub tau_m: f64,
    /// Membrane capacitance (nF)
    pub c_m: f64, 
    /// Timestep (ms)
    pub dt: f64, 
    /// Last timestep the neuron has spiked 
    pub last_firing_time: Option<usize>,
    /// Whether the voltage was increasing in the last step
    pub was_increasing: bool,
    /// Whether the neuron is currently spiking
    pub is_spiking: bool,
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

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for FitzHughNagumoNeuron<T, R> {
    fn default() -> Self {
        FitzHughNagumoNeuron {
            current_voltage: 0.,
            v_init: 0.,
            v_th: 1.,
            w_init: 0.,
            w: 0.,
            resistance: 0.1,
            a: 0.7,
            b: 0.8,
            gap_conductance: 1.0,
            tau_m: 12.5,
            c_m: 1.,
            dt: 0.1,
            last_firing_time: None,
            was_increasing: false,
            is_spiking: false,
            potentiation_type: PotentiationType::Excitatory,
            stdp_params: STDPParameters::default(),
            gaussian_params: GaussianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<T>::default(),
            ligand_gates: LigandGatedChannels::<R>::default()
        }
    }
}

impl FitzHughNagumoNeuron<ApproximateNeurotransmitter, ApproximateReceptor> {
    /// Returns the default implementation of the neuron
    pub fn default_impl() -> Self {
        FitzHughNagumoNeuron::default()
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> FitzHughNagumoNeuron<T, R> {
    fn get_dv_change(&self, i: f64) -> f64 {
        (  
            self.current_voltage - (self.current_voltage.powf(3.) / 3.) 
            - self.w + self.resistance * i
        ) * self.dt
    }

    fn get_dw_change(&self) -> f64 {
        (self.current_voltage + self.a + self.b * self.w) * (self.dt / self.tau_m)
    }

    fn handle_spiking(&mut self, last_voltage: f64) -> bool {
        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for FitzHughNagumoNeuron<T, R> {
    type T = T;
    type R = R;

    fn get_ligand_gates(&self) -> &LigandGatedChannels<R> {
        &self.ligand_gates
    }

    fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
        &self.synaptic_neurotransmitters
    }

    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let dv = self.get_dv_change(input_current);
        let dw = self.get_dw_change();
        let last_voltage = self.current_voltage;

        self.current_voltage += dv;
        self.w += dw;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.handle_spiking(last_voltage)
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&NeurotransmitterConcentrations>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);

        let dv = self.get_dv_change(input_current);
        let dw = self.get_dw_change();
        let neurotransmitter_dv = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
        let last_voltage = self.current_voltage;

        self.current_voltage += dv + neurotransmitter_dv;
        self.w += dw;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);

        self.handle_spiking(last_voltage)
    }
}
