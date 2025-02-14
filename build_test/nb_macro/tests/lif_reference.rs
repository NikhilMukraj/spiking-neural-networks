use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;
use spiking_neural_networks::neuron::iterate_and_spike::DefaultReceptorsNeurotransmitterType;
use spiking_neural_networks::neuron::iterate_and_spike::{
    CurrentVoltage, GapConductance, IsSpiking, IterateAndSpike, LastFiringTime,
    DefaultReceptors, NeurotransmitterConcentrations, NeurotransmitterKinetics,
    Neurotransmitters, ReceptorKinetics, Timestep,
};
use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;


#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct ReferenceIntegrateAndFire<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f32,
    pub e: f32,
    pub v_reset: f32,
    pub v_th: f32,
    pub gap_conductance: f32,
    pub dt: f32,
    pub c_m: f32,
    pub is_spiking: bool,
    pub last_firing_time: Option<usize>,
    pub synaptic_neurotransmitters: Neurotransmitters<DefaultReceptorsNeurotransmitterType, T>,
    pub receptors: DefaultReceptors<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> ReferenceIntegrateAndFire<T, R> {
    fn handle_spiking(&mut self) -> bool {
        self.is_spiking = self.current_voltage >= self.v_th;
        if self.is_spiking {
            self.current_voltage = self.v_reset;
        }
        self.is_spiking
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike
    for ReferenceIntegrateAndFire<T, R>
{
    type N = DefaultReceptorsNeurotransmitterType;
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
        self.synaptic_neurotransmitters.get_concentrations()
    }
    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        let dv = ((self.current_voltage - self.e) + input_current) * self.dt;
        self.current_voltage += dv;
        self.synaptic_neurotransmitters
            .apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));
        self.handle_spiking()
    }
    fn iterate_with_neurotransmitter_and_spike(
        &mut self,
        input_current: f32,
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.receptors.update_receptor_kinetics(t_total, self.dt);
        self.receptors
            .set_receptor_currents(self.current_voltage, self.dt);
        let dv = ((self.current_voltage - self.e) + input_current) * self.dt;
        self.current_voltage += dv;
        self.current_voltage += self.receptors.get_receptor_currents(self.dt, self.c_m);
        self.synaptic_neurotransmitters
            .apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));
        self.handle_spiking()
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for ReferenceIntegrateAndFire<T, R> {
    fn default() -> Self {
        ReferenceIntegrateAndFire {
            e: 0.,
            v_reset: -75.,
            v_th: -55.,
            current_voltage: 0.,
            dt: 0.1,
            c_m: 1.,
            gap_conductance: 10.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters:
                Neurotransmitters::<DefaultReceptorsNeurotransmitterType, T>::default(),
            receptors: DefaultReceptors::<R>::default(),
        }
    }
}
