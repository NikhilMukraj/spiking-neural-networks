use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;
use spiking_neural_networks::neuron::iterate_and_spike::{CurrentVoltage, GapConductance, GaussianFactor, LastFiringTime, IsSpiking, IterateAndSpike, GaussianParameters, LigandGatedChannels, Neurotransmitters, NeurotransmitterKinetics, ReceptorKinetics, NeurotransmitterConcentrations};
use spiking_neural_networks::neuron::iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor};



#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct BasicIntegrateAndFire<T: ApproximateNeurotransmitter, R: ApproximateReceptor> {
	current_voltage: f32,
	e: f32,
	v_reset: f32,
	v_th: f32,
	gap_conductance: f32,
	is_spiking: bool,
	gaussian_params: GaussianParameters,
	synaptic_neurotransmitters: Neurotransmitters<T>,
	ligand_gates: LigandGatedChannels<R>,
}

impl<T: ApproximateNeurotransmitter, R: ApproximateReceptor> BasicIntegrateAndFire<T, R> {
	fn handle_spiking(&mut self) -> bool {
		if self.current_voltage >= self.v_th {
			self.current_voltage = self.v_reset;
		}
	}
}

impl<T: ApproximateNeurotransmitter, R: ApproximateReceptor> IterateAndSpike for BasicIntegrateAndFire<T, R> {
	fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
		self.synaptic_neurotransmitters.get_concentrations()
	}
	
	fn iterate_and_spike(&mut self, input_current: f32) -> bool {
		let dv = (((self.current_voltage - self.e) + input_current)) * dt;
		self.current_voltage += dv;
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
	
	fn iterate_with_neurotransmitter_and_spike(
		&mut self,
		input_current: f32,
		t_total: &NeurotransmitterConcentrations,
	) -> bool {
		self.ligand_gates.update_receptor_kinetics(t_total);
		self.ligand_gates.set_receptor_currents(self.current_voltage);
		let dv = (((self.current_voltage - self.e) + input_current)) * dt;
		self.current_voltage += dv;
		self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
}