use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;
use spiking_neural_networks::neuron::iterate_and_spike::{CurrentVoltage, GapConductance, GaussianFactor, LastFiringTime, IsSpiking, IterateAndSpike, GaussianParameters, LigandGatedChannels, Neurotransmitters, NeurotransmitterKinetics, ReceptorKinetics, NeurotransmitterConcentrations};
use spiking_neural_networks::neuron::iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor};
use spiking_neural_networks::neuron::ion_channels::BasicGatingVariable;
use spiking_neural_networks::neuron::ion_channels::TimestepIndependentIonChannel;


#[derive(Debug, Clone, Copy)]
pub struct TestLeak {
	pub e: f32,
	pub g: f32,
	pub current: f32,
}

impl TimestepIndependentIonChannel for TestLeak {
	fn update_current(&mut self, voltage: f32) {
		self.current = (self.g * (self.current_voltage - self.e));
	}

	fn get_current(&self) -> f32 { self.current }
}

#[derive(Debug, Clone, Copy)]
pub struct TestChannel {
	pub e: f32,
	pub g: f32,
	pub n: BasicGatingVariable,
	pub current: f32,
}

impl TimestepIndependentIonChannel for TestChannel {
	fn update_current(&mut self, voltage: f32) {
		self.current = ((self.g * self.n.state) * (self.current_voltage - self.e));
	}

	fn get_current(&self) -> f32 { self.current }
}

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct BasicIntegrateAndFire<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
	pub current_voltage: f32,
	pub e: f32,
	pub v_reset: f32,
	pub v_th: f32,
	pub gap_conductance: f32,
	pub dt: f32,
	pub c_m: f32,
	pub is_spiking: bool,
	pub last_firing_time: Option<usize>,
	pub gaussian_params: GaussianParameters,
	pub synaptic_neurotransmitters: Neurotransmitters<T>,
	pub ligand_gates: LigandGatedChannels<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> BasicIntegrateAndFire<T, R> {
	fn handle_spiking(&mut self) -> bool {
		self.is_spiking = self.current_voltage >= self.v_th;
		if self.is_spiking {
			self.current_voltage = self.v_reset;
		}
	
		self.is_spiking
	}
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for BasicIntegrateAndFire<T, R> {
	fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
		self.synaptic_neurotransmitters.get_concentrations()
	}
	
	fn iterate_and_spike(&mut self, input_current: f32) -> bool {
		let dv = (((self.current_voltage - self.e) + input_current)) * self.dt;
		self.current_voltage += dv;
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
	
	fn iterate_with_neurotransmitter_and_spike(
		&mut self,
		input_current: f32,
		t_total: &NeurotransmitterConcentrations,
	) -> bool {
		self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
		self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);
		let dv = (((self.current_voltage - self.e) + input_current)) * self.dt;
		self.current_voltage += dv;
		self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
}

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct IonChannelNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
	pub current_voltage: f32,
	pub r: f32,
	pub gap_conductance: f32,
	pub dt: f32,
	pub c_m: f32,
	pub l: TestLeak,
	pub is_spiking: bool,
	pub last_firing_time: Option<usize>,
	pub gaussian_params: GaussianParameters,
	pub synaptic_neurotransmitters: Neurotransmitters<T>,
	pub ligand_gates: LigandGatedChannels<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IonChannelNeuron<T, R> {
	fn handle_spiking(&mut self) -> bool {
		self.is_spiking = self.current_voltage >= self.v_th;
		if self.is_spiking {
			self.current_voltage = self.v_reset;
		}
	
		self.is_spiking
	}
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for IonChannelNeuron<T, R> {
	fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
		self.synaptic_neurotransmitters.get_concentrations()
	}
	
	fn iterate_and_spike(&mut self, input_current: f32) -> bool {
		self.l.update_current(self.current_voltage, self.dt);
		let dv = ((self.l.current + (self.r * input_current))) * self.dt;
		self.current_voltage += dv;
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
	
	fn iterate_with_neurotransmitter_and_spike(
		&mut self,
		input_current: f32,
		t_total: &NeurotransmitterConcentrations,
	) -> bool {
		self.l.update_current(self.current_voltage, self.dt);
		self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
		self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);
		let dv = ((self.l.current + (self.r * input_current))) * self.dt;
		self.current_voltage += dv;
		self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
}
