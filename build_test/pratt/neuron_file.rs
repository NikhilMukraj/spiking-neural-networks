use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;
use spiking_neural_networks::neuron::iterate_and_spike::{CurrentVoltage, GapConductance, GaussianFactor, LastFiringTime, IsSpiking, IterateAndSpike, GaussianParameters, LigandGatedChannels, Neurotransmitters, NeurotransmitterKinetics, ReceptorKinetics, NeurotransmitterConcentrations};
use spiking_neural_networks::neuron::iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor};
use spiking_neural_networks::neuron::ion_channels::BasicGatingVariable;
use spiking_neural_networks::neuron::ion_channels::TimestepIndependentIonChannel;


#[derive(Debug, Clone, Copy)]
pub struct TestChannel {
	e: f32,
	g: f32,
	n: BasicGatingVariable,
	current: f32,
}

impl TimestepIndependentIonChannel for TestChannel {
	fn update_current(&mut self, voltage: f32) {
		self.current = ((self.g * self.n.state) * (self.current_voltage - self.e));
	}

	fn get_current(&self) -> f32 { self.current }
}

#[derive(Debug, Clone, Copy)]
pub struct TestLeak {
	e: f32,
	g: f32,
	current: f32,
}

impl TimestepIndependentIonChannel for TestLeak {
	fn update_current(&mut self, voltage: f32) {
		self.current = (self.g * (self.current_voltage - self.e));
	}

	fn get_current(&self) -> f32 { self.current }
}

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct BasicIntegrateAndFire<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
	current_voltage: f32,
	e: f32,
	v_reset: f32,
	v_th: f32,
	gap_conductance: f32,
	dt: f32,
	c_m: f32,
	is_spiking: bool,
	last_firing_time: Option<usize>,
	gaussian_params: GaussianParameters,
	synaptic_neurotransmitters: Neurotransmitters<T>,
	ligand_gates: LigandGatedChannels<R>,
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
		self.ligand_gates.update_receptor_kinetics(t_total);
		self.ligand_gates.set_receptor_currents(self.current_voltage);
		let dv = (((self.current_voltage - self.e) + input_current)) * self.dt;
		self.current_voltage += dv;
		self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
}

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct IonChannelNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
	current_voltage: f32,
	r: f32,
	gap_conductance: f32,
	dt: f32,
	c_m: f32,
	l: TestLeak,
	is_spiking: bool,
	last_firing_time: Option<usize>,
	gaussian_params: GaussianParameters,
	synaptic_neurotransmitters: Neurotransmitters<T>,
	ligand_gates: LigandGatedChannels<R>,
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
		self.ligand_gates.update_receptor_kinetics(t_total);
		self.ligand_gates.set_receptor_currents(self.current_voltage);
		let dv = ((self.l.current + (self.r * input_current))) * self.dt;
		self.current_voltage += dv;
		self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
		self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
		self.handle_spiking()
	}
}
