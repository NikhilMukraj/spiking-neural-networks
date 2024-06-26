//! # Spiking Neural Networks
//! 
//! `spiking_neural_networks` is a package focused on designing neuron models
//! with neurotransmission and calculating dynamics between neurons over time.
//! Neuronal dynamics are made using traits so they can be expanded via the 
//! type system to add new dynamics for different neurotransmitters, receptors
//! or neuron models. Currently implements system for spike trains, spike time depedent
//! plasticity, basic attractors, and dynamics for neurons connected in a lattice. 
//! See below for examples and how to add custom models.
//! 
//! ### FitzHugh-Nagumo Model with Static Input
//! 
//! ![FitzHugh-Nagumo with static current input](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/fhn_static.png?raw=true)
//! 
//! ### Coupled Izhikevich Neurons
//! 
//! ![Coupled Izhikevich models](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/izhikevich.png?raw=true)
//! 
//! ### Hodgkin Huxley Model with Neurotransmission
//! 
//! ![Hodgkin Huxley model voltage and neurotransmitter over time](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/hodgkin_huxley_neurotransmission.png?raw=true)
//! 
//! ### Spike Time Dependent Plasticity Weights over Time
//! 
//! ![STDP weights over time](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/stdp.png?raw=true)
//! 
//! ### Hopfield Network Pattern Reconstruction
//! 
//! ![Discrete Hopfield network pattern reconstruction](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/hopfield.png?raw=true)
//! 
//! ### Lattice
//! 
//! ![Voltage over time](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/lattice.gif?raw=true)
//!
//! ## Example Code
//! 
//! See [`examples folder`](https://docs.rs/crate/spiking_neural_networks/latest/source/examples/) for more examples.
//! 
//! ### Coupling neurons with current input
//! 
//! ```rust
//! use spiking_neural_networks::{
//!     neuron::{
//!         iterate_and_spike::{
//!             IterateAndSpike, weight_neurotransmitter_concentration,
//!         },
//!         signed_gap_junction,
//!     }
//! };
//! 
//! /// Calculates one iteration of two coupled neurons where the presynaptic neuron
//! /// has a static input current while the postsynaptic neuron takes
//! /// the current input and neurotransmitter input from the presynaptic neuron,
//! /// returns whether each neuron is spiking
//! /// 
//! /// - `presynaptic_neuron` : a neuron that implements `IterateAndSpike`
//! /// 
//! /// - `postsynaptic_neuron` : a neuron that implements `IterateAndSpike`
//! /// 
//! /// - `do_receptor_kinetics` : use `true` to update receptor gating values of 
//! /// the neurons based on neurotransmitter input during the simulation
//! /// 
//! /// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
//! pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
//!    presynaptic_neuron: &mut T, 
//!    postsynaptic_neuron: &mut T,
//!    input_current: f64,
//!    do_receptor_kinetics: bool,
//!    gaussian: bool,
//!) -> (bool, bool) {
//!    let (t_total, post_current, input_current) = if gaussian {
//!        // gets normally distributed factor to add noise with by scaling
//!        let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
//!        let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();
//!
//!        // scaling to add noise
//!        let input_current = input_current * pre_gaussian_factor;
//!
//!        // calculates input current to postsynaptic neuron
//!        let post_current = signed_gap_junction(
//!            &*presynaptic_neuron,
//!            &*postsynaptic_neuron,
//!        );
//!
//!        // calculates postsynaptic neurotransmitter input
//!        let t_total = if do_receptor_kinetics {
//!            // weights neurotransmitter with random noise
//!            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!            weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);
//!
//!            Some(t)
//!        } else {
//!            // returns None to indicate no update to receptor gating variables
//!            None
//!        };
//!
//!        (t_total, post_current, input_current)
//!    } else {
//!        // calculates input current to postsynaptic neuron
//!        let post_current = signed_gap_junction(
//!            &*presynaptic_neuron,
//!            &*postsynaptic_neuron,
//!        );
//!
//!       // calculates postsynaptic neurotransmitter input
//!       let t_total = if do_receptor_kinetics {
//!            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!            Some(t)
//!        } else {
//!            // returns None to indicate no update to receptor gating variables
//!            None
//!        };
//!
//!        (t_total, post_current, input_current)
//!    };
//!
//!    // updates presynaptic neuron by one step
//!    let pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);
//!
//!    // updates postsynaptic neuron by one step
//!    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!        post_current,
//!        t_total.as_ref(),
//!    );
//!
//!    (pre_spiking, post_spiking)
//!}
//! ```
//! 
//! ### Coupling neurons with spike train input
//! 
//! ```rust
//! use spiking_neural_networks::{
//!     neuron::{
//!         iterate_and_spike::{
//!             IterateAndSpike, weight_neurotransmitter_concentration,
//!         },
//!         spike_train::SpikeTrain,
//!         spike_train_gap_juncton,
//!     }
//! };
//! 
//! /// Calculates one iteration of two coupled neurons where the presynaptic neuron
//! /// has a spike train input while the postsynaptic neuron takes
//! /// the current input and neurotransmitter input from the presynaptic neuron,
//! /// also updates the last firing times of each neuron and spike train given the
//! /// current timestep of the simulation, returns whether each neuron is spiking
//! /// 
//! /// - `spike_train` : a spike train that implements `Spiketrain`
//! /// 
//! /// - `presynaptic_neuron` : a neuron that implements `IterateAndSpike`
//! /// 
//! /// - `postsynaptic_neuron` : a neuron that implements `IterateAndSpike`
//! /// 
//! /// - `timestep` : the current timestep of the simulation
//! /// 
//! /// - `do_receptor_kinetics` : use `true` to update receptor gating values of 
//! /// the neurons based on neurotransmitter input during the simulation
//! /// 
//! /// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
//! pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
//!     spike_train: &mut T,
//!     presynaptic_neuron: &mut U, 
//!     postsynaptic_neuron: &mut U,
//!     timestep: usize,
//!     do_receptor_kinetics: bool,
//!     gaussian: bool,
//! ) -> (bool, bool, bool) {
//!     // calculates input from spike train to presynaptic neuron given the current
//!     // timestep of the simulation
//!     let input_current = spike_train_gap_juncton(spike_train, presynaptic_neuron, timestep);
//!     
//!     let (pre_t_total, post_t_total, current) = if gaussian {
//!         // gets normally distributed factor to add noise with by scaling
//!         let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
//!         let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();
//!     
//! 
//!         // calculates presynaptic neurotransmitter input
//!         let pre_t_total = if do_receptor_kinetics {
//!             // weights neurotransmitter with random noise
//!             let mut t = spike_train.get_neurotransmitter_concentrations();
//!             weight_neurotransmitter_concentration(&mut t, pre_gaussian_factor);
//!     
//!             Some(t)
//!         } else {
//!             // returns None to indicate no update to receptor gating variables
//!             None
//!         };
//!     
//!         // calculates input current to postsynaptic neuron
//!         let current = signed_gap_junction(
//!             &*presynaptic_neuron,
//!             &*postsynaptic_neuron,
//!         );
//!     
//! 
//!         // calculates postsynaptic neurotransmitter input
//!         let post_t_total = if do_receptor_kinetics {
//!             // weights neurotransmitter with random noise
//!             let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!             weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);
//!     
//!             Some(t)
//!         } else {
//!             // returns None to indicate no update to receptor gating variables
//!             None
//!         };
//!     
//!         (pre_t_total, post_t_total, current)
//!     } else {
//!         // calculates presynaptic neurotransmitter input
//!         let pre_t_total = if do_receptor_kinetics {
//!             let t = spike_train.get_neurotransmitter_concentrations();
//!             Some(t)
//!         } else {
//!             // returns None to indicate no update to receptor gating variables
//!             None
//!         };
//!     
//!         // calculates current to postsynaptic neuron
//!         let current = signed_gap_junction(
//!             &*presynaptic_neuron,
//!             &*postsynaptic_neuron,
//!         );
//!     
//!         let post_t_total = if do_receptor_kinetics {
//!             let t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!             Some(t)
//!         } else {
//!             // returns None to indicate no update to receptor gating variables
//!             None
//!         };
//!     
//!         (pre_t_total, post_t_total, current)
//!     };
//!     
//!     // iterates neuron and if firing sets the last firing time to keep
//!     // track of activity in order to calculate input on the next iteration
//!     let spike_train_spiking = spike_train.iterate();   
//!     if spike_train_spiking {
//!         spike_train.set_last_firing_time(Some(timestep));
//!     }
//!     
//!     // iterates presynaptic neuron based on current and neurotransmitter input
//!     let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!         input_current,
//!         pre_t_total.as_ref(),
//!     );
//!     if pre_spiking {
//!         presynaptic_neuron.set_last_firing_time(Some(timestep));
//!     }
//!     
//!     // iterates presynaptic neuron based on current and neurotransmitter input
//!     let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!         current,
//!         post_t_total.as_ref(),
//!     ); 
//!     if post_spiking {
//!         postsynaptic_neuron.set_last_firing_time(Some(timestep));
//!     }
//!     
//!     (spike_train_spiking, pre_spiking, post_spiking)
//! }
//! ```
//! 
//! ### Coupling neurons with spike time dependent plasticity
//! 
//! ```rust
//! use std::collections::HashMap;
//! use crate::spiking_neural_networks::{
//!     neuron::{
//!         integrate_and_fire::IzhikevichNeuron,
//!         iterate_and_spike::{
//!             IterateAndSpike, GaussianParameters, NeurotransmitterConcentrations,
//!             ApproximateNeurotransmitter, ApproximateReceptor,
//!             weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
//!         },
//!         update_weight_stdp, signed_gap_junction,
//!     },
//!     distribution::limited_distr,
//! };
//! 
//! 
//! /// Generates keys in an ordered manner to ensure columns in file are ordered
//! fn generate_keys(n: usize) -> Vec<String> {
//!     let mut keys_vector: Vec<String> = vec![];
//! 
//!     for i in 0..n {
//!         keys_vector.push(format!("presynaptic_voltage_{}", i))
//!     }
//!     keys_vector.push(String::from("postsynaptic_voltage"));
//!     for i in 0..n {
//!         keys_vector.push(format!("weight_{}", i));
//!     }
//! 
//!     keys_vector
//! }
//! 
//! /// Updates each presynaptic neuron's weights given the timestep
//! /// and whether the neuron is spiking along with the state of the
//! /// postsynaptic neuron
//! fn update_isolated_presynaptic_neuron_weights<T: IterateAndSpike>(
//!     neurons: &mut Vec<T>,
//!     neuron: &T,
//!     weights: &mut Vec<f64>,
//!     delta_ws: &mut Vec<f64>,
//!     timestep: usize,
//!     is_spikings: Vec<bool>,
//! ) {
//!     for (n, i) in is_spikings.iter().enumerate() {
//!         if *i {
//!             // update firing times if spiking
//!             neurons[n].set_last_firing_time(Some(timestep));
//!             delta_ws[n] = update_weight_stdp(&neurons[n], &*neuron);
//!             weights[n] += delta_ws[n];
//!         }
//!     }
//! }
//! 
//! /// Tests spike time dependent plasticity on a set of given neurons
//! /// 
//! /// `presynaptic_neurons` : a set of input neurons
//! ///
//! /// `postsynaptic_neuron` : a single output neuron
//! ///
//! /// `iterations` : number of timesteps to simulate neurons for
//! ///
//! /// `input_current` : an input current for the presynaptic neurons to take input from
//! ///
//! /// `input_current_deviation` : degree of noise to add to input currents to introduce changes
//! /// in postsynaptic input
//! ///
//! /// `weight_params` : parameters to use to randomly initialize the weights on the 
//! /// input presynaptic neurons
//! ///
//! /// `do_receptor_kinetics` : whether to update receptor gating values based on neurotransmitter input
//! fn test_isolated_stdp<T: IterateAndSpike>(
//!     presynaptic_neurons: &mut Vec<T>,
//!     postsynaptic_neuron: &mut T,
//!     iterations: usize,
//!     input_current: f64,
//!     input_current_deviation: f64,
//!     weight_params: &GaussianParameters,
//!     do_receptor_kinetics: bool,
//! ) -> HashMap<String, Vec<f64>> {
//!     let n = presynaptic_neurons.len();
//! 
//!     // generate different currents
//!     let input_currents: Vec<f64> = (0..n).map(|_| 
//!             input_current * limited_distr(1.0, input_current_deviation, 0., 2.)
//!         )
//!         .collect();
//! 
//!     // generate random weights
//!     let mut weights: Vec<f64> = (0..n).map(|_| weight_params.get_random_number())
//!         .collect();
//! 
//!     let mut delta_ws: Vec<f64> = (0..n)
//!         .map(|_| 0.0)
//!         .collect();
//! 
//!     // generate hashmap to save history of simulation
//!     let mut output_hashmap: HashMap<String, Vec<f64>> = HashMap::new();
//!     let keys_vector = generate_keys(n);
//!     for i in keys_vector.iter() {
//!         output_hashmap.insert(String::from(i), vec![]);
//!     }
//! 
//!     for timestep in 0..iterations {
//!         // calculates weighted current inputs and averages them to ensure input does not get too high,
//!         // otherwise neuronal dynamics becomes unstable
//!         let calculated_current: f64 = (0..n)
//!             .map(
//!                 |i| {
//!                     let output = weights[i] * signed_gap_junction(
//!                         &presynaptic_neurons[i], 
//!                         &*postsynaptic_neuron
//!                     );
//! 
//!                     output / (n as f64)
//!                 }
//!             ) 
//!             .collect::<Vec<f64>>()
//!             .iter()
//!             .sum();
//!         // calculates weighted neurotransmitter inputs
//!         let presynaptic_neurotransmitters: Option<NeurotransmitterConcentrations> = match do_receptor_kinetics {
//!             true => Some({
//!                 let neurotransmitters_vec = (0..n) 
//!                     .map(|i| {
//!                         let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
//!                         weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);
//! 
//!                         presynaptic_neurotransmitter
//!                     }
//!                 ).collect::<Vec<NeurotransmitterConcentrations>>();
//! 
//!                 let mut neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);
//! 
//!                 weight_neurotransmitter_concentration(&mut neurotransmitters, (1 / n) as f64); 
//! 
//!                 neurotransmitters
//!             }),
//!             false => None
//!         };
//!         
//!         // adds noise to current inputs with normally distributed random noise
//!         let noise_factor = postsynaptic_neuron.get_gaussian_factor();
//!         let presynaptic_inputs: Vec<f64> = (0..n)
//!             .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
//!             .collect();
//!         let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
//!             .map(|(presynaptic_neuron, input_value)| {
//!                 presynaptic_neuron.iterate_and_spike(*input_value)
//!             })
//!             .collect();
//!         // iterates postsynaptic neuron based on calculated inputs
//!         let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!             noise_factor * calculated_current,
//!             presynaptic_neurotransmitters.as_ref(),
//!         );
//! 
//!         update_isolated_presynaptic_neuron_weights(
//!             presynaptic_neurons, 
//!             &postsynaptic_neuron,
//!             &mut weights, 
//!             &mut delta_ws, 
//!             timestep, 
//!             is_spikings,
//!         );
//! 
//!         // if postsynaptic neuron fires then update the firing time
//!         // and update the weight accordingly
//!         if is_spiking {
//!             postsynaptic_neuron.set_last_firing_time(Some(timestep));
//!             for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
//!                 delta_ws[n_neuron] = update_weight_stdp(i, postsynaptic_neuron);
//!                 weights[n_neuron] += delta_ws[n_neuron];
//!             }
//!         }
//! 
//!         for (index, i) in presynaptic_neurons.iter().enumerate() {
//!             output_hashmap.get_mut(&format!("presynaptic_voltage_{}", index))
//!                 .expect("Could not find hashmap value")
//!                 .push(i.get_current_voltage());
//!         }
//!         output_hashmap.get_mut("postsynaptic_voltage").expect("Could not find hashmap value")
//!             .push(postsynaptic_neuron.get_current_voltage());
//!         for (index, i) in weights.iter().enumerate() {
//!             output_hashmap.get_mut(&format!("weight_{}", index))
//!                 .expect("Could not find hashmap value")
//!                 .push(*i);
//!         }
//!     }
//! 
//!     output_hashmap
//! }
//! ```
//! 
//! ### Custom `IterateAndSpike` implementation
//! 
//! ```rust
//! use spiking_neural_networks::iterate_and_spike_traits::IterateAndSpikeBase;
//! use spiking_neural_networks::neuron::iterate_and_spike::{
//!     GaussianFactor, GaussianParameters, Potentiation, PotentiationType, STDPParameters,
//!     IsSpiking, STDP, CurrentVoltage, GapConductance, IterateAndSpike, 
//!     LastFiringTime, NeurotransmitterConcentrations, LigandGatedChannels, 
//!     ReceptorKinetics, NeurotransmitterKinetics, Neurotransmitters,
//!     ApproximateNeurotransmitter, ApproximateReceptor,
//! };
//! 
//! 
//! /// A FitzHugh-Nagumo neuron 
//! #[derive(Debug, Clone, IterateAndSpikeBase)]
//! pub struct FitzHughNagumoNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
//!     /// Membrane potential
//!     pub current_voltage: f64,
//!     /// Initial voltage
//!     pub v_init: f64,
//!     /// Voltage threshold for spike calculation (mV)
//!     pub v_th: f64,
//!     /// Adaptive value
//!     pub w: f64,
//!     // Initial adaptive value
//!     pub w_init: f64,
//!     /// Resistance value
//!     pub resistance: f64,
//!     /// Adaptive value modifier
//!     pub a: f64,
//!     /// Adaptive value integration constant
//!     pub b: f64,
//!     /// Controls conductance of input gap junctions
//!     pub gap_conductance: f64, 
//!     /// Membrane time constant (ms)
//!     pub tau_m: f64,
//!     /// Membrane capacitance (nF)
//!     pub c_m: f64, 
//!     /// Timestep (ms)
//!     pub dt: f64, 
//!     /// Last timestep the neuron has spiked 
//!     pub last_firing_time: Option<usize>,
//!     /// Whether the voltage was increasing in the last step
//!     pub was_increasing: bool,
//!     /// Whether the neuron is currently spiking
//!     pub is_spiking: bool,
//!     /// Potentiation type of neuron
//!     pub potentiation_type: PotentiationType,
//!     /// STDP parameters
//!     pub stdp_params: STDPParameters,
//!     /// Parameters used in generating noise
//!     pub gaussian_params: GaussianParameters,
//!     /// Postsynaptic neurotransmitters in cleft
//!     pub synaptic_neurotransmitters: Neurotransmitters<T>,
//!     /// Ionotropic receptor ligand gated channels
//!     pub ligand_gates: LigandGatedChannels<R>,
//! }
//! 
//! impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> FitzHughNagumoNeuron<T, R> {
//!     // calculates change in voltage
//!     fn get_dv_change(&self, i: f64) -> f64 {
//!         (  
//!             self.current_voltage - (self.current_voltage.powf(3.) / 3.) 
//!             - self.w + self.resistance * i
//!         ) * self.dt
//!     }
//! 
//!     // calculates change in adaptive value
//!     fn get_dw_change(&self) -> f64 {
//!         (self.current_voltage + self.a + self.b * self.w) * (self.dt / self.tau_m)
//!     }
//! 
//!     // checks if neuron is currently spiking but seeing if the neuron is increasing in
//!     // reference to the last inputted voltage and if it is above a certain
//!     // voltage threshold, if it is then the neuron is considered spiking
//!     // and `true` is returned, otherwise `false` is returned
//!     fn handle_spiking(&mut self, last_voltage: f64) -> bool {
//!         let increasing_right_now = last_voltage < self.current_voltage;
//!         let threshold_crossed = self.current_voltage > self.v_th;
//!         let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;
//! 
//!         self.is_spiking = is_spiking;
//!         self.was_increasing = increasing_right_now;
//! 
//!         is_spiking
//!     }
//! }
//! 
//! impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for FitzHughNagumoNeuron<T, R> {
//!     type T = T;
//!     type R = R;
//! 
//!     fn get_ligand_gates(&self) -> &LigandGatedChannels<R> {
//!         &self.ligand_gates
//!     }
//! 
//!     fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
//!         &self.synaptic_neurotransmitters
//!     }
//! 
//!     fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
//!         self.synaptic_neurotransmitters.get_concentrations()
//!     }
//!     
//!     // updates voltage and adaptive values as well as the 
//!     // neurotransmitters, receptor current is not factored in,
//!     // and spiking is handled and returns whether it is currently spiking
//!     fn iterate_and_spike(&mut self, input_current: f64) -> bool {
//!         let dv = self.get_dv_change(input_current);
//!         let dw = self.get_dw_change();
//!         let last_voltage = self.current_voltage;
//! 
//!         self.current_voltage += dv;
//!         self.w += dw;
//! 
//!         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
//! 
//!         self.handle_spiking(last_voltage)
//!     }
//! 
//!     // updates voltage and adaptive values as well as the 
//!     // neurotransmitters, receptor current is factored in and receptor gating
//!     // is updated if `t_total` is not `None`, spiking is handled at the end
//!     // of the method and returns whether it is currently spiking
//!     fn iterate_with_neurotransmitter_and_spike(
//!         &mut self, 
//!         input_current: f64, 
//!         t_total: Option<&NeurotransmitterConcentrations>,
//!     ) -> bool {
//!         self.ligand_gates.update_receptor_kinetics(t_total);
//!         self.ligand_gates.set_receptor_currents(self.current_voltage);
//! 
//!         let dv = self.get_dv_change(input_current);
//!         let dw = self.get_dw_change();
//!         let neurotransmitter_dv = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
//!         let last_voltage = self.current_voltage;
//! 
//!         self.current_voltage += dv + neurotransmitter_dv;
//!         self.w += dw;
//! 
//!         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
//! 
//!         self.handle_spiking(last_voltage)
//!     }
//! }
//! ```
//! 
//! ### Custom `NeurotransmitterKinetics` implementation
//! 
//! ```rust
//! use spiking_neural_networks::iterate_and_spike::NeurotransmitterKinetics;
//! 
//! /// An approximation of neurotransmitter kinetics that sets the concentration to the 
//! /// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
//! /// slowly through exponential decay that scales based on the `decay_constant` and `dt`
//! #[derive(Debug, Clone, Copy)]
//! pub struct ExponentialDecayNeurotransmitter {
//!     /// Maximal neurotransmitter concentration (mM)
//!     pub t_max: f64,
//!     /// Current neurotransmitter concentration (mM)
//!     pub t: f64,
//!     /// Voltage threshold for detecting spikes (mV)
//!     pub v_th: f64,
//!     /// Amount to decay neurotransmitter concentration by
//!     pub decay_constant: f64,
//!     /// Timestep factor in decreasing neurotransmitter concentration (ms)
//!     pub dt: f64,
//! }
//! 
//! // used to determine when voltage spike occurs
//! fn heaviside(x: f64) -> f64 {
//!     if x > 0. {
//!         1.
//!     } else {
//!         0.
//!     }
//! }
//! 
//! // calculate change in concentration
//! fn exp_decay(x: f64, l: f64, dt: f64) -> f64 {
//!     -x * (dt / -l).exp()
//! }
//! 
//! impl NeurotransmitterKinetics for ExponentialDecayNeurotransmitter {
//!     fn apply_t_change(&mut self, voltage: f64) {
//!         let t_change = exp_decay(self.t, self.decay_constant, self.dt);
//!         // add change and account for spike
//!         self.t += t_change + (heaviside(voltage - self.v_th) * self.t_max);
//!         self.t = self.t_max.min(self.t.max(0.)); // clamp values
//!     }
//! 
//!     fn get_t(&self) -> f64 {
//!         self.t
//!     }
//! 
//!     fn set_t(&mut self, t: f64) {
//!         self.t = t;
//!     }
//! }
//! ```
//! 
//! ### Custom `ReceptorKinetics` implementation
//! 
//! ```rust
//! use spiking_neural_networks::iterate_and_spike::ReceptorKinetics;
//! 
//! /// Receptor dynamics approximation that sets the receptor
//! /// gating value to the inputted neurotransmitter concentration and
//! /// then exponentially decays the receptor over time
//! #[derive(Debug, Clone, Copy)]
//! pub struct ExponentialDecayReceptor {
//!     /// Maximal receptor gating value
//!     pub r_max: f64,
//!     /// Receptor gating value
//!     pub r: f64,
//!     /// Amount to decay neurotransmitter concentration by
//!     pub decay_constant: f64,
//!     /// Timestep factor in decreasing neurotransmitter concentration (ms)
//!     pub dt: f64,
//! }
//! 
//! // calculate change in receptor gating variable over time
//! fn exp_decay(x: f64, l: f64, dt: f64) -> f64 {
//!     -x * (dt / -l).exp()
//! }
//! 
//! impl ReceptorKinetics for ExponentialDecayReceptor {
//!     fn apply_r_change(&mut self, t: f64) {
//!         // calculate and apply change
//!         self.r += exp_decay(self.r, self.decay_constant, self.dt) + t;
//!         self.r = self.r_max.min(self.r.max(0.)); // clamp values
//!     }
//!
//!     fn get_r(&self) -> f64 {
//!         self.r
//!     }
//!
//!     fn set_r(&mut self, r: f64) {
//!         self.r = r;
//!     }
//! }
//!
//! // automatically generate defaults so `LigandGatedChannels`
//! // can use default receptor settings in construction
//! macro_rules! impl_exp_decay_receptor_default {
//!     ($trait:ident, $method:ident) => {
//!         impl $trait for ExponentialDecayReceptor {
//!             fn $method() -> Self {
//!                 ExponentialDecayReceptor { 
//!                     r_max: 1.0,
//!                     r: 0.,
//!                     decay_constant: 2.,
//!                     dt: 0.1,
//!                    }
//!             }
//!         }
//!     };
//! }
//!
//! impl_exp_decay_receptor_default!(Default, default);
//! impl_exp_decay_receptor_default!(AMPADefault, ampa_default);
//! impl_exp_decay_receptor_default!(GABAaDefault, gabaa_default);
//! impl_exp_decay_receptor_default!(GABAbDefault, gabab_default);
//! impl_exp_decay_receptor_default!(NMDADefault, nmda_default);
//! ```

pub mod correlation;
pub mod distribution;
pub mod eeg;
pub mod fitting;
pub mod ga;
pub mod graph;
pub mod neuron;
