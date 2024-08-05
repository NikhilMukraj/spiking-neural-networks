//! # Spiking Neural Networks
//! 
//! `spiking_neural_networks` is a package focused on designing neuron models
//! with neurotransmission and calculating dynamics between neurons over time.
//! Neuronal dynamics are made using traits so they can be expanded via the 
//! type system to add new dynamics for different neurotransmitters, receptors
//! or neuron models. Currently implements system for spike trains, spike time depedent
//! plasticity, basic attractors, reward modulated dynamics,
//! and dynamics for neurons connected in a lattice. 
//! See below for examples and how to add custom models.
//! 
//! ### Morris-Lecar Model with Static Input
//! 
//! ![Morris-Lecar with static current input](https://github.com/NikhilMukraj/spiking-neural-networks/blob/main/images/ml_static.png?raw=true)
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
//! use std::collections::HashMap;
//! use spiking_neural_networks::{
//!     neuron::{
//!         iterate_and_spike::{
//!             IterateAndSpike, weight_neurotransmitter_concentration,
//!         },
//!         gap_junction,
//!     }
//! };
//! 
//! /// Calculates one iteration of two coupled neurons where the presynaptic neuron
//! /// has a static input current while the postsynaptic neuron takes
//! /// the current input and neurotransmitter input from the presynaptic neuron,
//! /// returns whether each neuron is spiking
//! /// 
//! /// - `presynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
//! /// 
//! /// - `postsynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
//! /// 
//! /// - `electrical_synapse` : use `true` to update neurons based on electrical gap junctions
//! /// 
//! /// - `chemical_synapse` : use `true` to update receptor gating values of 
//! /// the neurons based on neurotransmitter input during the simulation
//! /// 
//! /// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
//! pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
//!     presynaptic_neuron: &mut T, 
//!     postsynaptic_neuron: &mut T,
//!     input_current: f32,
//!     electrical_synapse: bool,
//!     chemical_synapse: bool,
//!     gaussian: bool,
//! ) -> (bool, bool) {
//!     let (t_total, post_current, input_current) = if gaussian {
//!         // gets normally distributed factor to add noise with by scaling
//!         let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
//!         let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();
//!
//!         // scaling to add noise
//!         let input_current = input_current * pre_gaussian_factor;
//!
//!         // calculates electrical input current to postsynaptic neuron
//!         let post_current = if electrical_synapse {
//!               gap_junction(
//!                  &*presynaptic_neuron,
//!                  &*postsynaptic_neuron,
//!             ) * post_gaussian_factor
//!        } else {
//!             0.
//!        };
//!
//!        // calculates postsynaptic neurotransmitter input
//!        let t_total = if chemical_synapse {
//!            // weights neurotransmitter with random noise
//!            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!            weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);
//!
//!            t
//!        } else {
//!            // returns empty hashmap to indicate no chemical transmission
//!            HashMap::new()
//!        };
//!
//!        (t_total, post_current, input_current)
//!    } else {
//!         // calculates input current to postsynaptic neuron
//!        let post_current = if electrical_synapse {
//!             gap_junction(
//!                 &*presynaptic_neuron,
//!                 &*postsynaptic_neuron,
//!             )
//!        } else {
//!             0.
//!        };
//!
//!       // calculates postsynaptic neurotransmitter input
//!       let t_total = if chemical_synapse {
//!            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!            t
//!       } else {
//!            // returns empty hashmap to indicate no chemical transmission
//!            HashMap::new()
//!       };
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
//!        &t_total,
//!    );
//!
//!    (pre_spiking, post_spiking)
//! }
//! ```
//! 
//! ### Coupling neurons with spike train input
//! 
//! ```rust
//! use std::collections::HashMap;
//! use spiking_neural_networks::{
//!     neuron::{
//!         iterate_and_spike::{
//!             IterateAndSpike, weight_neurotransmitter_concentration,
//!         },
//!         spike_train::SpikeTrain,
//!         spike_train_gap_juncton, gap_junction,
//!     }
//! };
//! 
//! /// Calculates one iteration of two coupled neurons where the presynaptic neuron
//! /// has a spike train input while the postsynaptic neuron takes
//! /// the current input and neurotransmitter input from the presynaptic neuron,
//! /// also updates the last firing times of each neuron and spike train given the
//! /// current timestep of the simulation, returns whether each neuron is spiking
//! /// 
//! /// - `spike_train` : a spike train that implements [`SpikeTrain`]
//! /// 
//! /// - `presynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
//! /// 
//! /// - `postsynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
//! /// 
//! /// - `timestep` : the current timestep of the simulation
//! /// 
//! /// - `electrical_synapse` : use `true` to update neurons based on electrical gap junctions
//! /// 
//! /// - `chemical_synapse` : use `true` to update receptor gating values of 
//! /// the neurons based on neurotransmitter input during the simulation
//! /// 
//! /// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
//! pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
//!     spike_train: &mut T,
//!     presynaptic_neuron: &mut U, 
//!     postsynaptic_neuron: &mut U,
//!     timestep: usize,
//!     electrical_synapse: bool,
//!     chemical_synapse: bool,
//!     gaussian: bool,
//! ) -> (bool, bool, bool) {
//!     let (pre_t_total, post_t_total, pre_current, post_current) = if gaussian {
//!         // gets normally distributed factor to add noise with by scaling
//!         let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
//!         let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();
//! 
//!         // calculates presynaptic neurotransmitter input
//!         let pre_t_total = if chemical_synapse {
//!             // weights neurotransmitter with random noise
//!             let mut t = spike_train.get_neurotransmitter_concentrations();
//!             weight_neurotransmitter_concentration(&mut t, pre_gaussian_factor);
//! 
//!             t
//!         } else {
//!             // returns empty hashmap to indicate no chemical transmission
//!             HashMap::new()
//!         };
//! 
//!         let (pre_current, post_current) = if electrical_synapse {
//!             // calculates input from spike train to presynaptic neuron given the current
//!             // timestep of the simulation
//!             let pre_current = spike_train_gap_juncton(
//!                 spike_train, 
//!                 presynaptic_neuron, 
//!                 timestep
//!             ) * pre_gaussian_factor;
//! 
//!             // input from presynaptic neuron to postsynaptic
//!             let post_current = gap_junction(
//!                 &*presynaptic_neuron,
//!                 &*postsynaptic_neuron,
//!             ) * post_gaussian_factor;
//! 
//!             (pre_current, post_current)
//!         } else {
//!             // returns 0 if no electrical synapse
//!             (0., 0.)
//!         };
//! 
//!         let post_t_total = if chemical_synapse {
//!             let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
//!             weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);
//! 
//!             t
//!         } else {
//!             // returns empty hashmap to indicate no chemical transmission
//!             HashMap::new()
//!         };
//! 
//!         (pre_t_total, post_t_total, pre_current, post_current)
//!     } else {
//!         let pre_t_total = if chemical_synapse {
//!             spike_train.get_neurotransmitter_concentrations()
//!         } else {
//!             // returns empty hashmap to indicate no chemical transmission
//!             HashMap::new()
//!         };
//! 
//!         // calculates currents from electrical gap junctions
//!         let (pre_current, current) = if electrical_synapse {
//!             // calculates input from spike train to presynaptic neuron given the current
//!             // timestep of the simulation
//!             let pre_current = spike_train_gap_juncton(
//!                 spike_train, 
//!                 presynaptic_neuron, 
//!                 timestep
//!             );
//! 
//!             // input from presynaptic neuron to postsynaptic
//!             let current = gap_junction(
//!                 &*presynaptic_neuron,
//!                 &*postsynaptic_neuron,
//!             );
//! 
//!             (pre_current, current)
//!         } else {
//!             // returns 0 if no electrical synapse
//!             (0., 0.)
//!         };
//! 
//!         // calculates neurotransmitter input
//!         let post_t_total = if chemical_synapse {
//!             presynaptic_neuron.get_neurotransmitter_concentrations()
//!         } else {
//!             // returns empty hashmap to indicate no chemical transmission
//!             HashMap::new()
//!         };
//! 
//!         (pre_t_total, post_t_total, pre_current, current)
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
//!         pre_current,
//!         &pre_t_total,
//!     );
//!     if pre_spiking {
//!         presynaptic_neuron.set_last_firing_time(Some(timestep));
//!     }
//! 
//!     // iterates presynaptic neuron based on current and neurotransmitter input
//!     let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!         post_current,
//!         &post_t_total,
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
//!         plasticity::{Plasticity, STDP},
//!         gap_junction,
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
//! /// Tests spike time dependent plasticity on a set of given neurons
//! /// 
//! /// `presynaptic_neurons` : a set of input neurons
//! ///
//! /// `postsynaptic_neuron` : a single output neuron
//! ///
//! /// `stdp_params` : parameters for the plasticity rule
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
//! /// - `electrical_synapse` : use `true` to update neurons based on electrical gap junctions
//! /// 
//! /// - `chemical_synapse` : use `true` to update receptor gating values of 
//! /// the neurons based on neurotransmitter input during the simulation
//! fn test_isolated_stdp<T: IterateAndSpike>(
//!     presynaptic_neurons: &mut Vec<T>,
//!     postsynaptic_neuron: &mut T,
//!     stdp_params: &STDP,
//!     iterations: usize,
//!     input_current: f32,
//!     input_current_deviation: f32,
//!     weight_params: &GaussianParameters,
//!     electrical_synapse: bool,
//!     chemical_synapse: bool,
//! ) -> HashMap<String, Vec<f32>> {
//!     let n = presynaptic_neurons.len();
//! 
//!     // generate different currents
//!     let input_currents: Vec<f32> = (0..n).map(|_| 
//!             input_current * limited_distr(1.0, input_current_deviation, 0., 2.)
//!         )
//!         .collect();
//! 
//!     // generate random weights
//!     let mut weights: Vec<f32> = (0..n).map(|_| weight_params.get_random_number())
//!         .collect();
//! 
//!     // generate hashmap to save history of simulation
//!     let mut output_hashmap: HashMap<String, Vec<f32>> = HashMap::new();
//!     let keys_vector = generate_keys(n);
//!     for i in keys_vector.iter() {
//!         output_hashmap.insert(String::from(i), vec![]);
//!     }
//! 
//!     for timestep in 0..iterations {
//!         // calculates weighted current inputs and averages them to ensure input does not get too high,
//!         // otherwise neuronal dynamics becomes unstable
//!         let calculated_current: f32 = if electrical_synapse { 
//!             (0..n).map(
//!                     |i| {
//!                         let output = weights[i] * gap_junction(
//!                             &presynaptic_neurons[i], 
//!                             &*postsynaptic_neuron
//!                         );
//! 
//!                         output / (n as f32)
//!                     }
//!                 ) 
//!                 .collect::<Vec<f32>>()
//!                 .iter()
//!                 .sum()
//!             } else {
//!                 // returns 0 if no electrical synapses to represent to electrical transmission
//!                 0.
//!             };
//! 
//!         // calculates weighted neurotransmitter inputs
//!         let presynaptic_neurotransmitters: NeurotransmitterConcentrations = if chemical_synapse {
//!             let neurotransmitters_vec = (0..n) 
//!                 .map(|i| {
//!                     let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
//!                     weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);
//! 
//!                     presynaptic_neurotransmitter
//!                 }
//!             ).collect::<Vec<NeurotransmitterConcentrations>>();
//! 
//!             let mut neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);
//! 
//!             weight_neurotransmitter_concentration(&mut neurotransmitters, (1 / n) as f32); 
//! 
//!             neurotransmitters
//!         } else {
//!             // returns empty hashmap to indicate no chemical transmission
//!             HashMap::new()
//!         };
//!         
//!         // adds noise to current inputs with normally distributed random noise
//!         let presynaptic_inputs: Vec<f32> = (0..n)
//!             .map(|i| input_currents[i] * presynaptic_neurons[i].get_gaussian_factor())
//!             .collect();
//!         let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
//!             .map(|(presynaptic_neuron, input_value)| {
//!                 presynaptic_neuron.iterate_and_spike(*input_value)
//!             })
//!             .collect();
//!         // iterates postsynaptic neuron based on calculated inputs
//!         let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//!             calculated_current,
//!             &presynaptic_neurotransmitters,
//!         );
//! 
//!         // updates each presynaptic neuron's weights given the timestep
//!         // and whether the neuron is spiking along with the state of the
//!         // postsynaptic neuron
//!         for (n, i) in is_spikings.iter().enumerate() {
//!             if *i {
//!                 presynaptic_neurons[n].set_last_firing_time(Some(timestep));
//!                 <STDP as Plasticity<T, T, f32>>::update_weight(
//!                     stdp_params, 
//!                     &mut weights[n],
//!                     &presynaptic_neurons[n], 
//!                     postsynaptic_neuron
//!                 );
//!             }
//!         }
//!         
//!         // if postsynaptic neuron fires then update the firing time
//!         // and update the weight accordingly
//!         if is_spiking {
//!             postsynaptic_neuron.set_last_firing_time(Some(timestep));
//!             for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
//!                 <STDP as Plasticity<T, T, f32>>::update_weight(
//!                     stdp_params, 
//!                     &mut weights[n_neuron],
//!                     i, 
//!                     postsynaptic_neuron
//!                 );
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
//! use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;
//! use spiking_neural_networks::neuron::iterate_and_spike::{
//!     GaussianFactor, GaussianParameters, IsSpiking, Timestep,
//!     CurrentVoltage, GapConductance, IterateAndSpike, 
//!     LastFiringTime, NeurotransmitterConcentrations, LigandGatedChannels, 
//!     ReceptorKinetics, NeurotransmitterKinetics, Neurotransmitters,
//!     ApproximateNeurotransmitter, ApproximateReceptor,
//! };
//! use spiking_neural_networks::neuron::ion_channels::{
//!     BasicGatingVariable, IonChannel, TimestepIndependentIonChannel,
//! };
//!  
//! 
//! /// A calcium channel with reduced dimensionality
//! #[derive(Debug, Clone, Copy)]
//! pub struct ReducedCalciumChannel {
//!     /// Conductance of calcium channel (nS)
//!     pub g_ca: f32,
//!     /// Reversal potential (mV)
//!     pub v_ca: f32,
//!     /// Gating variable steady state
//!     pub m_ss: f32,
//!     /// Tuning parameter
//!     pub v_1: f32,
//!     /// Tuning parameter
//!     pub v_2: f32,
//!     /// Current output
//!     pub current: f32,
//! }
//! 
//! impl TimestepIndependentIonChannel for ReducedCalciumChannel {
//!     fn update_current(&mut self, voltage: f32) {
//!         self.m_ss = 0.5 * (1. + ((voltage - self.v_1) / self.v_2).tanh());
//! 
//!         self.current = self.g_ca * self.m_ss * (voltage - self.v_ca);
//!     }
//! 
//!     fn get_current(&self) -> f32 {
//!         self.current
//!     }
//! }
//! 
//! /// A potassium channel based on steady state calculations
//! #[derive(Debug, Clone, Copy)]
//! pub struct KSteadyStateChannel {
//!     /// Conductance of potassium channel (nS)
//!     pub g_k: f32,
//!     /// Reversal potential (mV)
//!     pub v_k: f32,
//!     /// Gating variable
//!     pub n: f32,
//!     /// Gating variable steady state
//!     pub n_ss: f32,
//!     /// Gating decay
//!     pub t_n: f32,
//!     /// Reference frequency
//!     pub phi: f32,
//!     /// Tuning parameter
//!     pub v_3: f32,
//!     /// Tuning parameter
//!     pub v_4: f32,
//!     /// Current output
//!     pub current: f32
//! }
//! 
//! impl KSteadyStateChannel {
//!     fn update_gating_variables(&mut self, voltage: f32) {
//!         self.n_ss = 0.5 * (1. + ((voltage - self.v_3) / self.v_4).tanh());
//!         self.t_n = 1. / (self.phi * ((voltage - self.v_3) / (2. * self.v_4)).cosh());
//!     }
//! }
//! 
//! impl IonChannel for KSteadyStateChannel { 
//!     fn update_current(&mut self, voltage: f32, dt: f32) {
//!         self.update_gating_variables(voltage);
//! 
//!         let n_change = ((self.n_ss - self.n) / self.t_n) * dt;
//! 
//!         self.n += n_change;
//! 
//!         self.current = self.g_k * self.n * (voltage - self.v_k);
//!     }
//! 
//!     fn get_current(&self) -> f32 {
//!         self.current
//!     }
//! }
//! 
//! /// An implementation of a leak channel
//! #[derive(Debug, Clone, Copy)]
//! pub struct LeakChannel {
//!     /// Conductance of leak channel (nS)
//!     pub g_l: f32,
//!     /// Reversal potential (mV)
//!     pub v_l: f32,
//!     /// Current output
//!     pub current: f32
//! }
//! 
//! impl TimestepIndependentIonChannel for LeakChannel {
//!     fn update_current(&mut self, voltage: f32) {
//!         self.current = self.g_l * (voltage - self.v_l);
//!     }
//! 
//!     fn get_current(&self) -> f32 {
//!         self.current
//!     }
//! }
//! 
//! #[derive(Debug, Clone, IterateAndSpikeBase)]
//! pub struct MorrisLecarNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
//!     /// Membrane potential (mV)
//!     pub current_voltage: f32, 
//!     /// Voltage threshold (mV)
//!     pub v_th: f32,
//!     /// Initial voltage value (mV)
//!     pub v_init: f32,
//!     /// Controls conductance of input gap junctions
//!     pub gap_conductance: f32,
//!     /// Calcium channel
//!     pub ca_channel: ReducedCalciumChannel,
//!     /// Potassium channel
//!     pub k_channel: KSteadyStateChannel,
//!     /// Leak channel
//!     pub leak_channel: LeakChannel,
//!     /// Membrane capacitance (nF)
//!     pub c_m: f32,
//!     /// Timestep in (ms)
//!     pub dt: f32,
//!     /// Whether the neuron is spiking
//!     pub is_spiking: bool,
//!     /// Whether the voltage was increasing in the last step
//!     pub was_increasing: bool,
//!     /// Last timestep the neuron has spiked
//!     pub last_firing_time: Option<usize>,
//!     /// Parameters used in generating noise
//!     pub gaussian_params: GaussianParameters,
//!     /// Postsynaptic neurotransmitters in cleft
//!     pub synaptic_neurotransmitters: Neurotransmitters<T>,
//!     /// Ionotropic receptor ligand gated channels
//!     pub ligand_gates: LigandGatedChannels<R>,
//! }
//! 
//! impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> MorrisLecarNeuron<T, R> {
//!     /// Updates channel states based on current voltage
//!     pub fn update_channels(&mut self) {
//!         self.ca_channel.update_current(self.current_voltage);
//!         self.k_channel.update_current(self.current_voltage, self.dt);
//!         self.leak_channel.update_current(self.current_voltage);
//!     }
//!     
//!     /// Calculates change in voltage given an input current
//!     pub fn get_dv_change(&self, i: f32) -> f32 {
//!         (i - self.leak_channel.current - self.ca_channel.current - self.k_channel.current)
//!         * (self.dt / self.c_m)
//!     }
//! 
//!     // checks if neuron is currently spiking but seeing if the neuron is increasing in
//!     // reference to the last inputted voltage and if it is above a certain
//!     // voltage threshold, if it is then the neuron is considered spiking
//!     // and `true` is returned, otherwise `false` is returned
//!     fn handle_spiking(&mut self, last_voltage: f32) -> bool {
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
//! impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for MorrisLecarNeuron<T, R> {
//!     fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {
//!         self.synaptic_neurotransmitters.get_concentrations()
//!     }
//! 
//!     // updates voltage and adaptive values as well as the 
//!     // neurotransmitters, receptor current is not factored in,
//!     // and spiking is handled and returns whether it is currently spiking
//!     fn iterate_and_spike(&mut self, input_current: f32) -> bool {
//!         self.update_channels();
//! 
//!         let last_voltage = self.current_voltage;
//!         self.current_voltage += self.get_dv_change(input_current);
//! 
//!         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);
//! 
//!         self.handle_spiking(last_voltage)
//!     }
//! 
//!     // updates voltage and adaptive values as well as the 
//!     // neurotransmitters, receptor current is factored in and receptor gating
//!     // is updated spiking is handled at the end of the method and 
//!     // returns whether it is currently spiking
//!     fn iterate_with_neurotransmitter_and_spike(
//!         &mut self, 
//!         input_current: f32, 
//!         t_total: &NeurotransmitterConcentrations,
//!     ) -> bool {
//!         self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
//!         self.ligand_gates.set_receptor_currents(self.current_voltage);
//!         
//!         self.update_channels();
//! 
//!         let last_voltage = self.current_voltage;
//!         let receptor_current = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);
//!         self.current_voltage += self.get_dv_change(input_current) + receptor_current;
//! 
//!         self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);
//! 
//!         self.handle_spiking(last_voltage)
//!     }
//! }
//! ```
//! 
//! ### Custom `NeurotransmitterKinetics` implementation
//! 
//! ```rust
//! use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterKinetics;
//! 
//! /// An approximation of neurotransmitter kinetics that sets the concentration to the 
//! /// maximal value when a spike is detected (input `voltage` is greater than `v_th`) and
//! /// slowly through exponential decay that scales based on the `decay_constant` and `dt`
//! #[derive(Debug, Clone, Copy)]
//! pub struct ExponentialDecayNeurotransmitter {
//!     /// Maximal neurotransmitter concentration (mM)
//!     pub t_max: f32,
//!     /// Current neurotransmitter concentration (mM)
//!     pub t: f32,
//!     /// Voltage threshold for detecting spikes (mV)
//!     pub v_th: f32,
//!     /// Amount to decay neurotransmitter concentration by
//!     pub decay_constant: f32,
//! }
//! 
//! // used to determine when voltage spike occurs
//! fn heaviside(x: f32) -> f32 {
//!     if x > 0. {
//!         1.
//!     } else {
//!         0.
//!     }
//! }
//! 
//! // calculate change in concentration
//! fn exp_decay(x: f32, l: f32, dt: f32) -> f32 {
//!     -x * (dt / -l).exp()
//! }
//! 
//! impl NeurotransmitterKinetics for ExponentialDecayNeurotransmitter {
//!     fn apply_t_change(&mut self, voltage: f32, dt: f32) {
//!         let t_change = exp_decay(self.t, self.decay_constant, dt);
//!         // add change and account for spike
//!         self.t += t_change + (heaviside(voltage - self.v_th) * self.t_max);
//!         self.t = self.t_max.min(self.t.max(0.)); // clamp values
//!     }
//! 
//!     fn get_t(&self) -> f32 {
//!         self.t
//!     }
//! 
//!     fn set_t(&mut self, t: f32) {
//!         self.t = t;
//!     }
//! }
//! ```
//! 
//! ### Custom `ReceptorKinetics` implementation
//! 
//! ```rust
//! use spiking_neural_networks::neuron::iterate_and_spike::{
//!     ReceptorKinetics, AMPADefault, GABAaDefault, GABAbDefault, NMDADefault,
//! };
//! 
//! /// Receptor dynamics approximation that sets the receptor
//! /// gating value to the inputted neurotransmitter concentration and
//! /// then exponentially decays the receptor over time
//! #[derive(Debug, Clone, Copy)]
//! pub struct ExponentialDecayReceptor {
//!     /// Maximal receptor gating value
//!     pub r_max: f32,
//!     /// Receptor gating value
//!     pub r: f32,
//!     /// Amount to decay neurotransmitter concentration by
//!     pub decay_constant: f32,
//! }
//! 
//! // calculate change in receptor gating variable over time
//! fn exp_decay(x: f32, l: f32, dt: f32) -> f32 {
//!     -x * (dt / -l).exp()
//! }
//! 
//! impl ReceptorKinetics for ExponentialDecayReceptor {
//!     fn apply_r_change(&mut self, t: f32, dt: f32) {
//!         // calculate and apply change
//!         self.r += exp_decay(self.r, self.decay_constant, dt) + t;
//!         self.r = self.r_max.min(self.r.max(0.)); // clamp values
//!     }
//!
//!     fn get_r(&self) -> f32 {
//!         self.r
//!     }
//!
//!     fn set_r(&mut self, r: f32) {
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
//!                 }
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
pub mod error;
pub mod fitting;
pub mod ga;
pub mod graph;
pub mod reinforcement;
pub mod neuron;
