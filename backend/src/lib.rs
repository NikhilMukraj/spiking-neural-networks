//! # Spiking Neural Networks
//! 
//! `spiking_neural_networks` is a package focused on designing neuron models
//! with neurotransmission and calculating dynamics between neurons over time.
//! Neuronal dynamics are made using traits so they can be expanded via the 
//! type system to add new dynamics for different neurotransmitters, receptors
//! or neuron models. Currently implements system for spike trains, spike time depedent
//! plasticity, basic attractors and dynamics for neurons connected in a lattice. 
//! See below for examples and how to add custom models.
//!
//! ## Examples:
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
//! // copy example from examples folder
//! 
//! fn test_isolated_stdp<T: IterateAndSpike>(
//! presynaptic_neurons: &mut Vec<T>,
//! postsynaptic_neuron: &mut T,
//! iterations: usize,
//! input_current: f64,
//! input_current_deviation: f64,
//! weight_params: &GaussianParameters,
//! do_receptor_kinetics: bool,
//! ) -> HashMap<String, Vec<f64>> {
//! 
//! }
//! ```
//! 
//! ### Custom `IterateAndSpike` implementation
//! 
//! ```rust
//! #[derive(Debug, Clone)]
//! pub struct FitzHughNagumo<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
//!     // should include bursting, should have parameter to set whether bursting occurs,
//!     // like multiplying by 0 to get rid of bursting term
//! }
//! 
//! impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> for FitzHugoNagumo {
//! 
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
