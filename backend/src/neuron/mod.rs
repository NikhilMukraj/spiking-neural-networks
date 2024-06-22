//! A collection of various neuron models as well as methods to connect the neurons together,
//! simulate neuronal dynamics including basic electrical synapses, neurotransmission,
//! the effect of ligand gated channels, other channels, integrate and fire models, 
//! and conductance based models.
//! 
//! Also includes various traits that can be implemented to customize neurotransmitter, receptor,
//! neuronal, and spike train dynamics to the enable the swapping out different models
//! with new ones without having to rewrite functionality.

use std::{
    f64::consts::PI, 
    collections::{HashMap, HashSet},
    io::Result,
};
pub mod integrate_and_fire;
pub mod hodgkin_huxley;
pub mod fitzhugh_nagumo;
pub mod attractors;
pub mod spike_train;
use spike_train::{SpikeTrain, NeuralRefractoriness};
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, Potentiation, LastFiringTime, STDP,
    IterateAndSpike, PotentiationType, Neurotransmitters, 
    NeurotransmitterType, NeurotransmitterConcentrations,
    weight_neurotransmitter_concentration, aggregate_neurotransmitter_concentrations,
};
use crate::graph::{GraphFunctionality, GraphPosition};


/// Calculates the current between two neurons based on the voltage and
/// the gap conductance of the synapse
fn gap_junction<T: CurrentVoltage, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    postsynaptic_neuron.get_gap_conductance() * 
    (presynaptic_neuron.get_current_voltage() - postsynaptic_neuron.get_current_voltage())
}

/// Calculates the current between two neurons based on the voltage,
/// the gap conductance of the synapse, and the potentiation of the
/// presynaptic neuron, both neurons should implement `CurrentVoltage`,
/// the presynaptic neuron should implement `Potentation`, and
/// the postsynaptic neuron should implemenent `GapConductance`
pub fn signed_gap_junction<T: CurrentVoltage + Potentiation, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    sign * gap_junction(presynaptic_neuron, postsynaptic_neuron)
}

/// Calculates one iteration of two coupled neurons where the presynaptic neuron
/// has a static input current while the postsynaptic neuron takes
/// the current input and neurotransmitter input from the presynaptic neuron,
/// returns whether each neuron is spiking
/// 
/// - `presynaptic_neuron` : a neuron that implements `IterateAndSpike`
/// 
/// - `postsynaptic_neuron` : a neuron that implements `IterateAndSpike`
/// 
/// - `do_receptor_kinetics` : use `true` to update receptor gating values of 
/// the neurons based on neurotransmitter input during the simulation
/// 
/// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    input_current: f64,
    do_receptor_kinetics: bool,
    gaussian: bool,
) -> (bool, bool) {
    let (t_total, post_current, input_current) = if gaussian {
        let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
        let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();

        let input_current = input_current * pre_gaussian_factor;

        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);

            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    } else {
        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    };

    let pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        t_total.as_ref(),
    );

    (pre_spiking, post_spiking)
}

/// Calculates the input to the postsynaptic neuron given a spike train
/// and the potenation of the spike train as well as the current timestep 
/// of the simulation
pub fn spike_train_gap_juncton<T: SpikeTrain + Potentiation, U: GapConductance>(
    presynaptic_neuron: &T,
    postsynaptic_neuron: &U,
    timestep: usize,
) -> f64 {
    let (v_max, v_resting) = presynaptic_neuron.get_height();

    if let None = presynaptic_neuron.get_last_firing_time() {
        return v_resting;
    }

    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    let last_firing_time = presynaptic_neuron.get_last_firing_time().unwrap();
    let refractoriness_function = presynaptic_neuron.get_refractoriness_function();
    let dt = presynaptic_neuron.get_refractoriness_timestep();
    let conductance = postsynaptic_neuron.get_gap_conductance();

    sign * conductance * refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
}

/// Calculates one iteration of two coupled neurons where the presynaptic neuron
/// has a spike train input while the postsynaptic neuron takes
/// the current input and neurotransmitter input from the presynaptic neuron,
/// also updates the last firing times of each neuron and spike train given the
/// current timestep of the simulation, returns whether each neuron is spiking
/// 
/// - `spike_train` : a spike train that implements `Spiketrain`
/// 
/// - `presynaptic_neuron` : a neuron that implements `IterateAndSpike`
/// 
/// - `postsynaptic_neuron` : a neuron that implements `IterateAndSpike`
/// 
/// - `timestep` : the current timestep of the simulation
/// 
/// - `do_receptor_kinetics` : use `true` to update receptor gating values of 
/// the neurons based on neurotransmitter input during the simulation
/// 
/// - `gaussian` : use `true` to add normally distributed random noise to inputs of simulations
pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
    spike_train: &mut T,
    presynaptic_neuron: &mut U, 
    postsynaptic_neuron: &mut U,
    timestep: usize,
    do_receptor_kinetics: bool,
    gaussian: bool,
) -> (bool, bool, bool) {
    let input_current = spike_train_gap_juncton(spike_train, presynaptic_neuron, timestep);

    let (pre_t_total, post_t_total, current) = if gaussian {
        let pre_gaussian_factor = presynaptic_neuron.get_gaussian_factor();
        let post_gaussian_factor = postsynaptic_neuron.get_gaussian_factor();

        let pre_t_total = if do_receptor_kinetics {
            let mut t = spike_train.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, pre_gaussian_factor);

            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_gaussian_factor);

            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    } else {
        let pre_t_total = if do_receptor_kinetics {
            let t = spike_train.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    };

    let spike_train_spiking = spike_train.iterate();   
    if spike_train_spiking {
        spike_train.set_last_firing_time(Some(timestep));
    }
    
    let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        input_current,
        pre_t_total.as_ref(),
    );
    if pre_spiking {
        presynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        current,
        post_t_total.as_ref(),
    ); 
    if post_spiking {
        postsynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    (spike_train_spiking, pre_spiking, post_spiking)
}

/// Calculates and returns the change in weight based off of STDP (spike time dependent plasticity)
/// given one presynaptic neuron that implements `LastFiringTime` to get the last time it fired
/// as well as a postsynaptic neuron that implements `STDP`
pub fn update_weight_stdp<T: LastFiringTime, U: STDP>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let mut delta_w: f64 = 0.;

    match (presynaptic_neuron.get_last_firing_time(), postsynaptic_neuron.get_last_firing_time()) {
        (Some(t_pre), Some(t_post)) => {
            let (t_pre, t_post): (f64, f64) = (t_pre as f64, t_post as f64);

            if t_pre < t_post {
                delta_w = postsynaptic_neuron.get_stdp_params().a_plus * 
                    (-1. * (t_pre - t_post).abs() / postsynaptic_neuron.get_stdp_params().tau_plus).exp();
            } else if t_pre > t_post {
                delta_w = -1. * postsynaptic_neuron.get_stdp_params().a_minus * 
                    (-1. * (t_post - t_pre).abs() / postsynaptic_neuron.get_stdp_params().tau_minus).exp();
            }
        },
        _ => {}
    };

    return delta_w;
}

/// Handles history of a lattice
pub trait LatticeHistory: Default {
    /// Stores the current state of the lattice given the cell grid
    fn update<T: IterateAndSpike>(&mut self, state: &Vec<Vec<T>>);
}

/// Stores EEG value history
#[derive(Debug, Clone)]
pub struct EEGHistory {
    /// EEG values
    history: Vec<f64>,
    /// Voltage from EEG equipment (mV)
    reference_voltage: f64,
    /// Distance from neurons to equipment (mm)
    distance: f64,
    /// Conductivity of medium (S/mm)
    conductivity: f64,
}

impl Default for EEGHistory {
    fn default() -> Self {
        EEGHistory {
            history: Vec::new(),
            reference_voltage: 0.007, // 0.007 mV or 7 uV
            distance: 0.8, // 0.8 mm
            conductivity: 251., // 251 S/mm or 0.251 S/m
        }
    }
}


fn get_grid_voltages<T: CurrentVoltage>(grid: &Vec<Vec<T>>) -> Vec<Vec<f64>> {
    grid.iter()
        .map(|i| {
            i.iter()
                .map(|j| j.get_current_voltage())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
}

impl LatticeHistory for EEGHistory {
    fn update<T: IterateAndSpike>(&mut self, state: &Vec<Vec<T>>) {
        let mut total_current = 0.;
        let voltages = get_grid_voltages(state);

        for row in voltages {
            for value in row {
                total_current += value - self.reference_voltage;
            }
        }
    
        let eeg_value = (1. / (4. * PI * self.conductivity * self.distance)) * total_current;

        self.history.push(eeg_value);
    }
}

/// Stores history as grid of voltages
#[derive(Debug, Clone)]
pub struct GridVoltageHistory {
    /// Voltage history
    history: Vec<Vec<Vec<f64>>>
}

impl Default for GridVoltageHistory {
    fn default() -> Self {
        GridVoltageHistory { history: Vec::new() }
    }
}

impl LatticeHistory for GridVoltageHistory {
    fn update<T: IterateAndSpike>(&mut self, state: &Vec<Vec<T>>) {
        self.history.push(get_grid_voltages::<T>(state));
    }
}

macro_rules! impl_reset_timing  {
    () => {
        /// Resets the last firing time of the neurons to `None`
        /// and resets the `internal_clock` to `0`
        pub fn reset_timing(&mut self) {
            self.internal_clock = 0;
            self.cell_grid.iter_mut()
                .for_each(|i| {
                    i.iter_mut()
                        .for_each(|j| {
                            j.set_last_firing_time(None)
                    })
                });
        } 
    };
}

/// Lattice of `IterateAndSpike` neurons
#[derive(Debug, Clone)]
pub struct Lattice<T: IterateAndSpike, U: GraphFunctionality, V: LatticeHistory> {
    /// Grid of neurons
    pub cell_grid: Vec<Vec<T>>,
    /// Graph connecting internal neurons and storing weights between neurons
    pub graph: U,
    /// History of grid
    pub grid_history: V,
    /// Whether to update graph's history of weights
    pub update_graph_history: bool,
    /// Whether to update grid's history
    pub update_grid_history: bool,
    /// Whether to update weights with STDP when iterating
    pub do_stdp: bool,
    /// Whether to update receptor gating values based on neurotransmitter
    pub do_receptor_kinetics: bool,
    /// Whether to add normally distributed random noise
    pub gaussian: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<T: IterateAndSpike, U: GraphFunctionality, V: LatticeHistory> Lattice<T, U, V> {
    impl_reset_timing!();

    /// Calculates electrical input value from positions
    fn calculate_internal_input_from_positions(
        &self,
        position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>, 
    ) -> f64 {
        let (x, y) = position.pos;
        let postsynaptic_neuron = &self.cell_grid[x][y];

        let mut input_val = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;
                let input_cell = &self.cell_grid[pos_x][pos_y];

                let final_input = signed_gap_junction(input_cell, postsynaptic_neuron);
                
                final_input * self.graph.lookup_weight(&input_position, position).unwrap().unwrap()
            })
            .sum();

        if self.gaussian {
            input_val *= self.cell_grid[x][y].get_gaussian_factor();
        }

        input_val /= input_positions.len() as f64;

        return input_val;
    }

    /// Calculates neurotransmitter input value from positions
    fn calculate_internal_neurotransmitter_input_from_positions(
        &self,
        position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>, 
    ) -> NeurotransmitterConcentrations {
        let input_vals = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;
                let input_cell = &self.cell_grid[pos_x][pos_y];

                let mut final_input = input_cell.get_neurotransmitter_concentrations();
                let weight = self.graph.lookup_weight(&input_position, position).unwrap().unwrap();
                
                weight_neurotransmitter_concentration(&mut final_input, weight);

                final_input
            })
            .collect::<Vec<NeurotransmitterConcentrations>>();

        let mut input_val = aggregate_neurotransmitter_concentrations(&input_vals);

        if self.gaussian {
            let (x, y) = position.pos;
            weight_neurotransmitter_concentration(&mut input_val, self.cell_grid[x][y].get_gaussian_factor());
        }

        weight_neurotransmitter_concentration(&mut input_val, (1 / input_positions.len()) as f64);

        return input_val;
    }

    /// Gets all internal electrical inputs 
    fn get_internal_electrical_inputs(&self) -> HashMap<GraphPosition, f64> {
        // eventually convert to this, same with neurotransmitter input
        // let inputs: HashMap<Position, f64> = graph
        //     .get_every_node()
        //     .par_iter()
        //     .map(|&pos| {
        //     // .. calculating input
        //     (pos, change)
        //     });
        //     .collect();

        self.graph.get_every_node()
            .iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(&pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_input_from_positions(
                    &pos,
                    &input_positions,
                );

                (*pos, input)
            })
            .collect()
    }

    /// Gets all internal neurotransmitter inputs 
    fn get_internal_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f64>, Option<HashMap<GraphPosition, NeurotransmitterConcentrations>>) {
        let neurotransmitter_inputs = match self.do_receptor_kinetics {
            true => {
                let neurotransmitters: HashMap<GraphPosition, NeurotransmitterConcentrations> = self.graph.get_every_node()
                    .iter()
                    .map(|&pos| {
                        let input_positions = self.graph.get_incoming_connections(&pos)
                            .expect("Cannot find position");

                        let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                            &pos,
                            &input_positions,
                        );

                        (pos, neurotransmitter_input)
                    })
                    .collect();
                    
                Some(neurotransmitters)
            },
            false => None,
        };

        let inputs = self.get_internal_electrical_inputs();

        (inputs, neurotransmitter_inputs)
    }

    /// Updates internal weights based on STDP
    fn update_weights_from_spiking_neuron(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<()> {
        let given_neuron = &self.cell_grid[x][y];
        
        let input_positions = self.graph.get_incoming_connections(&pos)?;

        for i in input_positions {
            let (x_in, y_in) = i.pos;
            let current_weight = self.graph.lookup_weight(&i, &pos)?.unwrap();
                                        
            self.graph.edit_weight(
                &i, 
                &pos, 
                Some(current_weight + update_weight_stdp(&self.cell_grid[x_in][y_in], given_neuron))
            )?;
        }

        let out_going_connections = self.graph.get_outgoing_connections(&pos)?;

        for i in out_going_connections {
            let (x_out, y_out) = i.pos;
            let current_weight = self.graph.lookup_weight(&pos, &i)?.unwrap();

            self.graph.edit_weight(
                &pos, 
                &i, 
                Some(current_weight + update_weight_stdp(given_neuron, &self.cell_grid[x_out][y_out]))
            )?; 
        }

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of electrical and neurotransmitter inputs
    pub fn iterate(
        &mut self, 
        inputs: &HashMap<GraphPosition, f64>, 
        neurotransmitter_inputs: &Option<HashMap<GraphPosition, NeurotransmitterConcentrations>>,
    ) -> Result<()> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos.pos;
            let input_value = *inputs.get(&pos).unwrap();

            let input_neurotransmitter = match neurotransmitter_inputs {
                Some(ref neurotransmitter_hashmap) => Some(neurotransmitter_hashmap.get(&pos).unwrap()),
                None => None,
            };

            let is_spiking = self.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                input_value, input_neurotransmitter,
            );

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_stdp && is_spiking {
                self.update_weights_from_spiking_neuron(x, y, &pos)?;
            } 
        }

        if self.update_graph_history {
            self.graph.update_history();
        }
        if self.update_grid_history {
            self.grid_history.update(&self.cell_grid);
        }
        self.internal_clock += 1;

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of only electrical inputs
    pub fn iterate_electrical_only(
        &mut self,
        inputs: &HashMap<GraphPosition, f64>,
    ) -> Result<()> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos.pos;
            let input_value = *inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_and_spike(input_value);

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_stdp && is_spiking {
                self.update_weights_from_spiking_neuron(x, y, &pos)?;
            } 
        }

        if self.update_graph_history {
            self.graph.update_history();
        }
        if self.update_grid_history {
            self.grid_history.update(&self.cell_grid);
        }
        self.internal_clock += 1;

        Ok(())
    }

    /// Iterates the lattice based only on internal connections for a given amount of time
    pub fn run_lattice(
        &mut self, 
        iterations: usize,
    ) -> Result<()> {
        for _ in 0..iterations {       
            let (inputs, neurotransmitter_inputs) = self.get_internal_neurotransmitter_inputs();
    
            self.iterate(&inputs, &neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates lattice based only on internal connections for a given amount of time using
    /// only electrical inputs
    pub fn run_lattice_electrical_only(
        &mut self,
        iterations: usize,
    ) -> Result<()> {
        for _ in 0..iterations {
            let inputs = self.get_internal_electrical_inputs();

            self.iterate_electrical_only(&inputs)?;
        }

        Ok(())
    }
}

/// Handles history of a spike train lattice
pub trait SpikeTrainLatticeHistory {
    /// Stores the current state of the lattice given the cell grid
    fn update<T: SpikeTrain>(&mut self, state: &Vec<Vec<T>>);
}

/// Stores history as a grid of voltages
#[derive(Debug, Clone)]
pub struct SpikeTrainGridHistory {
    /// Voltage history
    history: Vec<Vec<Vec<f64>>>,
}

impl Default for SpikeTrainGridHistory {
    fn default() -> Self {
        SpikeTrainGridHistory { history: Vec::new() }
    }
}

impl SpikeTrainLatticeHistory for SpikeTrainGridHistory {
    fn update<T: SpikeTrain>(&mut self, state: &Vec<Vec<T>>) {
        self.history.push(get_grid_voltages::<T>(&state));
    }
}

/// Lattice of `SpikeTrain` neurons
#[derive(Debug, Clone)]
pub struct SpikeTrainLattice<T: SpikeTrain, U: SpikeTrainLatticeHistory> {
    /// Grid of spike trains
    cell_grid: Vec<Vec<T>>,
    /// History of grid states
    grid_history: U,
    /// Whether to update grid history
    update_grid_history: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    internal_clock: usize,
}

impl<T: SpikeTrain, U: SpikeTrainLatticeHistory> SpikeTrainLattice<T, U> {
    impl_reset_timing!();

    /// Iterates one simulation timestep lattice
    fn iterate(&mut self) {
        self.cell_grid.iter_mut()
            .for_each(|i| {
                i.iter_mut()
                    .for_each(|j| {
                        let is_spiking = j.iterate();
                        if is_spiking {
                            j.set_last_firing_time(Some(self.internal_clock))
                        }
                })
            });

        if self.update_grid_history {
            self.grid_history.update(&self.cell_grid);
        }
        self.internal_clock += 1;
    }

    /// Iterates simulation for the given amount of time
    pub fn run_lattice(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.iterate();
        }
    }
}
