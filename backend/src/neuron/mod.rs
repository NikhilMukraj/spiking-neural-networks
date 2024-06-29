//! A collection of various neuron models as well as methods to connect the neurons together,
//! simulate neuronal dynamics including basic electrical synapses, neurotransmission,
//! the effect of ligand gated channels, other channels, integrate and fire models, 
//! and conductance based models.
//! 
//! Includes methods to represent a group of neurons together in a lattice or multiple connected
//! groups of neurons through a network of lattices.
//! 
//! Also includes various traits that can be implemented to customize neurotransmitter, receptor,
//! neuronal, and spike train dynamics to the enable the swapping out different models
//! with new ones without having to rewrite functionality.

use std::{
    collections::{
        hash_map::{Values, ValuesMut}, 
        HashMap, HashSet
    }, 
    f64::consts::PI, 
    result::Result,
};
use rand::{Rng, seq::SliceRandom};
pub mod integrate_and_fire;
pub mod hodgkin_huxley;
pub mod fitzhugh_nagumo;
pub mod attractors;
pub mod spike_train;
use spike_train::{SpikeTrain, NeuralRefractoriness};
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, IterateAndSpike, LastFiringTime, GaussianParameters,
    Potentiation, PotentiationType, STDP, NeurotransmitterConcentrations, 
    NeurotransmitterType, Neurotransmitters, aggregate_neurotransmitter_concentrations, 
    weight_neurotransmitter_concentration, 
};
use crate::error::{GraphError, LatticeNetworkError, LatticeNetworkErrorKind};
use crate::graph::{Graph, GraphPosition, AdjacencyMatrix};


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
/// presynaptic neuron, both neurons should implement [`CurrentVoltage`],
/// the presynaptic neuron should implement [`Potentiation`], and
/// the postsynaptic neuron should implemenent [`GapConductance`]
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
/// - `presynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
/// 
/// - `postsynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
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
    let effect = refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt);

    sign * conductance * effect
}

/// Calculates one iteration of two coupled neurons where the presynaptic neuron
/// has a spike train input while the postsynaptic neuron takes
/// the current input and neurotransmitter input from the presynaptic neuron,
/// also updates the last firing times of each neuron and spike train given the
/// current timestep of the simulation, returns whether each neuron is spiking
/// 
/// - `spike_train` : a spike train that implements [`SpikeTrain`]
/// 
/// - `presynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
/// 
/// - `postsynaptic_neuron` : a neuron that implements [`IterateAndSpike`]
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
/// given one presynaptic neuron that implements [`LastFiringTime`] to get the last time it fired
/// as well as a postsynaptic neuron that implements [`STDP`]
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
    /// Resets history
    fn reset(&mut self);
}

/// Stores EEG value history
#[derive(Debug, Clone)]
pub struct EEGHistory {
    /// EEG values
    pub history: Vec<f64>,
    /// Voltage from EEG equipment (mV)
    pub reference_voltage: f64,
    /// Distance from neurons to equipment (mm)
    pub distance: f64,
    /// Conductivity of medium (S/mm)
    pub conductivity: f64,
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

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Stores history as grid of voltages
#[derive(Debug, Clone)]
pub struct GridVoltageHistory {
    /// Voltage history
    pub history: Vec<Vec<Vec<f64>>>
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

    fn reset(&mut self) {
        self.history.clear();
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

fn positions_within_square(
    center_row: usize, 
    center_col: usize, 
    extent: usize, 
    size: (usize, usize)
) -> Vec<(usize, usize)> {
    let (row_length, col_length) = size;
    let mut positions = Vec::new();

    for row in center_row.saturating_sub(extent)..=(center_row + extent) {
        for col in center_col.saturating_sub(extent)..=(center_col + extent) {
            if (row != center_row || col != center_col) && (row < row_length && col < col_length) {
                positions.push((row, col));
            }
        }
    }

    positions
}

fn randomly_select_positions(
    mut positions: Vec<(usize, usize)>, 
    id: usize, 
    num_to_select: usize
) -> Vec<GraphPosition> {
    let mut rng = rand::thread_rng();

    positions.shuffle(&mut rng);
    positions.truncate(num_to_select);

    positions.iter()
        .map(|i| 
            GraphPosition { 
                id: id, 
                pos: *i,
            }
        )
        .collect()
}

/// Lattice of [`IterateAndSpike`] neurons
#[derive(Debug, Clone)]
pub struct Lattice<T: IterateAndSpike, U: Graph, V: LatticeHistory> {
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

impl<T: IterateAndSpike, U: Graph, V: LatticeHistory> Default for Lattice<T, U, V> {
    fn default() -> Self {
        Lattice {
            cell_grid: vec![],
            graph: U::default(),
            grid_history: V::default(),
            update_graph_history: false,
            update_grid_history: false,
            do_stdp: false,
            do_receptor_kinetics: false,
            gaussian: false,
            internal_clock: 0,
        }
    }
}

impl<T: IterateAndSpike> Lattice<T, AdjacencyMatrix, GridVoltageHistory> {
    // Generates a default lattice implementation given a neuron type
    pub fn default_impl() -> Self {
        Lattice::default()
    }
}

impl<T: IterateAndSpike, U: Graph, V: LatticeHistory> Lattice<T, U, V> {
    impl_reset_timing!();

    /// Gets id of lattice [`Graph`]
    pub fn get_id(&self) -> usize {
        self.graph.get_id()
    }

    /// Sets id of lattice [`Graph`] given an id
    pub fn set_id(&mut self, id: usize) {
        self.graph.set_id(id);
    }

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
        // convert on lattice network too
        // let inputs: HashMap<Position, f64> = graph
        //     .get_every_node()
        //     .par_iter()
        //     .map(|&pos| {
        //     // .. calculating input
        //     (pos, change)
        //     });
        //     .collect();

        self.graph.get_every_node_as_ref()
            .iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(&pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_input_from_positions(
                    &pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    /// Gets all internal neurotransmitter inputs 
    fn get_internal_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f64>, Option<HashMap<GraphPosition, NeurotransmitterConcentrations>>) {
        let neurotransmitter_inputs = match self.do_receptor_kinetics {
            true => {
                let neurotransmitters: HashMap<GraphPosition, NeurotransmitterConcentrations> = self.graph.get_every_node_as_ref()
                    .iter()
                    .map(|&pos| {
                        let input_positions = self.graph.get_incoming_connections(&pos)
                            .expect("Cannot find position");

                        let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                            &pos,
                            &input_positions,
                        );

                        (*pos, neurotransmitter_input)
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
    fn update_weights_from_spiking_neuron(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
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
    pub fn iterate_with_neurotransmission(
        &mut self, 
        inputs: &HashMap<GraphPosition, f64>, 
        neurotransmitter_inputs: &Option<HashMap<GraphPosition, NeurotransmitterConcentrations>>,
    ) -> Result<(), GraphError> {
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
    pub fn iterate(
        &mut self,
        inputs: &HashMap<GraphPosition, f64>,
    ) -> Result<(), GraphError> {
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

    /// Iterates the lattice based only on internal connections for a given amount of time using
    /// both electrical and neurotransmitter inputs, set `do_receptor_kinetics` to `true` to update
    /// receptor kinetics
    pub fn run_lattice_with_neurotransmission(
        &mut self, 
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {       
            let (inputs, neurotransmitter_inputs) = self.get_internal_electrical_and_neurotransmitter_inputs();
    
            self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates lattice based only on internal connections for a given amount of time using
    /// only electrical inputs
    pub fn run_lattice(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {
            let inputs = self.get_internal_electrical_inputs();

            self.iterate(&inputs)?;
        }

        Ok(())
    }

    fn generate_cell_grid(base_neuron: &T, num_rows: usize, num_cols: usize) -> Vec<Vec<T>> {
        (0..num_rows)
            .map(|_| {
                (0..num_cols)
                    .map(|_| {
                        base_neuron.clone()
                    })
                    .collect::<Vec<T>>()
            })
            .collect()
    }

    /// Populates a randomly connected lattice based on a single neuron to copy
    /// the parameters of throughout the lattice, the size of the lattice,
    /// and the radius from which neurons are allowed to be connected,
    /// use `&None` to connect every neuron with a weight of `1.` or
    /// provide gaussian parameters to generate randomly generated
    /// weights based on a normal distribution, (overwrites any pre-existing
    /// neurons or connections)
    pub fn populate_and_randomly_connect(
        &mut self,
        base_neuron: &T,
        num_rows: usize, 
        num_cols: usize,
        radius: usize,
        weight_params: &Option<GaussianParameters>,
    ) {
        let mut rng = rand::thread_rng();

        let id = self.get_id();

        self.graph = U::default();
        self.graph.set_id(id);
        self.cell_grid = Self::generate_cell_grid(base_neuron, num_rows, num_cols);

        for row in 0..num_rows {
            for col in 0..num_cols {
                let positions = positions_within_square(row, col, radius, (num_rows, num_cols));

                // let random_number: f64 = rng.gen_range(0..=1);
                // let scaled_number = 1. + random_number * (positions.len() as f64 - 1.);
                // let num_to_select = scaled_number as usize;

                let num_to_select = rng.gen_range(1..positions.len());
                let positions = randomly_select_positions(positions, self.graph.get_id(), num_to_select);
    
                self.graph.initialize_connections(
                    GraphPosition { id: self.graph.get_id(), pos: (row, col)}, 
                    positions, 
                    weight_params
                );
            }
        }
    }

    /// Populates a lattice given the dimensions and a base neuron to copy the parameters
    /// of without generating any connections within the graph, (overwrites any pre-existing
    /// neurons or connections)
    pub fn populate(&mut self, base_neuron: &T, num_rows: usize, num_cols: usize) {
        let id = self.get_id();

        self.graph = U::default();
        self.graph.set_id(id);
        self.cell_grid = Self::generate_cell_grid(base_neuron, num_rows, num_cols);

        for i in 0..num_rows {
            for j in 0..num_cols {
                self.graph.add_node(GraphPosition { id: self.graph.get_id(), pos: (i, j) })
            }
        }
    }

    /// Connects the neurons in a lattice together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`
    pub fn connect(
        &mut self, 
        connecting_conditional: fn((usize, usize), (usize, usize)) -> bool,
        weight_logic: Option<fn((usize, usize), (usize, usize)) -> f64>,
    ) {
        self.graph.get_every_node()
            .iter()
            .for_each(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(i.pos, j.pos) {
                        match weight_logic {
                            Some(logic) => {
                                self.graph.edit_weight(i, j, Some((logic)(i.pos, j.pos))).unwrap();
                            },
                            None => {
                                self.graph.edit_weight(i, j, Some(1.)).unwrap();
                            }
                        };
                    }
                }
            });
    }
}

/// Handles history of a spike train lattice
pub trait SpikeTrainLatticeHistory: Default {
    /// Stores the current state of the lattice given the cell grid
    fn update<T: SpikeTrain>(&mut self, state: &Vec<Vec<T>>);
    /// Resets history
    fn reset(&mut self);
}

/// Stores history as a grid of voltages
#[derive(Debug, Clone)]
pub struct SpikeTrainGridHistory {
    /// Voltage history
    pub history: Vec<Vec<Vec<f64>>>,
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

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Lattice of [`SpikeTrain`] neurons
#[derive(Debug, Clone)]
pub struct SpikeTrainLattice<T: SpikeTrain, U: SpikeTrainLatticeHistory> {
    /// Grid of spike trains
    pub cell_grid: Vec<Vec<T>>,
    /// History of grid states
    pub grid_history: U,
    /// Whether to update grid history
    pub update_grid_history: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
    /// Id for lattice
    pub id: usize,
}

impl<T: SpikeTrain, U: SpikeTrainLatticeHistory> Default for SpikeTrainLattice<T, U> {
    fn default() -> Self {
        SpikeTrainLattice {
            cell_grid: vec![],
            grid_history: U::default(),
            update_grid_history: false,
            internal_clock: 0,
            id: 0,
        }
    }
}

impl<T: SpikeTrain> SpikeTrainLattice<T, SpikeTrainGridHistory> {
    // Generates a default lattice implementation given a spike train type
    pub fn default_impl() -> Self {
        SpikeTrainLattice::default()
    }
}

impl<T: SpikeTrain, U: SpikeTrainLatticeHistory> SpikeTrainLattice<T, U> {
    impl_reset_timing!();

    /// Returns the identifier of the lattice
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Sets the identifier of the lattice
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }    

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

    pub fn populate(&mut self, base_spike_train: &T, num_rows: usize, num_cols: usize) {
        self.cell_grid = (0..num_rows)
            .map(|_| {
                (0..num_cols)
                    .map(|_| {
                        base_spike_train.clone()
                    })
                    .collect::<Vec<T>>()
            })
            .collect();
    }
}

/// Lattice Network
/// [`LatticeNetwork`] represents a series of lattices interconnected by a graph, each lattice
/// is associated to a unique identifier, [`Lattice`]s and [`SpikeTrainLattice`]s cannot have
/// the same identifiers, [`SpikeTrainLattice`]s cannot be postsynaptic because the spike trains
/// cannot take in an input
/// 
/// Use `connect` to generate connections between lattices:
/// ```rust
/// // assume lattice1 has id 0, lattice2 has id 1, and spike_train_lattice has id 2
/// fn one_to_one(x: (usize, usize), y: (usize, usize)) -> bool {
///     x == y
/// }
/// 
/// fn close_connect(x: (usize, usize), y: (usize, usize)) -> bool {
///     (x.0 - y.0).abs() < 2. && (x.1 - y.1).abs() < 2.
/// }
/// 
/// fn weight_function(x: (usize, usize), y: (usize, usize)) -> f64 {
///     (x.powf(2.) + y.powf(2.)).sqrt()
/// }
/// 
/// fn main() {
///     ...
///     let mut network = LatticeNetwork::generate_network(vec![lattice1, lattice2], vec![spike_train_lattice])?;
///     
///     // connects each corressponding neuron in the presynaptic lattice to a neuron in the
///     // postsynaptic lattice as long as their position is the same, scales the weight
///     // of the connection depending on the distance from each neuron
///     network.connect(0, 1, one_to_one, weight_function);
/// 
///     // connects the lattices in the same manner as before but does so in the opposite direction
///     network.connect(1, 0, one_to_one, weight_function);
/// 
///     // connections each spike train to a postsynaptic neuron in the postsynaptic lattice if 
///     // the neuron is close enough, sets each weight to 1.
///     network.connect(2, 0, close_connect, None);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LatticeNetwork
<T: IterateAndSpike, U: Graph, V: LatticeHistory, W: SpikeTrain, X: SpikeTrainLatticeHistory> 
{
    /// A hashmap of [`Lattice`]s associated with their respective identifier
    lattices: HashMap<usize, Lattice<T, U, V>>,
    /// A hashmap of [`SpikeTrainLattice`]s associated with their respective identifier
    spike_train_lattices: HashMap<usize, SpikeTrainLattice<W, X>>,
    /// An array of graphs connecting different lattices together
    connecting_graph: U,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<T, U, V, W, X> Default for LatticeNetwork<T, U, V, W, X>
where
    T: IterateAndSpike,
    U: Graph,
    V: LatticeHistory,
    W: SpikeTrain,
    X: SpikeTrainLatticeHistory,
{
    fn default() -> Self { 
        LatticeNetwork {
            lattices: HashMap::new(),
            spike_train_lattices: HashMap::new(),
            connecting_graph: U::default(),
            internal_clock: 0,
        }
    }
}

impl<T, U, V, W, X> LatticeNetwork<T, U, V, W, X>
where
    T: IterateAndSpike,
    U: Graph,
    V: LatticeHistory,
    W: SpikeTrain,
    X: SpikeTrainLatticeHistory,
{
    /// Generates a [`LatticeNetwork`] given lattices to use within the network, (all lattices
    /// must have unique id fields)
    pub fn generate_network(
        lattices: Vec<Lattice<T, U, V>>, 
        spike_train_lattices: Vec<SpikeTrainLattice<W, X>>
    ) -> Result<Self, LatticeNetworkError> {
        let mut network = LatticeNetwork::default();

        for lattice in lattices {
            network.add_lattice(lattice)?;
        }

        for spike_train_lattice in spike_train_lattices {
            network.add_spike_train_lattice(spike_train_lattice)?;
        }

        Ok(network)
    }

    /// Adds a [`Lattice`] to the network if the lattice has an id that is not already in the network
    pub fn add_lattice(
        &mut self, 
        lattice: Lattice<T, U, V>
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&lattice.get_id()) {
            return Err(LatticeNetworkError::new(LatticeNetworkErrorKind::GraphIDAlreadyPresent, file!(), line!()));
        }
        self.lattices.insert(lattice.get_id(), lattice);

        Ok(())
    }

    /// Adds a [`SpikeTrainLattice`] to the network if the lattice has an id that is 
    /// not already in the network
    pub fn add_spike_train_lattice(
        &mut self, 
        spike_train_lattice: SpikeTrainLattice<W, X>, 
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&spike_train_lattice.id) {
            return Err(LatticeNetworkError::new(LatticeNetworkErrorKind::GraphIDAlreadyPresent, file!(), line!()));
        }

        self.spike_train_lattices.insert(spike_train_lattice.id, spike_train_lattice);

        Ok(())
    }

    /// Resets the clock and last firing times for the entire network
    pub fn reset_timing(&mut self) {
        self.internal_clock = 0;

        self.lattices.values_mut()
            .for_each(|i| i.reset_timing());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.reset_timing());
    }

    /// Returns an immutable reference to all the lattice hashmaps
    pub fn get_lattices(&self) -> (&HashMap<usize, Lattice<T, U, V>>, &HashMap<usize, SpikeTrainLattice<W, X>>) {
        (&self.lattices, &self.spike_train_lattices)
    }

    /// Returns the set of [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values(&self) -> Values<usize, Lattice<T, U, V>> {
        self.lattices.values()
    }

    /// Returns a mutable set [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values_mut(&mut self) -> ValuesMut<usize, Lattice<T, U, V>> {
        self.lattices.values_mut()
    }

    /// Returns a reference to [`Lattice`] given the identifier
    pub fn get_lattice(&self, id: &usize) -> Option<&Lattice<T, U, V>> {
        self.lattices.get(id)
    }

    /// Returns a mutable reference to a [`Lattice`] given the identifier
    pub fn get_mut_lattice(&mut self, id: &usize) -> Option<&mut Lattice<T, U, V>> {
        self.lattices.get_mut(id)
    }

    /// Returns a reference to [`SpikeTrainLattice`] given the identifier
    pub fn get_spike_train_lattice(&self, id: &usize) -> Option<&SpikeTrainLattice<W, X>> {
        self.spike_train_lattices.get(id)
    }

    /// Returns a mutable reference to a [`SpikeTrainLattice`] given the identifier
    pub fn get_mut_spike_train_lattice(&mut self, id: &usize) -> Option<&mut SpikeTrainLattice<W, X>> {
        self.spike_train_lattices.get_mut(id)
    }

    /// Returns the set of [`SpikeTrainLattice`]s in the hashmap of spike train lattices
    pub fn spike_trains_values(&self) -> Values<usize, SpikeTrainLattice<W, X>> {
        self.spike_train_lattices.values()
    }

    /// Returns a mutable set [`SpikeTrainLattice`]s in the hashmap of spike train lattices    
    pub fn spike_trains_values_mut(&mut self) -> ValuesMut<usize, SpikeTrainLattice<W, X>> {
        self.spike_train_lattices.values_mut()
    }

    /// Returns an immutable reference to the connecting graph
    pub fn get_connecting_graph(&self) -> &U {
        &self.connecting_graph
    }

    /// Returns a hashset of all the ids
    pub fn get_all_ids(&self) -> HashSet<usize> {
        let mut ids = HashSet::new();

        self.lattices.keys()
            .for_each(|i| { ids.insert(*i); });
        self.spike_train_lattices.keys()
            .for_each(|i| { ids.insert(*i); });

        ids
    }

    /// Connects the neurons in lattices together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connectins ([`Lattice`] only)
    pub fn connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: fn((usize, usize), (usize, usize)) -> bool,
        weight_logic: Option<fn((usize, usize), (usize, usize)) -> f64>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::new(LatticeNetworkErrorKind::PostsynapticLatticeCannotBeSpikeTrain, file!(), line!()));
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::new(LatticeNetworkErrorKind::PresynapticIDNotFound, file!(), line!()));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::new(LatticeNetworkErrorKind::PostsynapticIDNotFound, file!(), line!()));
        }

        if self.lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id)
                .unwrap()
                .graph;
            self.lattices.get(&presynaptic_id).unwrap()
                .graph
                .get_every_node()
                .iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        if (connecting_conditional)(i.pos, j.pos) {
                            self.connecting_graph.add_node(*i);
                            self.connecting_graph.add_node(*j);

                            let weight = weight_logic.map_or(1., |logic| (logic)(i.pos, j.pos));
                            self.connecting_graph.edit_weight(i, j, Some(weight)).unwrap();
                        }
                    }
                });
        } else {
            let presynaptic_positions = self.spike_train_lattices.get(&presynaptic_id)
                .unwrap()
                .cell_grid
                .iter()
                .enumerate()
                .flat_map(|(n1, i)| {
                    i.iter()
                        .enumerate()
                        .map(move |(n2, _)| 
                            GraphPosition {
                                id: presynaptic_id, 
                                pos: (n1, n2),
                            }
                        )
                })
                .collect::<Vec<GraphPosition>>();

            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id).unwrap().graph;

            presynaptic_positions.iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        if (connecting_conditional)(i.pos, j.pos) {
                            self.connecting_graph.add_node(*i);
                            self.connecting_graph.add_node(*j);
                            
                            let weight = weight_logic.map_or(1., |logic| (logic)(i.pos, j.pos));
                            self.connecting_graph.edit_weight(i, j, Some(weight)).unwrap();
                        }
                    }
                });
        }

        Ok(())
    }

    fn get_all_input_positions(&self, pos: GraphPosition) -> HashSet<GraphPosition> {
        let mut input_positions = self.lattices[&pos.id].graph.get_incoming_connections(&pos)
            .expect("Cannot find position");

        match self.connecting_graph.get_incoming_connections(&pos) {
            Ok(value) => {
                input_positions.extend(value)
            },
            Err(_) => {}
        };
    
        input_positions
    }

    // fn get_all_output_positions(&self, pos: GraphPosition) -> HashSet<GraphPosition> {
    //     let mut output_positions = self.lattices[&pos.id].graph.get_outgoing_connections(&pos)
    //         .expect("Cannot find position");

    //     match self.connecting_graph.get_outgoing_connections(&pos) {
    //         Ok(value) => {
    //             output_positions.extend(value)
    //         },
    //         Err(_) => {}
    //     }

    //     output_positions
    // }

    fn calculate_electrical_input_from_positions(
        &self, 
        postsynaptic_position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>
    ) -> f64 {
        let postsynaptic_neuron: &T = &self.lattices.get(&postsynaptic_position.id)
            .unwrap()
            .cell_grid[postsynaptic_position.pos.0][postsynaptic_position.pos.1];

        let mut input_val = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;

                let final_input = if self.lattices.contains_key(&input_position.id) {
                    let input_cell = &self.lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    signed_gap_junction(input_cell, postsynaptic_neuron)
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    spike_train_gap_juncton(input_cell, postsynaptic_neuron, self.internal_clock)
                };
                
                let weight: f64 = self.connecting_graph.lookup_weight(&input_position, postsynaptic_position)
                    .unwrap_or(Some(0.))
                    .unwrap();

                final_input * weight
            })
            .sum::<f64>();

        if self.lattices.get(&postsynaptic_position.id).unwrap().gaussian {
            input_val *= postsynaptic_neuron.get_gaussian_factor();
        }

        input_val /= input_positions.len() as f64;

        return input_val;
    }

    fn calculate_neurotransmitter_input_from_positions(
        &self, 
        postsynaptic_position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>
    ) -> NeurotransmitterConcentrations {
        let postsynaptic_neuron: &T = &self.lattices.get(&postsynaptic_position.id)
            .unwrap()
            .cell_grid[postsynaptic_position.pos.0][postsynaptic_position.pos.1];

        let input_vals: Vec<NeurotransmitterConcentrations> = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;

                let mut neurotransmitter_input = if self.lattices.contains_key(&input_position.id) {
                    let input_cell = &self.lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    let final_input = input_cell.get_neurotransmitter_concentrations();

                    final_input
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    let final_input = input_cell.get_neurotransmitter_concentrations();

                    final_input
                };
                
                let weight: f64 = self.connecting_graph.lookup_weight(&input_position, postsynaptic_position)
                    .unwrap_or(Some(0.))
                    .unwrap();

                weight_neurotransmitter_concentration(&mut neurotransmitter_input, weight);

                neurotransmitter_input
            })
            .collect();

        let mut input_val = aggregate_neurotransmitter_concentrations(&input_vals);

        if self.lattices.get(&postsynaptic_position.id).unwrap().gaussian {
            weight_neurotransmitter_concentration(
                &mut input_val, 
                postsynaptic_neuron.get_gaussian_factor()
            );
        }

        weight_neurotransmitter_concentration(
            &mut input_val, 
            (1 / input_positions.len()) as f64
        );

        return input_val;
    }

    fn get_every_node_as_ref(&self) -> HashSet<&GraphPosition> {
        let mut nodes = HashSet::new();

        for i in self.lattices.values() {
            nodes.extend(i.graph.get_every_node_as_ref());
        }

        nodes
    }

    fn get_all_electrical_inputs(&self) -> HashMap<GraphPosition, f64> {
        // eventually paralellize
        // may need to remove the cloning in get every node
        self.get_every_node_as_ref()
            .iter()
            .map(|pos| {
                let input_positions = self.get_all_input_positions(**pos);

                let input = self.calculate_electrical_input_from_positions(
                    &pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    fn get_all_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f64>, HashMap<GraphPosition, Option<NeurotransmitterConcentrations>>) {
        let neurotransmitters_inputs = self.get_every_node_as_ref()
            .iter()
            .map(|pos| {
                let input = match self.lattices.get(&pos.id).unwrap().do_receptor_kinetics {
                    true => {
                        Some(
                            self.calculate_neurotransmitter_input_from_positions(
                                &pos,
                                &self.get_all_input_positions(**pos),
                            )
                        )
                    },
                    false => None,
                };

                (**pos, input)
            })
            .collect();

        let inputs = self.get_all_electrical_inputs();

        (inputs, neurotransmitters_inputs)
    }

    fn update_weights_from_spiking_neuron_across_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let given_neuron = &self.lattices.get(&pos.id).unwrap().cell_grid[x][y];

        for input_pos in self.connecting_graph.get_incoming_connections(&pos).unwrap_or(HashSet::new()) {
            let (x_in, y_in) = input_pos.pos;

            let current_weight: f64 = self.connecting_graph
                .lookup_weight(&input_pos, &pos)
                .unwrap_or(Some(0.))
                .unwrap();

            let dw = update_weight_stdp(
                &self.lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                given_neuron,
            );
                                        
            self.connecting_graph
                .edit_weight(
                    &input_pos, 
                    &pos, 
                    Some(current_weight + dw)
                )?;
        }

        for output_pos in self.connecting_graph.get_outgoing_connections(&pos).unwrap_or(HashSet::new()) {
            let (x_out, y_out) = output_pos.pos;

            let current_weight: f64 = self.connecting_graph
                .lookup_weight(&pos, &output_pos)
                .unwrap_or(Some(0.))
                .unwrap();

            let dw = update_weight_stdp(
                given_neuron,
                &self.lattices.get(&output_pos.id).unwrap().cell_grid[x_out][y_out], 
            );
                                        
            self.connecting_graph
                .edit_weight(
                    &pos, 
                    &output_pos, 
                    Some(current_weight + dw)
                )?;
        }

        Ok(())
    }

    fn update_weights_from_spiking_neurons_within_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = self.lattices.get_mut(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];
        
        for input_pos in current_lattice.graph.get_incoming_connections(&pos).unwrap_or(HashSet::new()) {
            let (x_in, y_in) = input_pos.pos;

            let current_weight: f64 = current_lattice.graph
                .lookup_weight(&input_pos, &pos)
                .unwrap_or(Some(0.))
                .unwrap();

            let dw = update_weight_stdp(
                &current_lattice.cell_grid[x_in][y_in], 
                given_neuron,
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &input_pos, 
                    &pos, 
                    Some(current_weight + dw)
                )?;
        }

        for output_pos in current_lattice.graph.get_outgoing_connections(&pos).unwrap_or(HashSet::new()) {
            let (x_out, y_out) = output_pos.pos;

            let current_weight: f64 = current_lattice.graph
                .lookup_weight(&pos, &output_pos)
                .unwrap_or(Some(0.))
                .unwrap();

            let dw = update_weight_stdp(
                given_neuron,
                &current_lattice.cell_grid[x_out][y_out], 
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &pos, 
                    &output_pos, 
                    Some(current_weight + dw)
                )?;
        }

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of electrical and neurotransmitter inputs
    pub fn iterate_with_neurotransmission(
        &mut self, 
        inputs: &HashMap<GraphPosition, f64>, 
        neurotransmitter_inputs: &HashMap<GraphPosition, Option<NeurotransmitterConcentrations>>,
    ) -> Result<(), GraphError> {
        let mut spiking_positions = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos.pos;
                let input_value = *inputs.get(&pos).unwrap();

                let input_neurotransmitter = neurotransmitter_inputs.get(&pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    input_value, input_neurotransmitter.as_ref(),
                );
    
                if is_spiking {
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                    if lattice.do_stdp {
                        spiking_positions.push((x, y, pos));
                    }
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        for (x, y, pos) in spiking_positions {
            self.update_weights_from_spiking_neuron_across_lattices(x, y, &pos)?;
            self.update_weights_from_spiking_neurons_within_lattices(x, y, &pos)?;
        }

        self.internal_clock += 1;

        for lattice in self.lattices.values_mut() {
            lattice.internal_clock = self.internal_clock;
        }

        self.spike_train_lattices.values_mut()
            .for_each(|i|{
                i.iterate();
            });

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of only electrical inputs
    pub fn iterate(
        &mut self,
        inputs: &HashMap<GraphPosition, f64>,
    ) -> Result<(), GraphError> {
        let mut spiking_positions = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos.pos;
                let input_value = *inputs.get(&pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_and_spike(input_value);
    
                if is_spiking {
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if is_spiking {
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                    if lattice.do_stdp {
                        spiking_positions.push((x, y, pos));
                    }
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        for (x, y, pos) in spiking_positions {
            self.update_weights_from_spiking_neuron_across_lattices(x, y, &pos)?;
            self.update_weights_from_spiking_neurons_within_lattices(x, y, &pos)?;
        }
        
        self.internal_clock += 1;

        for lattice in self.lattices.values_mut() {
            lattice.internal_clock = self.internal_clock;
        }

        self.spike_train_lattices.values_mut()
            .for_each(|i|{
                i.iterate();
            });

        Ok(())
    }

    /// Iterates the lattice based only on internal connections for a given amount of time using
    /// both electrical and neurotransmitter inputs, set `do_receptor_kinetics` to `true` to update
    /// receptor kinetics
    pub fn run_lattices_with_neurotransmission(
        &mut self, 
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {       
            let (inputs, neurotransmitter_inputs) = self.get_all_electrical_and_neurotransmitter_inputs();
    
            self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates lattice based only on internal connections for a given amount of time using
    /// only electrical inputs
    pub fn run_lattices(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {
            let inputs = self.get_all_electrical_inputs();

            self.iterate(&inputs)?;
        }

        Ok(())
    }
}
