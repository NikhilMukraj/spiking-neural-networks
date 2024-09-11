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
    f32::consts::PI, 
    result::Result,
};
use rayon::prelude::*;
pub mod integrate_and_fire;
pub mod ion_channels;
pub mod hodgkin_huxley;
pub mod morris_lecar;
pub mod attractors;
pub mod spike_train;
use spike_train::{DeltaDiracRefractoriness, NeuralRefractoriness, PoissonNeuron, SpikeTrain};
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    aggregate_neurotransmitter_concentrations, weight_neurotransmitter_concentration, 
    ApproximateNeurotransmitter, CurrentVoltage, GapConductance, GaussianParameters, 
    IsSpiking, IterateAndSpike, NeurotransmitterConcentrations, NeurotransmitterType 
};
pub mod plasticity;
use plasticity::{
    Plasticity, STDP, RewardModulator,
    RewardModulatedSTDP, RewardModulatedWeight, TraceRSTDP,
};
/// A set of macros to automatically derive traits necessary for the `IterateAndSpike` trait.
pub mod iterate_and_spike_traits {
    pub use iterate_and_spike_traits::*;
}
#[cfg(feature = "gpu")]
pub mod gpu_lattices;
use crate::error::{AgentError, GraphError, LatticeNetworkError, SpikingNeuralNetworksError};
use crate::graph::{Graph, GraphPosition, AdjacencyMatrix, ToGraphPosition, Position};
use crate::interactable::{Agent, UnsupervisedAgent};


/// Calculates the current between two neurons based on the voltage and
/// the gap conductance of the synapse
pub fn gap_junction<T: CurrentVoltage, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f32 {
    postsynaptic_neuron.get_gap_conductance() * 
    (presynaptic_neuron.get_current_voltage() - postsynaptic_neuron.get_current_voltage())
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
/// - `electrical_synapse` : use `true` to update neurons based on electrical gap junctions
/// 
/// - `chemical_synapse` : use `true` to update receptor gating values of 
///     the neurons based on neurotransmitter input during the simulation
/// 
/// - `gaussian` : use `Some(GaussianParameters)` to add randomly distributed normal noise to the input
///     of the presynaptic neuron
pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    input_current: f32,
    electrical_synapse: bool,
    chemical_synapse: bool,
    gaussian: Option<GaussianParameters>,
) -> (bool, bool) {
    let input_current = match gaussian {
        Some(params) => input_current * params.get_random_number(),
        None => input_current,
    };

    let post_current = if electrical_synapse {
        gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        )
    } else {
        0.
    };

    let t_total = if chemical_synapse {
        presynaptic_neuron.get_neurotransmitter_concentrations()
    } else {
        HashMap::new()
    };

    let pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        &t_total,
    );

    (pre_spiking, post_spiking)
}

/// Calculates the input to the postsynaptic neuron given a spike train
/// and the potenation of the spike train as well as the current timestep 
/// of the simulation
pub fn spike_train_gap_juncton<T: SpikeTrain, U: GapConductance>(
    presynaptic_neuron: &T,
    postsynaptic_neuron: &U,
    timestep: usize,
) -> f32 {
    let (v_max, v_resting) = presynaptic_neuron.get_height();

    if presynaptic_neuron.get_last_firing_time().is_none() {
        return v_resting;
    }

    let last_firing_time = presynaptic_neuron.get_last_firing_time().unwrap();
    let refractoriness_function = presynaptic_neuron.get_refractoriness_function();
    let dt = presynaptic_neuron.get_dt();
    let conductance = postsynaptic_neuron.get_gap_conductance();
    let effect = refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt);

    conductance * effect
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
/// - `electrical_synapse` : use `true` to update neurons based on electrical gap junctions
/// 
/// - `chemical_synapse` : use `true` to update receptor gating values of 
///     the neurons based on neurotransmitter input during the simulation
pub fn iterate_coupled_spiking_neurons_and_spike_train<N, T, U>(
    spike_train: &mut T,
    presynaptic_neuron: &mut U, 
    postsynaptic_neuron: &mut U,
    timestep: usize,
    electrical_synapse: bool,
    chemical_synapse: bool,
) -> (bool, bool, bool) 
where
    T: SpikeTrain<N=N>,
    U: IterateAndSpike<N=N>,
    N: NeurotransmitterType,
{
    let pre_t_total = if chemical_synapse {
        spike_train.get_neurotransmitter_concentrations()
    } else {
        HashMap::new()
    };

    let (pre_current, post_current) = if electrical_synapse {
        let pre_current = spike_train_gap_juncton(
            spike_train, 
            presynaptic_neuron, 
            timestep
        );

        let post_current = gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        (pre_current, post_current)
    } else {
        (0., 0.)
    };

    let post_t_total = if chemical_synapse {
        presynaptic_neuron.get_neurotransmitter_concentrations()
    } else {
        HashMap::new()
    };

    let spike_train_spiking = spike_train.iterate();   
    if spike_train_spiking {
        spike_train.set_last_firing_time(Some(timestep));
    }
    
    let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        pre_current,
        &pre_t_total,
    );
    if pre_spiking {
        presynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        &post_t_total,
    ); 
    if post_spiking {
        postsynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    (spike_train_spiking, pre_spiking, post_spiking)
}

/// Handles history of a lattice
pub trait LatticeHistory: Default + Send + Sync {
    /// Stores the current state of the lattice given the cell grid
    fn update<T: IterateAndSpike>(&mut self, state: &[Vec<T>]);
    /// Resets history
    fn reset(&mut self);
}

/// Stores EEG value history
#[derive(Debug, Clone)]
pub struct EEGHistory {
    /// EEG values
    pub history: Vec<f32>,
    /// Voltage from EEG equipment (mV)
    pub reference_voltage: f32,
    /// Distance from neurons to equipment (mm)
    pub distance: f32,
    /// Conductivity of medium (S/mm)
    pub conductivity: f32,
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

fn get_grid_voltages<T: CurrentVoltage>(grid: &[Vec<T>]) -> Vec<Vec<f32>> {
    grid.iter()
        .map(|i| {
            i.iter()
                .map(|j| j.get_current_voltage())
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}

impl LatticeHistory for EEGHistory {
    fn update<T: IterateAndSpike>(&mut self, state: &[Vec<T>]) {
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
#[derive(Default, Debug, Clone)]
pub struct GridVoltageHistory {
    /// Voltage history
    pub history: Vec<Vec<Vec<f32>>>
}

impl LatticeHistory for GridVoltageHistory {
    fn update<T: IterateAndSpike>(&mut self, state: &[Vec<T>]) {
        self.history.push(get_grid_voltages::<T>(state));
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Stores history as the average voltage over time
#[derive(Default, Debug, Clone)]
pub struct AverageVoltageHistory {
    /// Voltage history
    pub history: Vec<f32>,
}

impl LatticeHistory for AverageVoltageHistory {
    fn update<T: IterateAndSpike>(&mut self, state: &[Vec<T>]) {
        let voltages: Vec<f32> = get_grid_voltages::<T>(state).into_iter().flatten().collect();
        let length = voltages.len() as f32;
        self.history.push(
            voltages.into_iter().sum::<f32>() / length
        );
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Stores history of spikes over time
#[derive(Default, Debug, Clone)]
pub struct SpikeHistory {
    /// Spike history
    pub history: Vec<Vec<Vec<bool>>>,
}

impl SpikeHistory {
    /// Aggregates the spikes into a vector representing the firing rate,
    /// returns an empty vector if history is empty, also assumes grid is not ragged
    #[allow(clippy::needless_range_loop)]
    pub fn aggregate(&self) -> Vec<Vec<isize>> {
        let z_size = self.history.len();
        let y_size = match self.history.first() {
            Some(value) => value.len(),
            None => { return Vec::new() },
        };
        let x_size = match self.history[0].first() {
            Some(value) => value.len(),
            None => { return Vec::new() },
        };
    
        let mut aggregation = vec![vec![0; x_size]; y_size];
    
        for z in 0..z_size {
            for y in 0..y_size {
                for x in 0..x_size {
                    if self.history[z][y][x] {
                        aggregation[y][x] += 1;
                    }
                }
            }
        }
    
        aggregation
    }
}

impl LatticeHistory for SpikeHistory {
    fn update<T: IsSpiking>(&mut self, state: &[Vec<T>]) {
        self.history.push(
            state.iter()
                .map(|i| {
                    i.iter()
                        .map(|j| j.is_spiking())
                        .collect::<Vec<bool>>()
                    })
                .collect::<Vec<Vec<bool>>>()
        );
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Stores history of spikes over time for spike trains
#[derive(Default, Debug, Clone)]
pub struct SpikeTrainSpikeHistory {
    /// Spike history
    pub history: Vec<Vec<Vec<bool>>>,
}

impl SpikeTrainLatticeHistory for SpikeTrainSpikeHistory {
    fn update<T: IsSpiking>(&mut self, state: &[Vec<T>]) {
        self.history.push(
            state.iter()
                .map(|i| {
                    i.iter()
                        .map(|j| j.is_spiking())
                        .collect::<Vec<bool>>()
                    })
                .collect::<Vec<Vec<bool>>>()
        );
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

macro_rules! impl_apply {
    () => {
        /// Applies a function across the entire cell grid to each neuron
        pub fn apply<F>(&mut self, f: F)
        where
            F: Fn(&mut T),
        {
            for row in self.cell_grid.iter_mut() {
                for neuron in row {
                    f(neuron);
                }
            }
        }

        /// Applies a function across the entire cell grid to each neuron
        /// given the position, `(usize, usize)`, of the neuron and the neuron itself
        pub fn apply_given_position<F>(&mut self, f: F)
        where
            F: Fn(Position, &mut T),
        {
            for (i, row) in self.cell_grid.iter_mut().enumerate() {
                for (j, neuron) in row.iter_mut().enumerate() {
                    f((i, j), neuron);
                }
            }
        }
    };
}

/// Electrical inputs for internal calculations
pub type InternalElectricalInputs = HashMap<(usize, usize), f32>;

/// Chemical inputs for internal calculations
pub type InternalChemicalInputs<N> = HashMap<(usize, usize), NeurotransmitterConcentrations<N>>;

/// Lattice of [`IterateAndSpike`] neurons, each lattice has a corresponding [`Graph`] that
/// details the internal connections of the lattice, a grid of neurons stored as a 2
/// dimensional array, as well as a field to track the history of the lattice over time, 
/// by default history is not updated, use `run_lattice` to run the electrical synapses 
/// of the lattice for the given number of iterations, use `populate` to fill the lattice 
/// with a given neuron and its associated parameters, use `connect` to generate connections 
/// between neurons in the lattice, `connect` and `populate` should be used
/// to generate connections and grid instead of modifying the graph and grid directly,
/// after grid is generated, neuronal parameters and values can be freely edited
/// 
/// Use `connect` and `populate` to generate lattice:
/// ```rust
/// # use rand::Rng;
/// # use spiking_neural_networks::{
/// #     neuron::{
/// #         integrate_and_fire::IzhikevichNeuron,
/// #         Lattice
/// #     },
/// #     error::SpikingNeuralNetworksError,
/// # };
/// #
/// // has an 80% chance of returning true if distance from neuron to neuron is less than 2.,
/// // otherwise false
/// fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
///     (((x.0 as f32 - y.0 as f32).powf(2.) + (x.1 as f32 - y.1 as f32).powf(2.))).sqrt() <= 2. && 
///     rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
///     x != y
/// }
/// 
/// fn main() -> Result<(), SpikingNeuralNetworksError> {
///     // generate base neuron
///     let base_neuron = IzhikevichNeuron {
///        gap_conductance: 10.,
///        ..IzhikevichNeuron::default_impl()
///     };
///     
///     let mut lattice = Lattice::default_impl();
/// 
///     // creates 5x5 grid of neurons
///     lattice.populate(&base_neuron, 5, 5);
///     // connects each neuron depending on whether the neuron is in a radius of 2. with
///     // an 80% chance of connectiing, each neuron is connected with a default weight of 1.
///     lattice.connect(&connection_conditional, None);
/// 
///     // lattice is simulated for 500 iterations
///     lattice.run_lattice(500)?;
/// 
///     // randomly initialize starting values of neurons
///     lattice.apply(|neuron: &mut _| {
///         let mut rng = rand::thread_rng();
///         neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
///     });
/// 
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Lattice<
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: Plasticity<T, T, f32>,
    N: NeurotransmitterType,
> {
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
    /// Whether to use electrical synapses
    pub electrical_synapse: bool,
    /// Whether to use chemical synapses
    pub chemical_synapse: bool,
    /// Plasticity rule
    pub plasticity: W,
    /// Whether to update weights with based on plasticity when iterating
    pub do_plasticity: bool,
    /// Whether to calculate inputs in parallel
    pub parallel: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<N: NeurotransmitterType, T: IterateAndSpike<N=N>, U: Graph<K=(usize, usize), V=f32>, V: LatticeHistory, W: Plasticity<T, T, f32>> Default for Lattice<T, U, V, W, N> {
    fn default() -> Self {
        Lattice {
            cell_grid: vec![],
            graph: U::default(),
            grid_history: V::default(),
            update_graph_history: false,
            update_grid_history: false,
            electrical_synapse: true,
            chemical_synapse: false,
            do_plasticity: false,
            plasticity: W::default(),
            parallel: false,
            internal_clock: 0,
        }
    }
}

impl<N: NeurotransmitterType, T: IterateAndSpike<N=N>> Lattice<T, AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, STDP, N> {
    // Generates a default lattice implementation given a neuron type
    pub fn default_impl() -> Self {
        Lattice::default()
    }
}

impl<N: NeurotransmitterType, T: IterateAndSpike<N=N>, U: Graph<K=(usize, usize), V=f32>, V: LatticeHistory, W: Plasticity<T, T, f32>> Lattice<T, U, V, W, N> {
    impl_reset_timing!();
    impl_apply!();

    /// Gets id of lattice [`Graph`]
    pub fn get_id(&self) -> usize {
        self.graph.get_id()
    }

    /// Sets id of lattice [`Graph`] given an id
    pub fn set_id(&mut self, id: usize) {
        self.graph.set_id(id);
    }

    /// Sets the timestep variable of each neuron and plasticity modulator to `dt`
    pub fn set_dt(&mut self, dt: f32) {
        self.apply(|neuron| neuron.set_dt(dt));
        self.plasticity.set_dt(dt);
    }
    
    /// Sets the graph of the lattice given a new lattice, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_graph(&mut self, new_graph: U) -> Result<(), GraphError> {
        let id = self.get_id();
        for pos in new_graph.get_every_node_as_ref() {
            match self.cell_grid.get(pos.0) {
                Some(row) => match row.get(pos.1) {
                    Some(_) => { continue },
                    None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
                },
                None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
            }
        }
    
        self.graph = new_graph;
        self.set_id(id);
    
        Ok(())
    }

    /// Calculates electrical input value from positions
    fn calculate_internal_electrical_input_from_positions(
        &self,
        position: &(usize, usize),
        input_positions: &HashSet<(usize, usize)>, 
    ) -> f32 {
        let (x, y) = position;
        let postsynaptic_neuron = &self.cell_grid[*x][*y];

        let mut input_val = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position;
                let input_cell = &self.cell_grid[*pos_x][*pos_y];

                let final_input = gap_junction(input_cell, postsynaptic_neuron);
                
                final_input * self.graph.lookup_weight(input_position, position).unwrap().unwrap()
            })
            .sum();

        let averager = match input_positions.len() {
            0 => 1.,
            _ => input_positions.len() as f32,
        };

        input_val /= averager;

        input_val
    }

    /// Calculates neurotransmitter input value from positions
    fn calculate_internal_neurotransmitter_input_from_positions(
        &self,
        position: &(usize, usize),
        input_positions: &HashSet<(usize, usize)>, 
    ) -> NeurotransmitterConcentrations<N> {
        let input_vals = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position;
                let input_cell = &self.cell_grid[*pos_x][*pos_y];

                let mut final_input = input_cell.get_neurotransmitter_concentrations();
                let weight = self.graph.lookup_weight(input_position, position).unwrap().unwrap();
                
                weight_neurotransmitter_concentration(&mut final_input, weight);

                final_input
            })
            .collect::<Vec<NeurotransmitterConcentrations<N>>>();

        

        aggregate_neurotransmitter_concentrations(&input_vals)
    }

    /// Gets all internal electrical inputs 
    fn get_internal_electrical_inputs(&self) -> HashMap<(usize, usize), f32> {
        self.graph.get_every_node_as_ref()
            .iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    /// Gets all internal electrical inputs in parallel
    fn par_get_internal_electrical_inputs(&self) -> HashMap<(usize, usize), f32> {
        self.graph.get_every_node_as_ref()
            .par_iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    /// Gets all internal chemical inputs in parallel
    fn par_get_internal_neurotransmitter_inputs(&self) -> InternalChemicalInputs<N> {
        self.graph.get_every_node_as_ref()
            .par_iter()
            .map(|&pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, neurotransmitter_input)
            })
            .collect()
    }

    /// Gets all internal neurotransmitter inputs 
    fn get_internal_neurotransmitter_inputs(&self) -> InternalChemicalInputs<N> {
        self.graph.get_every_node_as_ref()
            .iter()
            .map(|&pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, neurotransmitter_input)
            })
            .collect()
    }

    /// Gets all internal electrical and neurotransmitter inputs 
    fn get_internal_electrical_and_neurotransmitter_inputs(&self) -> 
    (InternalElectricalInputs, InternalChemicalInputs<N>) {
        let neurotransmitter_inputs = self.get_internal_neurotransmitter_inputs();

        let inputs = self.get_internal_electrical_inputs();

        (inputs, neurotransmitter_inputs)
    }

    /// Gets all internal electrical and neurotransmitter inputs in parallel
    fn par_get_internal_electrical_and_neurotransmitter_inputs(&self) -> 
    (InternalElectricalInputs, InternalChemicalInputs<N>) {
        let neurotransmitter_inputs = self.par_get_internal_neurotransmitter_inputs();

        let inputs = self.par_get_internal_electrical_inputs();

        (inputs, neurotransmitter_inputs)
    }

    /// Updates internal weights based on plasticity trait
    fn update_weights_from_neurons(&mut self, x: usize, y: usize, pos: &(usize, usize)) -> Result<(), GraphError> {
        let given_neuron = &self.cell_grid[x][y];
        
        let input_positions = self.graph.get_incoming_connections(pos)?;

        for i in input_positions {
            let (x_in, y_in) = i;
            let mut current_weight = self.graph.lookup_weight(&i, pos)?.unwrap();
            self.plasticity.update_weight(&mut current_weight, &self.cell_grid[x_in][y_in], given_neuron);
                                        
            self.graph.edit_weight(
                &i, 
                pos, 
                Some(current_weight)
            )?;
        }

        let out_going_connections = self.graph.get_outgoing_connections(pos)?;

        for i in out_going_connections {
            let (x_out, y_out) = i;
            let mut current_weight = self.graph.lookup_weight(pos, &i)?.unwrap();
            self.plasticity.update_weight(&mut current_weight, given_neuron, &self.cell_grid[x_out][y_out]);

            self.graph.edit_weight(
                pos, 
                &i, 
                Some(current_weight)
            )?; 
        }

        Ok(())
    }

    /// Iterates lattice one simulation timestep given a set of electrical and neurotransmitter inputs
    pub fn iterate_with_neurotransmission(
        &mut self, 
        inputs: &HashMap<(usize, usize), f32>, 
        neurotransmitter_inputs: &HashMap<(usize, usize), NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;
            let input_value = *inputs.get(&pos).unwrap();

            let input_neurotransmitter = neurotransmitter_inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                input_value, input_neurotransmitter,
            );

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_plasticity && self.plasticity.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    /// Iterates one simulation timestep given only chemical inputs
    pub fn iterate_chemical_synapses_only(
        &mut self, 
        inputs: &HashMap<(usize, usize), NeurotransmitterConcentrations<N>>
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;

            let input_neurotransmitter = inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                0., input_neurotransmitter,
            );

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_plasticity && self.plasticity.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    /// Iterates lattice one simulation timestep given a set of only electrical inputs
    pub fn iterate(
        &mut self,
        inputs: &HashMap<(usize, usize), f32>,
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;
            let input_value = *inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_and_spike(input_value);

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_plasticity && self.plasticity.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    // pub fn par_iterate(
    //     &mut self,
    //     inputs: &HashMap<(usize, usize), f32>,
    // ) -> Result<(), GraphError> {
    //     // may need to only use par iter on the outer versus the inner
    //     // test par being par iter and only outer being par iter

    //     let current_clock = self.internal_clock;
    //     let do_plasticity = self.do_plasticity;
    //     let plasticity = &self.plasticity;

    //     let positions_to_update: Vec<_> = self.cell_grid.par_iter_mut().enumerate().map(|(x, row)| {
    //             let inner: Vec<_> = row.iter_mut().enumerate().filter_map(move |(y, given_neuron)| {
    //                 let pos = (x, y);
    //                 let input_value = *inputs.get(&pos).unwrap();

    //                 let is_spiking = given_neuron.iterate_and_spike(input_value);

    //                 if is_spiking {
    //                     given_neuron.set_last_firing_time(Some(current_clock));
    //                 }

    //                 if do_plasticity && plasticity.do_update(&given_neuron) {
    //                     Some((x, y, pos))
    //                 } else {
    //                     None
    //                 }
    //             }).collect();

    //             inner
    //         })
    //         .flatten()
    //         .collect();

    //     for (x, y, pos) in positions_to_update {
    //         self.update_weights_from_neurons(x, y, &pos)?;
    //     }

    //     if self.update_graph_history {
    //         self.graph.update_history();
    //     }
    //     if self.update_grid_history {
    //         self.grid_history.update(&self.cell_grid);
    //     }
    //     self.internal_clock += 1;

    //     Ok(())
    // }

    /// Iterates the lattice based only on internal connections for a given amount of time using
    /// both electrical and neurotransmitter inputs
    fn run_lattice_with_electrical_and_chemical_synapses(
        &mut self, 
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {       
            let (inputs, neurotransmitter_inputs) = if self.parallel {
                self.par_get_internal_electrical_and_neurotransmitter_inputs()
            } else {
                self.get_internal_electrical_and_neurotransmitter_inputs()
            };
    
            self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates the lattice based only on internal connections for a given amount of time using
    /// neurotransmitter inputs alone
    fn run_lattice_chemical_synapses_only(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {       
            let neurotransmitter_inputs = if self.parallel {
                self.par_get_internal_neurotransmitter_inputs()
            } else {
                self.get_internal_neurotransmitter_inputs()
            };
    
            self.iterate_chemical_synapses_only(&neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates lattice based only on internal connections for a given amount of time using
    /// only electrical inputs
    fn run_lattice_electrical_synapses_only(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {
            let inputs = if self.parallel {
                self.par_get_internal_electrical_inputs()
            } else {
                self.get_internal_electrical_inputs()
            };

            self.iterate(&inputs)?;
        }

        Ok(())
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag
    pub fn run_lattice(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattice_with_electrical_and_chemical_synapses(iterations),
            (true, false) => self.run_lattice_electrical_synapses_only(iterations),
            (false, true) => self.run_lattice_chemical_synapses_only(iterations),
            (false, false) => Ok(()),
        }
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
                self.graph.add_node((i, j))
            }
        }
    }

    /// Connects the neurons in a lattice together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) {
        self.graph.get_every_node()
            .iter()
            .for_each(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j) {
                        match weight_logic {
                            Some(logic) => {
                                self.graph.edit_weight(i, j, Some((logic)(*i, *j))).unwrap();
                            },
                            None => {
                                self.graph.edit_weight(i, j, Some(1.)).unwrap();
                            }
                        };
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }
            });
    }

    /// Connects the neurons in a lattice together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        let output: Result<Vec<_>, LatticeNetworkError> = self.graph.get_every_node()
            .iter()
            .map(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j)? {
                        match weight_logic {
                            Some(logic) => {
                                self.graph.edit_weight(i, j, Some((logic)(*i, *j)?)).unwrap();
                            },
                            None => {
                                self.graph.edit_weight(i, j, Some(1.)).unwrap();
                            }
                        };
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }

                Ok(())
            })
            .collect();

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }
}

impl<N: NeurotransmitterType, T: IterateAndSpike<N=N>, U: Graph<K=(usize, usize), V=f32>, V: LatticeHistory, W: Plasticity<T, T, f32>> UnsupervisedAgent for Lattice<T, U, V, W, N> {
    fn update(&mut self) -> Result<(), AgentError> {
        match self.run_lattice(1) {
            Ok(_) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(e.to_string())),
        }
    }
}

/// Handles history of a spike train lattice
pub trait SpikeTrainLatticeHistory: Default + Send + Sync {
    /// Stores the current state of the lattice given the cell grid
    fn update<T: SpikeTrain>(&mut self, state: &[Vec<T>]);
    /// Resets history
    fn reset(&mut self);
}

/// Stores history as a grid of voltages
#[derive(Default, Debug, Clone)]
pub struct SpikeTrainGridHistory {
    /// Voltage history
    pub history: Vec<Vec<Vec<f32>>>,
}

impl SpikeTrainLatticeHistory for SpikeTrainGridHistory {
    fn update<T: SpikeTrain>(&mut self, state: &[Vec<T>]) {
        self.history.push(get_grid_voltages::<T>(state));
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Lattice of [`SpikeTrain`] neurons
#[derive(Debug, Clone)]
pub struct SpikeTrainLattice<N: NeurotransmitterType, T: SpikeTrain<N=N>, U: SpikeTrainLatticeHistory> {
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

impl<N: NeurotransmitterType, T: SpikeTrain<N=N>, U: SpikeTrainLatticeHistory> Default for SpikeTrainLattice<N, T, U> {
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

impl<N: NeurotransmitterType, T: SpikeTrain<N=N>> SpikeTrainLattice<N, T, SpikeTrainGridHistory> {
    // Generates a default lattice implementation given a spike train type
    pub fn default_impl() -> Self {
        SpikeTrainLattice::default()
    }
}

impl<N: NeurotransmitterType, T: SpikeTrain<N=N>, U: SpikeTrainLatticeHistory> SpikeTrainLattice<N, T, U> {
    impl_reset_timing!();
    impl_apply!();

    /// Returns the identifier of the lattice
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Sets the identifier of the lattice
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }   

    /// Sets the timestep variable of each spike train to `dt`
    pub fn set_dt(&mut self, dt: f32) {
        self.apply(|neuron| neuron.set_dt(dt));
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


fn check_position<T>(
    cell_grid: &[Vec<T>],
    graph_pos: &GraphPosition,
) -> Result<(), SpikingNeuralNetworksError> {
    if let Some(row) = cell_grid.get(graph_pos.pos.0) {
        if row.get(graph_pos.pos.1).is_none() {
            return Err(SpikingNeuralNetworksError::from(
                GraphError::PositionNotFound(format!("{:#?}", graph_pos)),
            ));
        }
    } else {
        return Err(SpikingNeuralNetworksError::from(
            GraphError::PositionNotFound(format!("{:#?}", graph_pos)),
        ));
    }
    Ok(())
}

/// [`LatticeNetwork`] represents a series of lattices interconnected by a [`Graph`], each lattice
/// is associated to a unique identifier, [`Lattice`]s and [`SpikeTrainLattice`]s cannot have
/// the same identifiers, [`SpikeTrainLattice`]s cannot be postsynaptic because the spike trains
/// cannot take in an input, lattices should be populated before moving them to the network,
/// use the `connect` method instead of directly editing the connecting graph, use `run_lattices` 
/// to run the electrical synapses of the lattice network for the given number of iterations
/// 
/// Use `connect` to generate connections between lattices:
/// ```rust
/// # use spiking_neural_networks::{
/// #     neuron::{
/// #         integrate_and_fire::IzhikevichNeuron,
/// #         spike_train::PoissonNeuron,
/// #         Lattice, SpikeTrainLattice, LatticeNetwork,
/// #     },
/// #     error::SpikingNeuralNetworksError,
/// # };
/// #
/// fn one_to_one(x: (usize, usize), y: (usize, usize)) -> bool {
///     x == y
/// }
/// 
/// fn close_connect(x: (usize, usize), y: (usize, usize)) -> bool {
///     (x.0 as f32 - y.0 as f32).abs() < 2. && (x.1 as f32 - y.1 as f32).abs() <= 2.
/// }
/// 
/// fn weight_function(x: (usize, usize), y: (usize, usize)) -> f32 {
///     (((x.0 as f32 - y.0 as f32).powf(2.) + (x.1 as f32 - y.1 as f32).powf(2.))).sqrt()
/// }
/// 
/// fn main() -> Result<(), SpikingNeuralNetworksError>{
///     // generate base neuron
///     let base_neuron = IzhikevichNeuron {
///        gap_conductance: 10.,
///        ..IzhikevichNeuron::default_impl()
///     };
///     
///     // generate base spike train
///     let mut base_spike_train = PoissonNeuron {
///         chance_of_firing: 0.01,
///         ..PoissonNeuron::default_impl()
///     };
/// 
///     let mut lattice1 = Lattice::default_impl();
///     lattice1.set_id(0);
///     let mut lattice2 = lattice1.clone();
///     lattice2.set_id(1);
/// 
///     lattice1.populate(&base_neuron, 3, 3);
///     lattice2.populate(&base_neuron, 3, 3);
/// 
///     let mut spike_train_lattice = SpikeTrainLattice::default_impl();
///     spike_train_lattice.set_id(2);
///     spike_train_lattice.populate(&base_spike_train, 3, 3);
/// 
///     let lattices = vec![lattice1, lattice2];
///     let spike_train_lattices = vec![spike_train_lattice];
///     
///     let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
///     
///     // connects each corressponding neuron in the presynaptic lattice to a neuron in the
///     // postsynaptic lattice as long as their position is the same, scales the weight
///     // of the connection depending on the distance from each neuron
///     network.connect(0, 1, &one_to_one, Some(&weight_function));
/// 
///     // connects the lattices in the same manner as before but does so in the opposite direction
///     network.connect(1, 0, &one_to_one, Some(&weight_function));
/// 
///     // connections each spike train to a postsynaptic neuron in the postsynaptic lattice if 
///     // the neuron is close enough, sets each weight to 1.
///     network.connect(2, 0, &close_connect, None);
/// 
///     // note that connect will overwrite any pre-existing connections between the given
///     // lattices in the direction specified (presynaptic -> postsynaptic will be overwritten)
/// 
///     // runs network for given amount of iterations
///     network.run_lattices(500)?;
/// 
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LatticeNetwork<
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: SpikeTrain<N=N>, 
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=f32>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    N: NeurotransmitterType,
> {
    /// A hashmap of [`Lattice`]s associated with their respective identifier
    lattices: HashMap<usize, Lattice<T, U, V, Z, N>>,
    /// A hashmap of [`SpikeTrainLattice`]s associated with their respective identifier
    spike_train_lattices: HashMap<usize, SpikeTrainLattice<N, W, X>>,
    /// An array of graphs connecting different lattices together
    connecting_graph: Y,
    /// Whether to use electrical synapses throughout entire network
    pub electrical_synapse: bool,
    /// Whether to use chemical synapses throughout entire network
    pub chemical_synapse: bool,
    /// Whether to update connecting graph history
    pub update_connecting_graph_history: bool,
    /// Whether to use parallel input calculation
    pub parallel: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<T, U, V, W, X, Y, Z, N> Default for LatticeNetwork<T, U, V, W, X, Y, Z, N>
where
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32>,
    V: LatticeHistory,
    W: SpikeTrain<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=f32>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    N: NeurotransmitterType,
{
    fn default() -> Self { 
        LatticeNetwork {
            lattices: HashMap::new(),
            spike_train_lattices: HashMap::new(),
            connecting_graph: Y::default(),
            electrical_synapse: true,
            chemical_synapse: false,
            update_connecting_graph_history: false,
            parallel: false,
            internal_clock: 0,
        }
    }
}

impl<N, V, T> LatticeNetwork<
    T, 
    AdjacencyMatrix<(usize, usize), f32>, 
    V, 
    PoissonNeuron<N, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
    SpikeTrainGridHistory, 
    AdjacencyMatrix<GraphPosition, f32>, 
    STDP,
    N,
>
where
    T: IterateAndSpike<N=N>,
    V: LatticeHistory,
    N: NeurotransmitterType,
{
    // Generates a default lattice network implementation given a neuron and spike train type
    pub fn default_impl() -> Self { 
        LatticeNetwork::default()
    }
}

impl<T, U, V, W, X, Y, Z, N> LatticeNetwork<T, U, V, W, X, Y, Z, N>
where
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32> + ToGraphPosition<GraphPos=Y>,
    V: LatticeHistory,
    W: SpikeTrain<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=f32>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    N: NeurotransmitterType,
{
    /// Generates a [`LatticeNetwork`] given lattices to use within the network, (all lattices
    /// must have unique id fields)
    pub fn generate_network(
        lattices: Vec<Lattice<T, U, V, Z, N>>, 
        spike_train_lattices: Vec<SpikeTrainLattice<N, W, X>>
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
}

impl<T, U, V, W, X, Y, Z, N> LatticeNetwork<T, U, V, W, X, Y, Z, N>
where
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32>,
    V: LatticeHistory,
    W: SpikeTrain<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=f32>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    N: NeurotransmitterType,
{
    /// Sets the timestep variable for each neuron, spike train, and plasticity modulator to `dt`
    pub fn set_dt(&mut self, dt: f32) {
        self.lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
    }

    /// Adds a [`Lattice`] to the network if the lattice has an id that is not already in the network
    pub fn add_lattice(
        &mut self, 
        lattice: Lattice<T, U, V, Z, N>
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&lattice.get_id()) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(lattice.get_id()));
        }
        self.lattices.insert(lattice.get_id(), lattice);

        Ok(())
    }

    /// Adds a [`SpikeTrainLattice`] to the network if the lattice has an id that is 
    /// not already in the network
    pub fn add_spike_train_lattice(
        &mut self, 
        spike_train_lattice: SpikeTrainLattice<N, W, X>, 
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&spike_train_lattice.id) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(spike_train_lattice.id));
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
    
    /// Resets all grid histories in the network
    pub fn reset_grid_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
    }

    /// Resets all graph histories in the network (including connecting graph)
    pub fn reset_graph_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.graph.reset_history());

        self.connecting_graph.reset_history();
    }

    /// Returns the set of [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values(&self) -> Values<usize, Lattice<T, U, V, Z, N>> {
        self.lattices.values()
    }

    /// Returns a mutable set [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values_mut(&mut self) -> ValuesMut<usize, Lattice<T, U, V, Z, N>> {
        self.lattices.values_mut()
    }

    /// Returns a reference to [`Lattice`] given the identifier
    pub fn get_lattice(&self, id: &usize) -> Option<&Lattice<T, U, V, Z, N>> {
        self.lattices.get(id)
    }

    /// Returns a mutable reference to a [`Lattice`] given the identifier
    pub fn get_mut_lattice(&mut self, id: &usize) -> Option<&mut Lattice<T, U, V, Z, N>> {
        self.lattices.get_mut(id)
    }

    /// Returns a reference to [`SpikeTrainLattice`] given the identifier
    pub fn get_spike_train_lattice(&self, id: &usize) -> Option<&SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get(id)
    }

    /// Returns a mutable reference to a [`SpikeTrainLattice`] given the identifier
    pub fn get_mut_spike_train_lattice(&mut self, id: &usize) -> Option<&mut SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get_mut(id)
    }

    /// Returns the set of [`SpikeTrainLattice`]s in the hashmap of spike train lattices
    pub fn spike_trains_values(&self) -> Values<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values()
    }

    /// Returns a mutable set [`SpikeTrainLattice`]s in the hashmap of spike train lattices    
    pub fn spike_trains_values_mut(&mut self) -> ValuesMut<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values_mut()
    }

    /// Returns an immutable reference to the connecting graph
    pub fn get_connecting_graph(&self) -> &Y {
        &self.connecting_graph
    }

    /// Returns a hashset of each [`Lattice`] id
    pub fn get_all_lattice_ids(&self) -> HashSet<usize> {
        self.lattices.keys().cloned().collect()
    }

    /// Returns a hashset of each [`SpikeTrainLattice`] id
    pub fn get_all_spike_train_lattice_ids(&self) -> HashSet<usize> {
        self.spike_train_lattices.keys().cloned().collect()
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

    /// Sets the connecting graph to a new graph, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_connecting_graph(&mut self, new_graph: Y) -> Result<(), SpikingNeuralNetworksError> {
        let id = self.connecting_graph.get_id();

        for graph_pos in new_graph.get_every_node_as_ref() {
            if let Some(lattice) = self.lattices.get(&graph_pos.id) {
                check_position(&lattice.cell_grid, graph_pos)?;
            } else if let Some(spike_train_lattice) = self.spike_train_lattices.get(&graph_pos.id) {
                check_position(&spike_train_lattice.cell_grid, graph_pos)?;
            } else {
                return Err(SpikingNeuralNetworksError::from(
                    LatticeNetworkError::IDNotFoundInLattices(graph_pos.id),
                ));
            }
        }
    
        self.connecting_graph = new_graph;
        self.connecting_graph.set_id(id);
    
        Ok(())
    }

    /// Connects the neurons in lattices together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            self.connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
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
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(*i, *j));
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
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
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(i.pos, *j));
                            self.connecting_graph.edit_weight(i, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }
                });
        }

        Ok(())
    }

    /// Connects the neurons in lattices together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn falliable_connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            self.falliable_connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
        }

        let output: Result<Vec<_>, LatticeNetworkError> = if self.lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id)
                .unwrap()
                .graph;
            self.lattices.get(&presynaptic_id).unwrap()
                .graph
                .get_every_node()
                .iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j)? {
                            let weight = match weight_logic {
                                Some(logic) => logic(*i, *j)?,
                                None => 1.0,
                            };
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
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
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j)? {
                            let weight = match weight_logic {
                                Some(logic) => logic(i.pos, *j)?,
                                None => 1.0,
                            };
                            self.connecting_graph.edit_weight(i, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
        };

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetwork`] together given a 
    /// function to determine if the neurons should be connected given their position (usize, usize), 
    /// and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().connect(connecting_conditional, weight_logic);

        Ok(())
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetwork`] together given a 
    /// function (that can fail) to determine if the neurons should be connected given their position (usize, usize), 
    /// and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().falliable_connect(connecting_conditional, weight_logic)?;

        Ok(())
    }

    fn get_all_input_positions(&self, pos: GraphPosition) -> HashSet<GraphPosition> {
        let mut input_positions: HashSet<GraphPosition> = self.lattices[&pos.id].graph
            .get_incoming_connections(&pos.pos)
            .expect("Cannot find position")
            .iter()
            .map(|i| GraphPosition { id: pos.id, pos: *i})
            .collect();

        if let Ok(value) = self.connecting_graph.get_incoming_connections(&pos) {
            input_positions.extend(value)
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
    ) -> f32 {
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

                    gap_junction(input_cell, postsynaptic_neuron)
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    spike_train_gap_juncton(input_cell, postsynaptic_neuron, self.internal_clock)
                };

                let weight: f32 = if input_position.id != postsynaptic_position.id {
                    self.connecting_graph.lookup_weight(input_position, postsynaptic_position)
                        .unwrap_or(Some(0.))
                        .unwrap()
                } else {
                    self.lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap_or(Some(0.))
                        .unwrap()
                };

                final_input * weight
            })
            .sum::<f32>();

        let averager = match input_positions.len() {
            0 => 1.,
            _ => input_positions.len() as f32,
        };

        input_val /= averager;

        input_val
    }

    fn calculate_neurotransmitter_input_from_positions(
        &self, 
        postsynaptic_position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>
    ) -> NeurotransmitterConcentrations<N> {
        let input_vals: Vec<NeurotransmitterConcentrations<N>> = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;

                let mut neurotransmitter_input = if self.lattices.contains_key(&input_position.id) {
                    let input_cell = &self.lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y]; 

                    input_cell.get_neurotransmitter_concentrations()
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    input_cell.get_neurotransmitter_concentrations()
                };
                
                let weight: f32 = if input_position.id != postsynaptic_position.id {
                    self.connecting_graph.lookup_weight(input_position, postsynaptic_position)
                        .unwrap_or(Some(0.))
                        .unwrap()
                } else {
                    self.lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap_or(Some(0.))
                        .unwrap()
                };

                weight_neurotransmitter_concentration(&mut neurotransmitter_input, weight);

                neurotransmitter_input
            })
            .collect();

        

        aggregate_neurotransmitter_concentrations(&input_vals)
    }

    fn get_every_node(&self) -> HashSet<GraphPosition> {
        let mut nodes = HashSet::new();

        for i in self.lattices.values() {
            let current_nodes: HashSet<GraphPosition> = i.graph.get_every_node_as_ref()
                .iter()
                .map(|j| GraphPosition { id: i.get_id(), pos: **j})
                .collect();
            nodes.extend(current_nodes);
        }

        nodes
    }

    fn get_all_electrical_inputs(&self) -> HashMap<GraphPosition, f32> {
        self.get_every_node()
            .iter()
            .map(|pos| {
                let input_positions = self.get_all_input_positions(*pos);

                let input = self.calculate_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, input)
            })
            .collect()
    }

    fn get_all_neurotransmitter_inputs(&self) -> 
    HashMap<GraphPosition, NeurotransmitterConcentrations<N>> {
        self.get_every_node()
            .iter()
            .map(|pos| {
                let input = self.calculate_neurotransmitter_input_from_positions(
                    pos,
                    &self.get_all_input_positions(*pos),
                );

                (*pos, input)
            })
            .collect()
    }

    fn par_get_all_electrical_inputs(&self) -> HashMap<GraphPosition, f32> {
        self.get_every_node()
            .par_iter()
            .map(|pos| {
                let input_positions = self.get_all_input_positions(*pos);

                let input = self.calculate_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, input)
            })
            .collect()
    }

    fn par_get_all_neurotransmitter_inputs(&self) -> 
    HashMap<GraphPosition, NeurotransmitterConcentrations<N>> {
        self.get_every_node()
            .par_iter()
            .map(|pos| {
                let input = self.calculate_neurotransmitter_input_from_positions(
                    pos,
                    &self.get_all_input_positions(*pos),
                );

                (*pos, input)
            })
            .collect()
    }

    fn get_all_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f32>, HashMap<GraphPosition, NeurotransmitterConcentrations<N>>) {
        let neurotransmitters_inputs = self.get_all_neurotransmitter_inputs();

        let inputs = self.get_all_electrical_inputs();

        (inputs, neurotransmitters_inputs)
    }

    fn par_get_all_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f32>, HashMap<GraphPosition, NeurotransmitterConcentrations<N>>) {
        let neurotransmitters_inputs = self.par_get_all_neurotransmitter_inputs();

        let inputs = self.par_get_all_electrical_inputs();

        (inputs, neurotransmitters_inputs)
    }

    fn update_weights_from_neurons_across_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = &self.lattices.get(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];

        for input_pos in self.connecting_graph.get_incoming_connections(pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos.pos;

            let mut current_weight: f32 = self.connecting_graph
                .lookup_weight(&input_pos, pos)
                .unwrap_or(Some(0.))
                .unwrap();

            if self.lattices.contains_key(&input_pos.id) {
                current_lattice.plasticity.update_weight(
                    &mut current_weight,
                    &self.lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                    given_neuron,
                );
            } else {
                current_lattice.plasticity.update_weight(
                    &mut current_weight,
                    &self.spike_train_lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                    given_neuron,
                );
            }
                                        
            self.connecting_graph
                .edit_weight(
                    &input_pos, 
                    pos, 
                    Some(current_weight)
                )?;
        }

        for output_pos in self.connecting_graph.get_outgoing_connections(pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos.pos;
            let output_lattice = self.lattices.get(&output_pos.id).unwrap();

            let mut current_weight: f32 = self.connecting_graph
                .lookup_weight(pos, &output_pos)
                .unwrap_or(Some(0.))
                .unwrap();

            output_lattice.plasticity.update_weight(
                &mut current_weight,
                given_neuron,
                &output_lattice.cell_grid[x_out][y_out], 
            );
                                        
            self.connecting_graph
                .edit_weight(
                    pos, 
                    &output_pos, 
                    Some(current_weight)
                )?;
        }

        Ok(())
    }

    fn update_weights_from_neurons_within_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = self.lattices.get_mut(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];
        
        for input_pos in current_lattice.graph.get_incoming_connections(&pos.pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos;

            let mut current_weight: f32 = current_lattice.graph
                .lookup_weight(&input_pos, &pos.pos)
                .unwrap_or(Some(0.))
                .unwrap();

            current_lattice.plasticity.update_weight(
                &mut current_weight,
                &current_lattice.cell_grid[x_in][y_in], 
                given_neuron,
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &input_pos, 
                    &pos.pos, 
                    Some(current_weight)
                )?;
        }

        for output_pos in current_lattice.graph.get_outgoing_connections(&pos.pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos;

            let mut current_weight: f32 = current_lattice.graph
                .lookup_weight(&pos.pos, &output_pos)
                .unwrap_or(Some(0.))
                .unwrap();

            current_lattice.plasticity.update_weight(
                &mut current_weight,
                given_neuron,
                &current_lattice.cell_grid[x_out][y_out], 
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &pos.pos, 
                    &output_pos, 
                    Some(current_weight)
                )?;
        }

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of electrical and neurotransmitter inputs
    pub fn iterate_with_neurotransmission(
        &mut self, 
        inputs: &HashMap<GraphPosition, f32>, 
        neurotransmitter_inputs: &HashMap<GraphPosition, NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let input_value = *inputs.get(&graph_pos).unwrap();

                let input_neurotransmitter = neurotransmitter_inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    input_value, input_neurotransmitter,
                );

                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && 
                lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        for (x, y, pos) in positions_to_update {
            self.update_weights_from_neurons_across_lattices(x, y, &pos)?;
            self.update_weights_from_neurons_within_lattices(x, y, &pos)?;
        }

        if self.update_connecting_graph_history {
            self.connecting_graph.update_history();
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

    /// Iterates one simulation timestep lattice given a set of only chemical inputs
    pub fn iterate_chemical_synapses_only(
        &mut self,
        inputs: &HashMap<GraphPosition, NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let input_neurotransmitter = inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    0., input_neurotransmitter,
                );
    
                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        for (x, y, pos) in positions_to_update {
            self.update_weights_from_neurons_across_lattices(x, y, &pos)?;
            self.update_weights_from_neurons_within_lattices(x, y, &pos)?;
        }

        if self.update_connecting_graph_history {
            self.connecting_graph.update_history();
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
        inputs: &HashMap<GraphPosition, f32>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let input_value = *inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_and_spike(input_value);
    
                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && 
                lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        for (x, y, pos) in positions_to_update {
            self.update_weights_from_neurons_across_lattices(x, y, &pos)?;
            self.update_weights_from_neurons_within_lattices(x, y, &pos)?;
        }

        if self.update_connecting_graph_history {
            self.connecting_graph.update_history();
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

    /// Iterates the lattices based only on internal connections for a given amount of time using
    /// both electrical and neurotransmitter inputs
    fn run_lattices_with_electrical_and_chemical_synapses(
        &mut self, 
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {       
            let (inputs, neurotransmitter_inputs) = if self.parallel {
                self.par_get_all_electrical_and_neurotransmitter_inputs()
            } else {
                self.get_all_electrical_and_neurotransmitter_inputs()
            };
    
            self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;        
        }

        Ok(())
    }

    /// Iterates the lattices based only on internal connections for a given amount of time using
    /// neurotransmitter inputs alone
    fn run_lattices_chemical_synapses_only(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {
            let neurotransmitter_inputs = if self.parallel {
                self.par_get_all_neurotransmitter_inputs()
            } else {
                self.get_all_neurotransmitter_inputs()
            };
    
            self.iterate_chemical_synapses_only(&neurotransmitter_inputs)?;      
        }

        Ok(())
    }

    /// Iterates lattices based only on internal connections for a given amount of time using
    /// only electrical inputs
    fn run_lattices_electrical_synapses_only(
        &mut self,
        iterations: usize,
    ) -> Result<(), GraphError> {
        for _ in 0..iterations {
            let inputs = if self.parallel {
                self.par_get_all_electrical_inputs()
            } else {
                self.get_all_electrical_inputs()
            };

            self.iterate(&inputs)?;
        }

        Ok(())
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag
    pub fn run_lattices(&mut self, iterations: usize) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattices_with_electrical_and_chemical_synapses(iterations),
            (true, false) => self.run_lattices_electrical_synapses_only(iterations),
            (false, true) => self.run_lattices_chemical_synapses_only(iterations),
            (false, false) => Ok(()),
        }
    }
}

impl<T, U, V, W, X, Y, Z, N> UnsupervisedAgent for LatticeNetwork<T, U, V, W, X, Y, Z, N>
where
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32>,
    V: LatticeHistory,
    W: SpikeTrain<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=f32>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    N: NeurotransmitterType,
{
    fn update(&mut self) -> Result<(), AgentError> {
        match self.run_lattices(1) {
            Ok(_) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(e.to_string())),
        }
    }
}

// test r-stdp with new equations

// stdp with trace trait for this (TraceSTDP) (instead of calculating dw, it could modify weight directly)
// since state is updated each iteration, half of trace decay is calculated + current dw
// and second time half of trace decay is calculated + current dw
// weight = reward_modulator(trace)
// do_update is always considered to be true

// rolling synaptic updates will always trigger twice if update is always on, 
// add dw to field in trace and increment counter counter, if counter is 1, 
// sum dw and then calculate decay and reset counter and apply change to 
// weight accounting for trace, if that doesnt work try trace calculation times 0.5 on each update 

// for performance sake, update could be turned off if dopamine is 0
// plasticity trait could return whether to loop over presynaptic weights or 
// postsynaptic weights or both when calculating weights

// reward modulated lattice network has connecting graph with enum for weights
// enum { Weight(f32), RewardModulatedWeight(S) }
// connecting function could generate different enums

/// A lattice of neurons whose connections can be modulated by a reward signal
#[derive(Debug, Clone)]
pub struct RewardModulatedLattice<
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=S>, 
    V: LatticeHistory, 
    W: RewardModulator<T, T, S>, 
    N: NeurotransmitterType,
> {
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
    /// Whether to use electrical synapses
    pub electrical_synapse: bool,
    /// Whether to use chemical synapses
    pub chemical_synapse: bool,
    /// Whether to modulate lattice based on reward
    pub do_modulation: bool,
    /// Reward modulator for plasticity rule
    pub reward_modulator: W,
    /// Whether to calculate inputs in parallel
    pub parallel: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<S, T, U, V, W, N> Default for RewardModulatedLattice<S, T, U, V, W, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>,
    U: Graph<K = (usize, usize), V = S>,
    V: LatticeHistory,
    W: RewardModulator<T, T, S>,
    N: NeurotransmitterType,
{
    fn default() -> Self {
        RewardModulatedLattice { 
            cell_grid: vec![], 
            graph: U::default(), 
            grid_history: V::default(), 
            update_graph_history: false, 
            update_grid_history: false,
            electrical_synapse: true,
            chemical_synapse: false, 
            do_modulation: true, 
            reward_modulator: W::default(), 
            parallel: false,
            internal_clock: 0,
        }
    }
}

impl<N: NeurotransmitterType, T: IterateAndSpike<N=N>> RewardModulatedLattice<TraceRSTDP, T, AdjacencyMatrix<(usize, usize), TraceRSTDP>, GridVoltageHistory, RewardModulatedSTDP, N> {
    // Generates a default reward modulated lattice implementation given a neuron and spike train type
    pub fn default_impl() -> Self {
        RewardModulatedLattice::default()
    }
}

impl<S, T, U, V, W, N> RewardModulatedLattice<S, T, U, V, W, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=S>,
    V: LatticeHistory,
    W: RewardModulator<T, T, S>,
    N: NeurotransmitterType,
{
    impl_reset_timing!();
    impl_apply!();

    /// Gets id of lattice [`Graph`]
    pub fn get_id(&self) -> usize {
        self.graph.get_id()
    }

    /// Sets id of lattice [`Graph`] given an id
    pub fn set_id(&mut self, id: usize) {
        self.graph.set_id(id);
    }

    /// Sets the graph of the lattice given a new lattice, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_graph(&mut self, new_graph: U) -> Result<(), GraphError> {
        let id = self.get_id();
        for pos in new_graph.get_every_node_as_ref() {
            match self.cell_grid.get(pos.0) {
                Some(row) => match row.get(pos.1) {
                    Some(_) => { continue },
                    None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
                },
                None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
            }
        }
    
        self.graph = new_graph;
        self.set_id(id);
    
        Ok(())
    }

    /// Sets the timestep variable of each neuron and reward modulator to dt
    pub fn set_dt(&mut self, dt: f32) {
        self.apply(|neuron| neuron.set_dt(dt));
        self.reward_modulator.set_dt(dt);
    }

    /// Calculates electrical input value from positions
    fn calculate_internal_electrical_input_from_positions(
        &self,
        position: &(usize, usize),
        input_positions: &HashSet<(usize, usize)>, 
    ) -> f32 {
        let (x, y) = position;
        let postsynaptic_neuron = &self.cell_grid[*x][*y];

        let mut input_val = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position;
                let input_cell = &self.cell_grid[*pos_x][*pos_y];

                let final_input = gap_junction(input_cell, postsynaptic_neuron);
                
                final_input * self.graph.lookup_weight(input_position, position)
                    .unwrap().unwrap().get_weight()
            })
            .sum();

        let averager = match input_positions.len() {
            0 => 1.,
            _ => input_positions.len() as f32,
        };

        input_val /= averager;

        input_val
    }

    /// Calculates neurotransmitter input value from positions
    fn calculate_internal_neurotransmitter_input_from_positions(
        &self,
        position: &(usize, usize),
        input_positions: &HashSet<(usize, usize)>, 
    ) -> NeurotransmitterConcentrations<N> {
        let input_vals = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position;
                let input_cell = &self.cell_grid[*pos_x][*pos_y];

                let mut final_input = input_cell.get_neurotransmitter_concentrations();
                let trace = self.graph.lookup_weight(input_position, position).unwrap().unwrap();
                
                weight_neurotransmitter_concentration(&mut final_input, trace.get_weight());

                final_input
            })
            .collect::<Vec<NeurotransmitterConcentrations<N>>>();

        

        aggregate_neurotransmitter_concentrations(&input_vals)
    }

    /// Gets all internal electrical inputs 
    fn get_internal_electrical_inputs(&self) -> HashMap<(usize, usize), f32> {
        self.graph.get_every_node_as_ref()
            .iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    /// Gets all internal neurotransmitter inputs 
    fn get_internal_neurotransmitter_inputs(&self) -> 
    HashMap<(usize, usize), NeurotransmitterConcentrations<N>> {
        self.graph.get_every_node_as_ref()
            .iter()
            .map(|&pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, neurotransmitter_input)
            })
            .collect()
    }

    /// Gets all internal electrical and neurotransmitter inputs 
    fn get_internal_electrical_and_neurotransmitter_inputs(&self) -> 
    (InternalElectricalInputs, HashMap<(usize, usize), NeurotransmitterConcentrations<N>>) {
        let neurotransmitter_inputs = self.get_internal_neurotransmitter_inputs();

        let inputs = self.get_internal_electrical_inputs();

        (inputs, neurotransmitter_inputs)
    }

    /// Gets all internal electrical inputs in parallel
    fn par_get_internal_electrical_inputs(&self) -> HashMap<(usize, usize), f32> {
        self.graph.get_every_node_as_ref()
            .par_iter()
            .map(|pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let input = self.calculate_internal_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (**pos, input)
            })
            .collect()
    }

    /// Gets all internal neurotransmitter inputs in parallel
    fn par_get_internal_neurotransmitter_inputs(&self) -> 
    HashMap<(usize, usize), NeurotransmitterConcentrations<N>> {
        self.graph.get_every_node_as_ref()
            .par_iter()
            .map(|&pos| {
                let input_positions = self.graph.get_incoming_connections(pos)
                    .expect("Cannot find position");

                let neurotransmitter_input = self.calculate_internal_neurotransmitter_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, neurotransmitter_input)
            })
            .collect()
    }

    /// Gets all internal electrical and neurotransmitter inputs in parallel
    fn par_get_internal_electrical_and_neurotransmitter_inputs(&self) -> (InternalElectricalInputs, InternalChemicalInputs<N>) {
        let neurotransmitter_inputs = self.par_get_internal_neurotransmitter_inputs();

        let inputs = self.par_get_internal_electrical_inputs();

        (inputs, neurotransmitter_inputs)
    }
    
    /// Updates internal weights based on plasticity
    fn update_weights_from_neurons(&mut self, x: usize, y: usize, pos: &(usize, usize)) -> Result<(), GraphError> {
        let given_neuron = &self.cell_grid[x][y];
        
        let input_positions = self.graph.get_incoming_connections(pos)?;

        for i in input_positions {
            let (x_in, y_in) = i;
            let mut current_weight = self.graph.lookup_weight(&i, pos)?.unwrap();
            self.reward_modulator.update_weight(&mut current_weight, &self.cell_grid[x_in][y_in], given_neuron);
                                        
            self.graph.edit_weight(
                &i, 
                pos, 
                Some(current_weight)
            )?;
        }

        let out_going_connections = self.graph.get_outgoing_connections(pos)?;

        for i in out_going_connections {
            let (x_out, y_out) = i;
            let mut current_weight = self.graph.lookup_weight(pos, &i)?.unwrap();
            self.reward_modulator.update_weight(&mut current_weight, given_neuron, &self.cell_grid[x_out][y_out]);

            self.graph.edit_weight(
                pos, 
                &i, 
                Some(current_weight)
            )?; 
        }

        Ok(())
    }

    /// Iterates lattice one simulation timestep given a set of electrical and neurotransmitter inputs
    pub fn iterate_with_neurotransmission(
        &mut self, 
        inputs: &HashMap<(usize, usize), f32>, 
        neurotransmitter_inputs: &HashMap<(usize, usize), NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;
            let input_value = *inputs.get(&pos).unwrap();

            let input_neurotransmitter = neurotransmitter_inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                input_value, input_neurotransmitter,
            );

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_modulation && self.reward_modulator.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    /// Iterates lattice one simulation timestep given a set of neurotransmitter inputs
    pub fn iterate_with_chemical_synapses_only(
        &mut self, 
        neurotransmitter_inputs: &HashMap<(usize, usize), NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;

            let input_neurotransmitter = neurotransmitter_inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                0., input_neurotransmitter,
            );

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_modulation && self.reward_modulator.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    /// Iterates lattice one simulation timestep given a set of electrical inputs
    pub fn iterate(
        &mut self, 
        inputs: &HashMap<(usize, usize), f32>, 
    ) -> Result<(), GraphError> {
        for pos in self.graph.get_every_node() {
            let (x, y) = pos;

            let input_value = *inputs.get(&pos).unwrap();

            let is_spiking = self.cell_grid[x][y].iterate_and_spike(input_value);

            if is_spiking {
                self.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
            }

            if self.do_modulation && self.reward_modulator.do_update(&self.cell_grid[x][y]) {
                self.update_weights_from_neurons(x, y, &pos)?;
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

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical synapses only
    fn run_lattice_electrical_synapses_only(&mut self, reward: f32) -> Result<(), GraphError> {
        let inputs = if self.parallel {
            self.par_get_internal_electrical_inputs()
        } else {
            self.get_internal_electrical_inputs()
        };

        self.reward_modulator.update(reward);

        self.iterate(&inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical synapses only reward applying reward
    fn run_lattice_electrical_synapses_only_without_reward(&mut self) -> Result<(), GraphError> {
        let inputs = if self.parallel {
            self.par_get_internal_electrical_inputs()
        } else {
            self.get_internal_electrical_inputs()
        };

        self.iterate(&inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// chemical synapses only
    fn run_lattice_chemical_synapses_only(&mut self, reward: f32) -> Result<(), GraphError> {
        let neurotransmitter_inputs = if self.parallel { 
            self.par_get_internal_neurotransmitter_inputs()
        } else {
            self.get_internal_neurotransmitter_inputs()
        };

        self.reward_modulator.update(reward);

        self.iterate_with_chemical_synapses_only(&neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// chemical synapses only without applying reward
    fn run_lattice_chemical_synapses_only_without_reward(&mut self) -> Result<(), GraphError> {
        let neurotransmitter_inputs = if self.parallel { 
            self.par_get_internal_neurotransmitter_inputs()
        } else {
            self.get_internal_neurotransmitter_inputs()
        };

        self.iterate_with_chemical_synapses_only(&neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical and chemical synapses
    fn run_lattice_with_electrical_and_chemical_synapses(&mut self, reward: f32) -> Result<(), GraphError> {
        let (inputs, neurotransmitter_inputs) = if self.parallel {
            self.par_get_internal_electrical_and_neurotransmitter_inputs()
        } else {
            self.get_internal_electrical_and_neurotransmitter_inputs()
        };

        self.reward_modulator.update(reward);

        self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical and chemical synapses without a reward signal
    fn run_lattice_with_electrical_and_chemical_synapses_without_reward(&mut self) -> Result<(), GraphError> {
        let (inputs, neurotransmitter_inputs) = if self.parallel {
            self.par_get_internal_electrical_and_neurotransmitter_inputs()
        } else {
            self.get_internal_electrical_and_neurotransmitter_inputs()
        };

        self.iterate_with_neurotransmission(&inputs, &neurotransmitter_inputs)?;

        Ok(())
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag
    pub fn run_lattice(&mut self, reward: f32) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattice_with_electrical_and_chemical_synapses(reward),
            (true, false) => self.run_lattice_electrical_synapses_only(reward),
            (false, true) => self.run_lattice_chemical_synapses_only(reward),
            (false, false) => Ok(()),
        }
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag without a reward signal
    pub fn run_lattice_without_reward(&mut self) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattice_with_electrical_and_chemical_synapses_without_reward(),
            (true, false) => self.run_lattice_electrical_synapses_only_without_reward(),
            (false, true) => self.run_lattice_chemical_synapses_only_without_reward(),
            (false, false) => Ok(()),
        }
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
                self.graph.add_node((i, j))
            }
        }
    }

    /// Connects the neurons in a lattice together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be
    /// assumes lattice is already populated using the `populate` method
    pub fn connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: &dyn Fn(Position, Position) -> S,
    ) {
        self.graph.get_every_node()
            .iter()
            .for_each(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j) {
                        self.graph.edit_weight(i, j, Some((weight_logic)(*i, *j))).unwrap();
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }
            });
    }

    /// Connects the neurons in a lattice together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: &dyn Fn(Position, Position) -> Result<S, LatticeNetworkError>,
    ) -> Result<(), LatticeNetworkError> {
        let output: Result<Vec<_>, LatticeNetworkError> = self.graph.get_every_node()
            .iter()
            .map(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j)? {
                        self.graph.edit_weight(i, j, Some((weight_logic)(*i, *j)?)).unwrap();
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }

                Ok(())
            })
            .collect();

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
} 

impl<S, T, U, V, W, N> Agent for RewardModulatedLattice<S, T, U, V, W, N> 
where 
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=S>,
    V: LatticeHistory,
    W: RewardModulator<T, T, S>,
    N: NeurotransmitterType,
{
    fn update_and_apply_reward(&mut self, reward: f32) -> Result<(), AgentError> {
        match self.run_lattice(reward) {
            Ok(()) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(format!("Agent error: {}", e))),
        }
    }

    fn update(&mut self) -> Result<(), AgentError> {
        match self.run_lattice_without_reward() {
            Ok(()) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(format!("Agent error: {}", e))),
        }
    }
}

/// A connection that has either a non reward modulated weight a reward modulated weight
#[derive(Debug, Clone, Copy)]
pub enum RewardModulatedConnection<T: RewardModulatedWeight> {
    Weight(f32),
    RewardModulatedWeight(T),
}

impl<T: RewardModulatedWeight> RewardModulatedConnection<T> {
    /// Returns the weight value
    pub fn get_weight(&self) -> f32 {
        match &self {
            RewardModulatedConnection::Weight(value) => *value,
            RewardModulatedConnection::RewardModulatedWeight(weight) => weight.get_weight()
        }
    }
}

#[derive(Debug, Clone)]
enum GraphWrapper<'a, U: Graph<K = (usize, usize)>, V: Graph<K = (usize, usize)>> {
    Graph1(&'a U),
    Graph2(&'a V),
}

impl<'a, U, V> GraphWrapper<'a, U, V>
where
    U: Graph<K = (usize, usize)>,
    V: Graph<K = (usize, usize)>,
{
    fn get_every_node(&self) -> HashSet<(usize, usize)> {
        match self {
            GraphWrapper::Graph1(graph) => graph.get_every_node(),
            GraphWrapper::Graph2(graph) => graph.get_every_node(),
        }
    }
}

/// A network of lattices and reward modulated lattices that can be updated based on a reward
#[derive(Debug, Clone)]
pub struct RewardModulatedLatticeNetwork<
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: SpikeTrain<N=N>, 
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=RewardModulatedConnection<S>>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    R: RewardModulator<T, T, S> + RewardModulator<W, T, S>,
    C: Graph<K=(usize, usize), V=S>,
    N: NeurotransmitterType,
> {
    /// A hashmap of [`Lattice`]s associated with their respective identifier
    lattices: HashMap<usize, Lattice<T, U, V, Z, N>>,
    /// A hasmap of [`RewardModulatedLattice`]s associated with their respective identifier
    reward_modulated_lattices: HashMap<usize, RewardModulatedLattice<S, T, C, V, R, N>>,
    /// A hashmap of [`SpikeTrainLattice`]s associated with their respective identifier
    spike_train_lattices: HashMap<usize, SpikeTrainLattice<N, W, X>>,
    /// An array of graphs connecting different lattices together
    connecting_graph: Y,
    /// Whether to use electrical synapses throughout entire network
    pub electrical_synapse: bool,
    /// Whether to use chemical synapses throughout entire network
    pub chemical_synapse: bool,
    /// Whether to update connecting graph history
    pub update_connecting_graph_history: bool,
    /// Whether to calculate inputs in parallel
    pub parallel: bool,
    /// Internal clock keeping track of what timestep the lattice is at
    pub internal_clock: usize,
}

impl<S, T, U, V, W, X, Y, Z, R, C, N> Default for RewardModulatedLatticeNetwork<S, T, U, V, W, X, Y, Z, R, C, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: SpikeTrain<N=N>, 
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=RewardModulatedConnection<S>>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    R: RewardModulator<T, T, S> + RewardModulator<W, T, S>,
    C: Graph<K=(usize, usize), V=S>,
    N: NeurotransmitterType,
{
    fn default() -> Self { 
        RewardModulatedLatticeNetwork {
            lattices: HashMap::new(),
            reward_modulated_lattices: HashMap::new(),
            spike_train_lattices: HashMap::new(),
            electrical_synapse: true,
            chemical_synapse: false,
            connecting_graph: Y::default(),
            update_connecting_graph_history: false,
            parallel: false,
            internal_clock: 0,
        }
    }
}

impl<T, W, N> RewardModulatedLatticeNetwork<
    TraceRSTDP,
    T,
    AdjacencyMatrix<(usize, usize), f32>,
    GridVoltageHistory,
    W,
    SpikeTrainGridHistory,
    AdjacencyMatrix<GraphPosition, RewardModulatedConnection<TraceRSTDP>>,
    STDP,
    RewardModulatedSTDP,
    AdjacencyMatrix<(usize, usize), TraceRSTDP>,
    N,
>
where
    T: IterateAndSpike<N=N>,
    W: SpikeTrain<N=N>,
    N: NeurotransmitterType,
{
    // Generates a default reward modulated lattice network implementation given a neuron type
    // spike train that uses reward modulated spike time dependent plasticity
    pub fn default_impl() -> Self {
        RewardModulatedLatticeNetwork::default()
    }
}

impl<S, T, U, V, W, Y, X, Z, R, C, N> RewardModulatedLatticeNetwork<S, T, U, V, W, X, Y, Z, R, C, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>,
    U: Graph<K=(usize, usize), V=f32>,
    V: LatticeHistory,
    W: SpikeTrain<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=RewardModulatedConnection<S>>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    R: RewardModulator<T, T, S> + RewardModulator<W, T, S>,
    C: Graph<K=(usize, usize), V=S>,
    N: NeurotransmitterType,
{
    /// Generates a [`RewardModulatedLatticeNetwork`] given lattices to use within the network, 
    /// (all lattices must have unique id fields)
    pub fn generate_network(
        lattices: Vec<Lattice<T, U, V, Z, N>>, 
        reward_modulated_lattices: Vec<RewardModulatedLattice<S, T, C, V, R, N>>,
        spike_train_lattices: Vec<SpikeTrainLattice<N, W, X>>,
    ) -> Result<Self, LatticeNetworkError> {
        let mut network = RewardModulatedLatticeNetwork::default();

        for lattice in lattices {
            network.add_lattice(lattice)?;
        }

        for reward_modulated_lattice in reward_modulated_lattices {
            network.add_reward_modulated_lattice(reward_modulated_lattice)?;
        }

        for spike_train_lattice in spike_train_lattices {
            network.add_spike_train_lattice(spike_train_lattice)?;
        }

        Ok(network)
    }
}

impl<S, T, U, V, W, X, Y, Z, R, C, N> RewardModulatedLatticeNetwork<S, T, U, V, W, X, Y, Z, R, C, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: SpikeTrain<N=N>, 
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=RewardModulatedConnection<S>>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    R: RewardModulator<T, T, S> + RewardModulator<W, T, S>,
    C: Graph<K=(usize, usize), V=S>,
    N: NeurotransmitterType,
{
    /// Adds a [`Lattice`] to the network if the lattice has an id that is not already in the network
    pub fn add_lattice(
        &mut self, 
        lattice: Lattice<T, U, V, Z, N>
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&lattice.get_id()) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(lattice.get_id()));
        }
        self.lattices.insert(lattice.get_id(), lattice);

        Ok(())
    }

    /// Adds a [`RewardModulatedLattice`] to the network if the lattice has an id that is 
    /// not already in the network
    pub fn add_reward_modulated_lattice(
        &mut self, 
        reward_modulated_lattice: RewardModulatedLattice<S, T, C, V, R, N>
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&reward_modulated_lattice.get_id()) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(reward_modulated_lattice.get_id()));
        }
        self.reward_modulated_lattices.insert(
            reward_modulated_lattice.get_id(), reward_modulated_lattice
        );

        Ok(())
    }

    /// Adds a [`SpikeTrainLattice`] to the network if the lattice has an id that is 
    /// not already in the network
    pub fn add_spike_train_lattice(
        &mut self, 
        spike_train_lattice: SpikeTrainLattice<N, W, X>, 
    ) -> Result<(), LatticeNetworkError> {
        if self.get_all_ids().contains(&spike_train_lattice.id) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(spike_train_lattice.id));
        }

        self.spike_train_lattices.insert(spike_train_lattice.id, spike_train_lattice);

        Ok(())
    }

    /// Sets the timestep variable for each neuron, spike train, and plasticity modulator, 
    /// and reward modulator to `dt`
    pub fn set_dt(&mut self, dt: f32) {
        self.lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
        self.reward_modulated_lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
    }

    /// Resets the clock and last firing times for the entire network
    pub fn reset_timing(&mut self) {
        self.internal_clock = 0;

        self.lattices.values_mut()
            .for_each(|i| i.reset_timing());
        self.reward_modulated_lattices.values_mut()
            .for_each(|i| i.reset_timing());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.reset_timing());
    }

    /// Resets all grid histories in the network
    pub fn reset_grid_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
        self.reward_modulated_lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
    }

    /// Resets all graph histories in the network (including connecting graph)
    pub fn reset_graph_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.graph.reset_history());
        self.reward_modulated_lattices.values_mut()
            .for_each(|i| i.graph.reset_history());

        self.connecting_graph.reset_history();
    }

    /// Returns the set of [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values(&self) -> Values<usize, Lattice<T, U, V, Z, N>> {
        self.lattices.values()
    }

    /// Returns a mutable set [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values_mut(&mut self) -> ValuesMut<usize, Lattice<T, U, V, Z, N>> {
        self.lattices.values_mut()
    }

    /// Returns a reference to [`Lattice`] given the identifier
    pub fn get_lattice(&self, id: &usize) -> Option<&Lattice<T, U, V, Z, N>> {
        self.lattices.get(id)
    }

    /// Returns a mutable reference to a [`Lattice`] given the identifier
    pub fn get_mut_lattice(&mut self, id: &usize) -> Option<&mut Lattice<T, U, V, Z, N>> {
        self.lattices.get_mut(id)
    }

    /// Returns a immutable set [`RewardModulatedLattice`]s in the hashmap of lattices
    pub fn reward_modulated_lattices_values(&self) -> Values<usize, RewardModulatedLattice<S, T, C, V, R, N>> {
        self.reward_modulated_lattices.values()
    }

    /// Returns a mutable set [`RewardModulatedLattice`]s in the hashmap of lattices
    pub fn reward_modulated_lattices_values_mut(&mut self) -> ValuesMut<usize, RewardModulatedLattice<S, T, C, V, R, N>> {
        self.reward_modulated_lattices.values_mut()
    }

    /// Returns a reference to [`RewardModulatedLattice`] given the identifier
    pub fn get_reward_modulated_lattice(&self, id: &usize) -> Option<&RewardModulatedLattice<S, T, C, V, R, N>> {
        self.reward_modulated_lattices.get(id)
    }

    /// Returns a mutable reference to a [`RewardModulatedLattice`] given the identifier
    pub fn get_mut_reward_modulated_lattice(&mut self, id: &usize) -> Option<&mut RewardModulatedLattice<S, T, C, V, R, N>> {
        self.reward_modulated_lattices.get_mut(id)
    }

    /// Returns a reference to [`SpikeTrainLattice`] given the identifier
    pub fn get_spike_train_lattice(&self, id: &usize) -> Option<&SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get(id)
    }

    /// Returns a mutable reference to a [`SpikeTrainLattice`] given the identifier
    pub fn get_mut_spike_train_lattice(&mut self, id: &usize) -> Option<&mut SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get_mut(id)
    }

    /// Returns the set of [`SpikeTrainLattice`]s in the hashmap of spike train lattices
    pub fn spike_trains_values(&self) -> Values<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values()
    }

    /// Returns a mutable set [`SpikeTrainLattice`]s in the hashmap of spike train lattices    
    pub fn spike_trains_values_mut(&mut self) -> ValuesMut<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values_mut()
    }

    /// Returns an immutable reference to the connecting graph
    pub fn get_connecting_graph(&self) -> &Y {
        &self.connecting_graph
    }

    /// Sets the connecting graph to a new graph, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_connecting_graph(&mut self, new_graph: Y) -> Result<(), SpikingNeuralNetworksError> {
        let id = self.connecting_graph.get_id();

        for graph_pos in new_graph.get_every_node_as_ref() {
            if let Some(lattice) = self.lattices.get(&graph_pos.id) {
                check_position(&lattice.cell_grid, graph_pos)?;
            } else if let Some(spike_train_lattice) = self.spike_train_lattices.get(&graph_pos.id) {
                check_position(&spike_train_lattice.cell_grid, graph_pos)?;
            } else if let Some(reward_modulated_lattice) = self.reward_modulated_lattices.get(&graph_pos.id) {
                check_position(&reward_modulated_lattice.cell_grid, graph_pos)?;
            } else {
                return Err(SpikingNeuralNetworksError::from(
                    LatticeNetworkError::IDNotFoundInLattices(graph_pos.id),
                ));
            }
        }
    
        self.connecting_graph = new_graph;
        self.connecting_graph.set_id(id);
    
        Ok(())
    }

    /// Returns a hashset of each [`Lattice`] id
    pub fn get_all_lattice_ids(&self) -> HashSet<usize> {
        self.lattices.keys().cloned().collect()
    }

    /// Returns a hashset of each [`RewardModulatedLattice`] id
    pub fn get_all_reward_modulated_lattice_ids(&self) -> HashSet<usize> {
        self.reward_modulated_lattices.keys().cloned().collect()
    }

    /// Returns a hashset of each [`SpikeTrainLattice`] id
    pub fn get_all_spike_train_lattice_ids(&self) -> HashSet<usize> {
        self.spike_train_lattices.keys().cloned().collect()
    } 

    /// Returns a hashset of all the ids
    pub fn get_all_ids(&self) -> HashSet<usize> {
        let mut ids = HashSet::new();

        self.lattices.keys()
            .for_each(|i| { ids.insert(*i); });
        self.reward_modulated_lattices.keys()
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
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            if self.reward_modulated_lattices.contains_key(&postsynaptic_id) {
                return Err(LatticeNetworkError::ConnectFunctionMustHaveNonRewardModulatedLattice);
            } else {
                return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
            }
        }

        if self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            return Err(LatticeNetworkError::ConnectFunctionMustHaveNonRewardModulatedLattice);
        }

        if presynaptic_id == postsynaptic_id {
            self.connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
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
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(*i, *j));
                            self.connecting_graph.edit_weight(
                                &i_graph_pos, 
                                &j_graph_pos, 
                                Some(RewardModulatedConnection::Weight(weight))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
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
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(i.pos, *j));
                            self.connecting_graph.edit_weight(
                                i, 
                                &j_graph_pos, 
                                Some(RewardModulatedConnection::Weight(weight))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }
                });
        }

        Ok(())
    }

    /// Connects the neurons in lattices together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn falliable_connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            if self.reward_modulated_lattices.contains_key(&postsynaptic_id) {
                return Err(LatticeNetworkError::ConnectFunctionMustHaveNonRewardModulatedLattice);
            } else {
                return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
            }
        }

        if self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            return Err(LatticeNetworkError::ConnectFunctionMustHaveNonRewardModulatedLattice);
        }

        if presynaptic_id == postsynaptic_id {
            self.falliable_connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
        }

        let output: Result<Vec<_>, LatticeNetworkError> = if self.lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id)
                .unwrap()
                .graph;
            self.lattices.get(&presynaptic_id).unwrap()
                .graph
                .get_every_node()
                .iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j)? {
                            let weight = match weight_logic {
                                Some(func) => (func)(*i, *j)?,
                                None => 1.,
                            };
                            self.connecting_graph.edit_weight(
                                &i_graph_pos, 
                                &j_graph_pos, 
                                Some(RewardModulatedConnection::Weight(weight))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
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
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j)? {
                            let weight = match weight_logic {
                                Some(func) => (func)(i.pos, *j)?,
                                None => 1.,
                            };
                            self.connecting_graph.edit_weight(
                                i, 
                                &j_graph_pos, 
                                Some(RewardModulatedConnection::Weight(weight))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
        };

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Connects the neurons in lattices together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the potentially reward modulated weights between the 
    /// neurons should be, if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn connect_with_reward_modulation(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: &dyn Fn(Position, Position) -> RewardModulatedConnection<S>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.reward_modulated_lattices.contains_key(&postsynaptic_id) &&
        !self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            return Err(LatticeNetworkError::CannotConnectWithRewardModulatedConnection)
        }

        if !self.lattices.contains_key(&postsynaptic_id) && 
        !self.reward_modulated_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            return Err(LatticeNetworkError::RewardModulatedConnectionNotCompatibleInternally);
        }

        if self.lattices.contains_key(&presynaptic_id) || 
        self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph: GraphWrapper<U, C> = if self.lattices.contains_key(&postsynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            };

            let presynaptic_graph: GraphWrapper<U, C> = if self.lattices.contains_key(&presynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&presynaptic_id).unwrap().graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&presynaptic_id).unwrap().graph
                )
            };

            presynaptic_graph.get_every_node()
                .iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j) {
                            self.connecting_graph.edit_weight(
                                &i_graph_pos, &j_graph_pos, Some((weight_logic)(*i, *j))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
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

            let postsynaptic_graph = if self.lattices.contains_key(&postsynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            };

            presynaptic_positions.iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j) {
                            self.connecting_graph.edit_weight(
                                i, &j_graph_pos, Some((weight_logic)(i.pos, *j))
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }
                });
        }

        Ok(())
    }

    /// Connects the neurons in lattices together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the potentially reward modulated weights between the 
    /// neurons should be, if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn falliable_connect_with_reward_modulation(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: &dyn Fn(Position, Position) -> Result<RewardModulatedConnection<S>, LatticeNetworkError>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.reward_modulated_lattices.contains_key(&postsynaptic_id) &&
        !self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            return Err(LatticeNetworkError::CannotConnectWithRewardModulatedConnection)
        }

        if !self.lattices.contains_key(&postsynaptic_id) && 
        !self.reward_modulated_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            return Err(LatticeNetworkError::RewardModulatedConnectionNotCompatibleInternally);
        }

        let output: Result<Vec<_>, LatticeNetworkError> = if self.lattices.contains_key(&presynaptic_id) || 
        self.reward_modulated_lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph: GraphWrapper<U, C> = if self.lattices.contains_key(&postsynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            };

            let presynaptic_graph: GraphWrapper<U, C> = if self.lattices.contains_key(&presynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&presynaptic_id).unwrap().graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&presynaptic_id).unwrap().graph
                )
            };

            presynaptic_graph.get_every_node()
                .iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j)? {
                            self.connecting_graph.edit_weight(
                                &i_graph_pos, &j_graph_pos, Some((weight_logic)(*i, *j)?)
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
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

            let postsynaptic_graph = if self.lattices.contains_key(&postsynaptic_id) {
                GraphWrapper::Graph1(
                    &self.lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            } else {
                GraphWrapper::Graph2(
                    &self.reward_modulated_lattices.get(&postsynaptic_id)
                        .unwrap()
                        .graph
                )
            };

            presynaptic_positions.iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j)? {
                            self.connecting_graph.edit_weight(
                                i, &j_graph_pos, Some((weight_logic)(i.pos, *j)?)
                            ).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
        };

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetwork`] together given a 
    /// function to determine if the neurons should be connected given their position (usize, usize), 
    /// and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().connect(connecting_conditional, weight_logic);

        Ok(())
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetwork`] together given a 
    /// function (that can fail) to determine if the neurons should be connected given their position 
    /// (usize, usize), and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().falliable_connect(connecting_conditional, weight_logic)?;

        Ok(())
    }

    /// Connects the neurons in a [`RewardModulatedLattice`] within the [`RewardModulatedLatticeNetwork`] 
    /// together given a function to determine if the neurons should be connected given their position 
    /// (usize, usize), and a function to determine what the weight between the neurons should be,
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect_reward_modulated_lattice_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: &dyn Fn(Position, Position) -> S,
    ) -> Result<(), LatticeNetworkError> {
        if !self.reward_modulated_lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.reward_modulated_lattices.get_mut(&id)
            .unwrap()
            .connect(connecting_conditional, weight_logic);

        Ok(())
    }

    /// Connects the neurons in a [`RewardModulatedLattice`] within the [`RewardModulatedLatticeNetwork`] 
    /// together given a function to determine if the neurons should be connected given their position 
    /// (usize, usize), and a function to determine what the reward modulated weight between the neurons should be,
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect_reward_modulated_lattice_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: &dyn Fn(Position, Position) -> Result<S, LatticeNetworkError>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.reward_modulated_lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.reward_modulated_lattices.get_mut(&id)
            .unwrap()
            .falliable_connect(connecting_conditional, weight_logic)?;

        Ok(())
    }

    fn get_all_input_positions(&self, pos: GraphPosition) -> HashSet<GraphPosition> {
        let mut input_positions: HashSet<GraphPosition> = if self.lattices.contains_key(&pos.id) {
            self.lattices[&pos.id].graph
                .get_incoming_connections(&pos.pos)
                .expect("Cannot find position")
                .iter()
                .map(|i| GraphPosition { id: pos.id, pos: *i})
                .collect()
        } else {
            self.reward_modulated_lattices[&pos.id].graph
                .get_incoming_connections(&pos.pos)
                .expect("Cannot find position")
                .iter()
                .map(|i| GraphPosition { id: pos.id, pos: *i})
                .collect()
        };

        if let Ok(value) = self.connecting_graph.get_incoming_connections(&pos) {
            input_positions.extend(value)
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
    ) -> f32 {
        let is_not_reward_modulated = self.lattices.contains_key(&postsynaptic_position.id);

        let postsynaptic_neuron: &T = if is_not_reward_modulated {
            &self.lattices.get(&postsynaptic_position.id)
                .unwrap()
                .cell_grid[postsynaptic_position.pos.0][postsynaptic_position.pos.1]
        } else {
            &self.reward_modulated_lattices.get(&postsynaptic_position.id)
                .unwrap()
                .cell_grid[postsynaptic_position.pos.0][postsynaptic_position.pos.1]
        };

        let mut input_val = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;

                let final_input = if self.lattices.contains_key(&input_position.id) {
                    let input_cell = &self.lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    gap_junction(input_cell, postsynaptic_neuron)
                } else if self.reward_modulated_lattices.contains_key(&input_position.id) {
                    let input_cell = &self.reward_modulated_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    gap_junction(input_cell, postsynaptic_neuron)
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    spike_train_gap_juncton(input_cell, postsynaptic_neuron, self.internal_clock)
                };

                let weight: f32 = if input_position.id != postsynaptic_position.id {
                    self.connecting_graph.lookup_weight(input_position, postsynaptic_position)
                        .unwrap()
                        .unwrap()
                        .get_weight()
                } else if self.lattices.contains_key(&input_position.id) {
                    self.lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap_or(Some(0.))
                        .unwrap()
                } else {
                    self.reward_modulated_lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap()
                        .unwrap()
                        .get_weight()
                };

                final_input * weight
            })
            .sum::<f32>();

        let averager = match input_positions.len() {
            0 => 1.,
            _ => input_positions.len() as f32,
        };

        input_val /= averager;

        input_val
    }

    fn calculate_neurotransmitter_input_from_positions(
        &self, 
        postsynaptic_position: &GraphPosition,
        input_positions: &HashSet<GraphPosition>
    ) -> NeurotransmitterConcentrations<N> {
        let input_vals: Vec<NeurotransmitterConcentrations<N>> = input_positions
            .iter()
            .map(|input_position| {
                let (pos_x, pos_y) = input_position.pos;

                let mut neurotransmitter_input = if self.lattices.contains_key(&input_position.id) {
                    let input_cell = &self.lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    input_cell.get_neurotransmitter_concentrations()
                } else if self.reward_modulated_lattices.contains_key(&input_position.id) {
                    let input_cell = &self.reward_modulated_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    input_cell.get_neurotransmitter_concentrations()
                } else {
                    let input_cell = &self.spike_train_lattices.get(&input_position.id)
                        .unwrap()
                        .cell_grid[pos_x][pos_y];

                    input_cell.get_neurotransmitter_concentrations()
                };
                
                let weight: f32 = if input_position.id != postsynaptic_position.id {
                    self.connecting_graph.lookup_weight(input_position, postsynaptic_position)
                        .unwrap()
                        .unwrap()
                        .get_weight()
                } else if self.lattices.contains_key(&input_position.id) {
                    self.lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap_or(Some(0.))
                        .unwrap()
                } else {
                    self.reward_modulated_lattices.get(&input_position.id).unwrap()
                        .graph
                        .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                        .unwrap()
                        .unwrap()
                        .get_weight()
                };

                weight_neurotransmitter_concentration(&mut neurotransmitter_input, weight);

                neurotransmitter_input
            })
            .collect();

        aggregate_neurotransmitter_concentrations(&input_vals)
    }

    fn get_every_node(&self) -> HashSet<GraphPosition> {
        let mut nodes = HashSet::new();

        for i in self.lattices.values() {
            let current_nodes: HashSet<GraphPosition> = i.graph.get_every_node_as_ref()
                .iter()
                .map(|j| GraphPosition { id: i.get_id(), pos: **j})
                .collect();
            nodes.extend(current_nodes);
        }

        for i in self.reward_modulated_lattices.values() {
            let current_nodes: HashSet<GraphPosition> = i.graph.get_every_node_as_ref()
                .iter()
                .map(|j| GraphPosition { id: i.get_id(), pos: **j})
                .collect();
            nodes.extend(current_nodes);
        }

        nodes
    }

    fn get_all_electrical_inputs(&self) -> HashMap<GraphPosition, f32> {
        self.get_every_node()
            .iter()
            .map(|pos| {
                let input_positions = self.get_all_input_positions(*pos);

                let input = self.calculate_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, input)
            })
            .collect()
    }

    fn get_all_neurotransmitter_inputs(&self) -> 
    HashMap<GraphPosition, NeurotransmitterConcentrations<N>> {
        self.get_every_node()
            .iter()
            .map(|pos| {
                let input = self.calculate_neurotransmitter_input_from_positions(
                    pos,
                    &self.get_all_input_positions(*pos),
                );

                (*pos, input)
            })
            .collect()
    }

    fn get_all_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f32>, HashMap<GraphPosition, NeurotransmitterConcentrations<N>>) {
        let neurotransmitters_inputs = self.get_all_neurotransmitter_inputs();

        let inputs = self.get_all_electrical_inputs();

        (inputs, neurotransmitters_inputs)
    }

    fn par_get_all_electrical_inputs(&self) -> HashMap<GraphPosition, f32> {
        self.get_every_node()
            .par_iter()
            .map(|pos| {
                let input_positions = self.get_all_input_positions(*pos);

                let input = self.calculate_electrical_input_from_positions(
                    pos,
                    &input_positions,
                );

                (*pos, input)
            })
            .collect()
    }

    fn par_get_all_neurotransmitter_inputs(&self) -> 
    HashMap<GraphPosition, NeurotransmitterConcentrations<N>> {
        self.get_every_node()
            .par_iter()
            .map(|pos| {
                let input = self.calculate_neurotransmitter_input_from_positions(
                    pos,
                    &self.get_all_input_positions(*pos),
                );

                (*pos, input)
            })
            .collect()
    }

    fn par_get_all_electrical_and_neurotransmitter_inputs(&self) -> 
    (HashMap<GraphPosition, f32>, HashMap<GraphPosition, NeurotransmitterConcentrations<N>>) {
        let neurotransmitters_inputs = self.par_get_all_neurotransmitter_inputs();

        let inputs = self.par_get_all_electrical_inputs();

        (inputs, neurotransmitters_inputs)
    }

    fn update_weights_from_neurons_across_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = &self.lattices.get(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];

        for input_pos in self.connecting_graph.get_incoming_connections(pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos.pos;

            let connection = self.connecting_graph
                .lookup_weight(&input_pos, pos)
                .unwrap()
                .unwrap();

            match connection {
                RewardModulatedConnection::Weight(mut weight) => {
                    if self.lattices.contains_key(&input_pos.id) {
                        current_lattice.plasticity.update_weight(
                            &mut weight,
                            &self.lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                            given_neuron,
                        );
                    } else {
                        current_lattice.plasticity.update_weight(
                            &mut weight,
                            &self.spike_train_lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                            given_neuron,
                        );
                    }
                                                
                    self.connecting_graph
                        .edit_weight(
                            &input_pos, 
                            pos, 
                            Some(RewardModulatedConnection::Weight(weight)),
                        )?;
                },
                RewardModulatedConnection::RewardModulatedWeight(mut weight) => {
                    self.get_reward_modulated_lattice(&input_pos.id).unwrap().reward_modulator
                        .update_weight(
                            &mut weight, 
                            &self.get_reward_modulated_lattice(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                            given_neuron,
                        );
                    
                    self.connecting_graph
                        .edit_weight(
                            &input_pos, 
                            pos, 
                            Some(RewardModulatedConnection::RewardModulatedWeight(weight)),
                        )?;
                }
            }
        }

        for output_pos in self.connecting_graph.get_outgoing_connections(pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos.pos;

            let connection = self.connecting_graph
                .lookup_weight(&output_pos, pos)
                .unwrap()
                .unwrap();

            match connection {
                RewardModulatedConnection::Weight(mut weight) => {
                    current_lattice.plasticity.update_weight(
                        &mut weight,
                        given_neuron,
                        &self.lattices.get(&output_pos.id).unwrap().cell_grid[x_out][y_out], 
                    );
                                                
                    self.connecting_graph
                        .edit_weight(
                            pos, 
                            &output_pos, 
                            Some(RewardModulatedConnection::Weight(weight)),
                        )?;
                },
                RewardModulatedConnection::RewardModulatedWeight(mut weight) => {
                    self.get_reward_modulated_lattice(&output_pos.id).unwrap().reward_modulator
                        .update_weight(
                            &mut weight, 
                            given_neuron,
                            &self.get_reward_modulated_lattice(&output_pos.id).unwrap().cell_grid[x_out][y_out], 
                        );
                    
                    self.connecting_graph
                        .edit_weight(
                            pos, 
                            &output_pos, 
                            Some(RewardModulatedConnection::RewardModulatedWeight(weight)),
                        )?;
                }
            }
        }

        Ok(())
    }

    fn update_weights_from_neurons_within_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = self.lattices.get_mut(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];
        
        for input_pos in current_lattice.graph.get_incoming_connections(&pos.pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos;

            let mut current_weight: f32 = current_lattice.graph
                .lookup_weight(&input_pos, &pos.pos)
                .unwrap_or(Some(0.))
                .unwrap();

            current_lattice.plasticity.update_weight(
                &mut current_weight,
                &current_lattice.cell_grid[x_in][y_in], 
                given_neuron,
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &input_pos, 
                    &pos.pos, 
                    Some(current_weight)
                )?;
        }

        for output_pos in current_lattice.graph.get_outgoing_connections(&pos.pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos;

            let mut current_weight: f32 = current_lattice.graph
                .lookup_weight(&pos.pos, &output_pos)
                .unwrap_or(Some(0.))
                .unwrap();

            current_lattice.plasticity.update_weight(
                &mut current_weight,
                given_neuron,
                &current_lattice.cell_grid[x_out][y_out], 
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &pos.pos, 
                    &output_pos, 
                    Some(current_weight)
                )?;
        }

        Ok(())
    }

    fn update_weights_from_neurons_across_reward_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = &self.reward_modulated_lattices.get(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];

        for input_pos in self.connecting_graph.get_incoming_connections(pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos.pos;

            let connection = self.connecting_graph
                .lookup_weight(&input_pos, pos)
                .unwrap()
                .unwrap();

            match connection {
                RewardModulatedConnection::Weight(mut weight) => {
                        if self.lattices.contains_key(&input_pos.id) {
                            self.get_lattice(&input_pos.id).unwrap().plasticity.update_weight(
                                &mut weight,
                                &self.lattices.get(&input_pos.id).unwrap().cell_grid[x_in][y_in], 
                                given_neuron,
                            );
                                                        
                            self.connecting_graph
                                .edit_weight(
                                    &input_pos, 
                                    pos, 
                                    Some(RewardModulatedConnection::Weight(weight)),
                                )?;
                        }
                    },
                RewardModulatedConnection::RewardModulatedWeight(mut weight) => {
                    if self.reward_modulated_lattices.contains_key(&input_pos.id) {
                        let input_neuron = &self.get_reward_modulated_lattice(&input_pos.id).unwrap()
                            .cell_grid[x_in][y_in];

                        current_lattice.reward_modulator
                            .update_weight(
                                &mut weight, 
                                input_neuron, 
                                given_neuron,
                            );
                    } else if self.lattices.contains_key(&input_pos.id) {
                        let input_neuron = &self.get_lattice(&input_pos.id).unwrap()
                            .cell_grid[x_in][y_in];

                        current_lattice.reward_modulator
                            .update_weight(
                                &mut weight, 
                                input_neuron, 
                                given_neuron,
                            );
                    } else {
                        let input_neuron = &self.get_spike_train_lattice(&input_pos.id).unwrap()
                            .cell_grid[x_in][y_in];

                        current_lattice.reward_modulator
                            .update_weight(
                                &mut weight, 
                                input_neuron, 
                                given_neuron,
                            );
                    };
                    
                    self.connecting_graph
                        .edit_weight(
                            &input_pos, 
                            pos, 
                            Some(RewardModulatedConnection::RewardModulatedWeight(weight)),
                        )?;
                }
            }
        }

        for output_pos in self.connecting_graph.get_outgoing_connections(pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos.pos;

            let connection = self.connecting_graph
                .lookup_weight(&output_pos, pos)
                .unwrap()
                .unwrap();

            match connection {
                RewardModulatedConnection::Weight(mut weight) => {
                    if self.lattices.contains_key(&output_pos.id) {
                        self.get_lattice(&output_pos.id).unwrap().plasticity.update_weight(
                            &mut weight,
                            given_neuron,
                            &self.lattices.get(&output_pos.id).unwrap().cell_grid[x_out][y_out], 
                        );
                                                    
                        self.connecting_graph
                            .edit_weight(
                                pos, 
                                &output_pos, 
                                Some(RewardModulatedConnection::Weight(weight)),
                            )?;
                    }
                },
                RewardModulatedConnection::RewardModulatedWeight(mut weight) => {
                    let output_neuron = if self.reward_modulated_lattices.contains_key(&output_pos.id) {
                        &self.get_reward_modulated_lattice(&output_pos.id).unwrap().cell_grid[x_out][y_out]
                    } else {
                        &self.get_lattice(&output_pos.id).unwrap().cell_grid[x_out][y_out]
                    };

                    current_lattice.reward_modulator
                        .update_weight(
                            &mut weight, 
                            given_neuron,
                            output_neuron, 
                        );
                    
                    self.connecting_graph
                        .edit_weight(
                            pos, 
                            &output_pos, 
                            Some(RewardModulatedConnection::RewardModulatedWeight(weight)),
                        )?;
                }
            }
        }

        Ok(())
    }

    fn update_weights_from_neurons_within_reward_lattices(&mut self, x: usize, y: usize, pos: &GraphPosition) -> Result<(), GraphError> {
        let current_lattice = self.reward_modulated_lattices.get_mut(&pos.id).unwrap();
        let given_neuron = &current_lattice.cell_grid[x][y];
        
        for input_pos in current_lattice.graph.get_incoming_connections(&pos.pos).unwrap_or_default() {
            let (x_in, y_in) = input_pos;

            let mut current_weight = current_lattice.graph
                .lookup_weight(&input_pos, &pos.pos)
                .unwrap()
                .unwrap();

            current_lattice.reward_modulator.update_weight(
                &mut current_weight,
                &current_lattice.cell_grid[x_in][y_in], 
                given_neuron,
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &input_pos, 
                    &pos.pos, 
                    Some(current_weight)
                )?;
        }

        for output_pos in current_lattice.graph.get_outgoing_connections(&pos.pos).unwrap_or_default() {
            let (x_out, y_out) = output_pos;

            let mut current_weight = current_lattice.graph
                .lookup_weight(&pos.pos, &output_pos)
                .unwrap()
                .unwrap();

                current_lattice.reward_modulator.update_weight(
                &mut current_weight,
                given_neuron,
                &current_lattice.cell_grid[x_out][y_out], 
            );
                                        
            current_lattice.graph
                .edit_weight(
                    &pos.pos, 
                    &output_pos, 
                    Some(current_weight)
                )?;
        }

        Ok(())
    }

    fn post_neuron_update_step(
        &mut self,
        positions_to_update: &[(usize, usize, GraphPosition)],
        positions_to_update_with_reward_modulation: &[(usize, usize, GraphPosition)],
    ) -> Result<(), GraphError> {
        for (x, y, pos) in positions_to_update {
            self.update_weights_from_neurons_across_lattices(*x, *y, pos)?;
            self.update_weights_from_neurons_within_lattices(*x, *y, pos)?;
        }

        for (x, y, pos) in positions_to_update_with_reward_modulation {
            self.update_weights_from_neurons_across_reward_lattices(*x, *y, pos)?;
            self.update_weights_from_neurons_within_reward_lattices(*x, *y, pos)?;
        }

        if self.update_connecting_graph_history {
            self.connecting_graph.update_history();
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
        inputs: &HashMap<GraphPosition, f32>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let input_value = *inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_and_spike(input_value);
    
                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        let mut positions_to_update_with_reward_modulation = Vec::new();

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            for pos in reward_modulated_lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: reward_modulated_lattice.get_id(), pos };

                let input_value = *inputs.get(&graph_pos).unwrap();

                let is_spiking = reward_modulated_lattice.cell_grid[x][y].iterate_and_spike(input_value);
    
                if is_spiking { 
                    reward_modulated_lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <R as RewardModulator<T, T, S>>::do_update(
                    &reward_modulated_lattice.reward_modulator, &reward_modulated_lattice.cell_grid[x][y]
                ) && reward_modulated_lattice.do_modulation {
                    positions_to_update_with_reward_modulation.push((x, y, graph_pos));
                }
            }
    
            if reward_modulated_lattice.update_graph_history {
                reward_modulated_lattice.graph.update_history();
            }
            if reward_modulated_lattice.update_grid_history {
                reward_modulated_lattice.grid_history.update(&reward_modulated_lattice.cell_grid);
            }
        }

        self.post_neuron_update_step(&positions_to_update, &positions_to_update_with_reward_modulation)?;

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of only chemical inputs
    pub fn iterate_with_chemical_synapses_only(
        &mut self,
        neurotransmitter_inputs: &HashMap<GraphPosition, NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let neurotransmitter_input = neurotransmitter_inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    0., neurotransmitter_input,
                );
    
                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        let mut positions_to_update_with_reward_modulation = Vec::new();

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            for pos in reward_modulated_lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: reward_modulated_lattice.get_id(), pos };

                let neurotransmitter_input = neurotransmitter_inputs.get(&graph_pos).unwrap();

                let is_spiking = reward_modulated_lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    0., neurotransmitter_input,
                );

                if is_spiking { 
                    reward_modulated_lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <R as RewardModulator<T, T, S>>::do_update(
                    &reward_modulated_lattice.reward_modulator, &reward_modulated_lattice.cell_grid[x][y]
                ) && reward_modulated_lattice.do_modulation {
                    positions_to_update_with_reward_modulation.push((x, y, graph_pos));
                }
            }
    
            if reward_modulated_lattice.update_graph_history {
                reward_modulated_lattice.graph.update_history();
            }
            if reward_modulated_lattice.update_grid_history {
                reward_modulated_lattice.grid_history.update(&reward_modulated_lattice.cell_grid);
            }
        }

        self.post_neuron_update_step(&positions_to_update, &positions_to_update_with_reward_modulation)?;

        Ok(())
    }

    /// Iterates one simulation timestep lattice given a set of only chemical inputs
    pub fn iterate_with_chemical_and_electrical_synapses(
        &mut self,
        inputs: &HashMap<GraphPosition, f32>,
        neurotransmitter_inputs: &HashMap<GraphPosition, NeurotransmitterConcentrations<N>>,
    ) -> Result<(), GraphError> {
        let mut positions_to_update = Vec::new();

        for lattice in self.lattices.values_mut() {
            for pos in lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: lattice.get_id(), pos };

                let input = inputs.get(&graph_pos).unwrap();
                let neurotransmitter_input = neurotransmitter_inputs.get(&graph_pos).unwrap();

                let is_spiking = lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    *input, neurotransmitter_input,
                );
    
                if is_spiking { 
                    lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <Z as Plasticity<T, T, f32>>::do_update(&lattice.plasticity, &lattice.cell_grid[x][y]) && lattice.do_plasticity {
                    positions_to_update.push((x, y, graph_pos));
                }
            }
    
            if lattice.update_graph_history {
                lattice.graph.update_history();
            }
            if lattice.update_grid_history {
                lattice.grid_history.update(&lattice.cell_grid);
            }
        }

        let mut positions_to_update_with_reward_modulation = Vec::new();

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            for pos in reward_modulated_lattice.graph.get_every_node() {
                let (x, y) = pos;
                let graph_pos = GraphPosition { id: reward_modulated_lattice.get_id(), pos };

                let input = inputs.get(&graph_pos).unwrap();
                let neurotransmitter_input = neurotransmitter_inputs.get(&graph_pos).unwrap();

                let is_spiking = reward_modulated_lattice.cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
                    *input, neurotransmitter_input,
                );

                if is_spiking { 
                    reward_modulated_lattice.cell_grid[x][y].set_last_firing_time(Some(self.internal_clock));
                }
    
                if <R as RewardModulator<T, T, S>>::do_update(
                    &reward_modulated_lattice.reward_modulator, &reward_modulated_lattice.cell_grid[x][y]
                ) && reward_modulated_lattice.do_modulation {
                    positions_to_update_with_reward_modulation.push((x, y, graph_pos));
                }
            }
    
            if reward_modulated_lattice.update_graph_history {
                reward_modulated_lattice.graph.update_history();
            }
            if reward_modulated_lattice.update_grid_history {
                reward_modulated_lattice.grid_history.update(&reward_modulated_lattice.cell_grid);
            }
        }

        self.post_neuron_update_step(&positions_to_update, &positions_to_update_with_reward_modulation)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical synapses only
    fn run_lattices_electrical_synapses_only(&mut self, reward: f32) -> Result<(), GraphError> {
        let inputs = if self.parallel {
            self.par_get_all_electrical_inputs()
        } else {
            self.get_all_electrical_inputs()
        };

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            <R as RewardModulator<T, T, S>>::update(
                &mut reward_modulated_lattice.reward_modulator, 
                reward
            );
        }

        self.iterate(&inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical synapses only without a reward signal
    fn run_lattices_electrical_synapses_only_without_reward(&mut self) -> Result<(), GraphError> {
        let inputs = if self.parallel {
            self.par_get_all_electrical_inputs()
        } else {
            self.get_all_electrical_inputs()
        };

        self.iterate(&inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// chemical synapses only
    fn run_lattices_chemical_synapses_only(&mut self, reward: f32) -> Result<(), GraphError> {
        let neurotransmitter_inputs = if self.parallel {
            self.par_get_all_neurotransmitter_inputs()
        } else {
            self.get_all_neurotransmitter_inputs()
        };

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            <R as RewardModulator<T, T, S>>::update(
                &mut reward_modulated_lattice.reward_modulator, 
                reward
            );
        }

        self.iterate_with_chemical_synapses_only(&neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// chemical synapses only without a reward signal
    fn run_lattices_chemical_synapses_only_without_reward(&mut self) -> Result<(), GraphError> {
        let neurotransmitter_inputs = if self.parallel {
            self.par_get_all_neurotransmitter_inputs()
        } else {
            self.get_all_neurotransmitter_inputs()
        };

        self.iterate_with_chemical_synapses_only(&neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical and chemical synapses
    fn run_lattices_with_electrical_and_chemical_synapses(&mut self, reward: f32) -> Result<(), GraphError> {
        let (inputs, neurotransmitter_inputs) = if self.parallel {
            self.par_get_all_electrical_and_neurotransmitter_inputs()
        } else {
            self.get_all_electrical_and_neurotransmitter_inputs()
        };

        for reward_modulated_lattice in self.reward_modulated_lattices.values_mut() {
            <R as RewardModulator<T, T, S>>::update(
                &mut reward_modulated_lattice.reward_modulator, 
                reward
            );
        }

        self.iterate_with_chemical_and_electrical_synapses(&inputs, &neurotransmitter_inputs)?;

        Ok(())
    }

    /// Calculates inputs for the lattice, iterates, and applies reward for one timestep for
    /// electrical and chemical synapses without a reward signal
    fn run_lattices_with_electrical_and_chemical_synapses_without_reward(&mut self) -> Result<(), GraphError> {
        let (inputs, neurotransmitter_inputs) = if self.parallel {
            self.par_get_all_electrical_and_neurotransmitter_inputs()
        } else {
            self.get_all_electrical_and_neurotransmitter_inputs()
        };

        self.iterate_with_chemical_and_electrical_synapses(&inputs, &neurotransmitter_inputs)?;

        Ok(())
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag
    pub fn run_lattices(&mut self, reward: f32) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattices_with_electrical_and_chemical_synapses(reward),
            (true, false) => self.run_lattices_electrical_synapses_only(reward),
            (false, true) => self.run_lattices_chemical_synapses_only(reward),
            (false, false) => Ok(()),
        }
    }

    /// Runs lattice given reward and dispatches correct run lattice method based on
    /// electrical and chemical synapses flag without a reward signal
    pub fn run_lattices_without_reward(&mut self) -> Result<(), GraphError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, true) => self.run_lattices_with_electrical_and_chemical_synapses_without_reward(),
            (true, false) => self.run_lattices_electrical_synapses_only_without_reward(),
            (false, true) => self.run_lattices_chemical_synapses_only_without_reward(),
            (false, false) => Ok(()),
        }
    }
}

impl<S, T, U, V, W, X, Y, Z, R, C, N> Agent for RewardModulatedLatticeNetwork<S, T, U, V, W, X, Y, Z, R, C, N>
where
    S: RewardModulatedWeight,
    T: IterateAndSpike<N=N>, 
    U: Graph<K=(usize, usize), V=f32>, 
    V: LatticeHistory, 
    W: SpikeTrain<N=N>, 
    X: SpikeTrainLatticeHistory,
    Y: Graph<K=GraphPosition, V=RewardModulatedConnection<S>>,
    Z: Plasticity<T, T, f32> + Plasticity<W, T, f32>,
    R: RewardModulator<T, T, S> + RewardModulator<W, T, S>,
    C: Graph<K=(usize, usize), V=S>,
    N: NeurotransmitterType,
{
    fn update_and_apply_reward(&mut self, reward: f32) -> Result<(), AgentError> {
        match self.run_lattices(reward) {
            Ok(()) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(format!("Agent error: {}", e))),
        }
    }

    fn update(&mut self) -> Result<(), AgentError> {
        match self.run_lattices_without_reward() {
            Ok(()) => Ok(()),
            Err(e) => Err(AgentError::AgentIterationFailure(format!("Agent error: {}", e))),
        }
    }
}

/// Generates a default agent type for a reward modulated lattice
#[macro_export]
#[doc(hidden)]
macro_rules! raw_create_agent_type_for_lattice {
    (
        $name:ident,
        $reward_mod_weight:ty,
        $reward_modulator:ty,
        $iterate_and_spike:ty,
        $neurotransitter_kind:ty,
    ) => {
        type $name = RewardModulatedLattice<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            spiking_neural_networks::neuron::GridVoltageHistory,
            $reward_modulator,
            $neurotransitter_kind,
        >;
    };

    (
        $name:ident,
        $reward_mod_weight:ty,
        $iterate_and_spike:ty,
        $reward_modulator:ty,
        $neurotransitter_kind:ty,
        lattice_history = $lattice_history:ty
    ) => {
        type $name = RewardModulatedLattice<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            $lattice_history,
            $reward_modulator,
            $neurotransitter_kind,
        >;
    };
}

#[doc(inline)]
pub use raw_create_agent_type_for_lattice as create_agent_type_for_lattice;

/// Generates a default agent type for a reward modulated lattice network
#[doc(hidden)]
#[macro_export]
macro_rules! raw_create_agent_type_for_network {
    (
        $name:ident,
        $plasticity:ty,
        $reward_mod_plasticity:ty,
        $reward_mod_weight:ty,
        $iterate_and_spike:ty,
        $spike_train:ty,
        $neurotransitter_kind:ty,
    ) => {
        type $name = RewardModulatedLatticeNetwork<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), f32>,
            spiking_neural_networks::neuron::GridVoltageHistory,
            $spike_train,
            spiking_neural_networks::neuron::SpikeTrainGridHistory,
            spiking_neural_networks::graph::AdjacencyMatrix<spiking_neural_networks::graph::GraphPosition, RewardModulatedConnection<$reward_mod_weight>>,
            $plasticity,
            $reward_mod_plasticity,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            $neurotransitter_kind,
        >;
    };

    
    (
        $name:ident,
        $plasticity:ty,
        $reward_mod_plasticity:ty,
        $reward_mod_weight:ty,
        $iterate_and_spike:ty,
        $spike_train:ty,
        $neurotransitter_kind:ty,
        lattice_history = $lattice_history:ty,
    ) => {
        type $name = RewardModulatedLatticeNetwork<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), f32>,
            $lattice_history,
            $spike_train,
            spiking_neural_networks::neuron::SpikeTrainGridHistory,
            spiking_neural_networks::graph::AdjacencyMatrix<
                spiking_neural_networks::graph::GraphPosition, 
                spiking_neural_networks::neuron::RewardModulatedConnection<$reward_mod_weight>
            >,
            $plasticity,
            $reward_mod_plasticity,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            $neurotransitter_kind,
        >;
    };

    (
        $name:ident,
        $plasticity:ty,
        $reward_mod_plasticity:ty,
        $reward_mod_weight:ty,
        $iterate_and_spike:ty,
        $spike_train:ty,
        $neurotransitter_kind:ty,
        spike_train_lattice_history = $spike_train_lattice_history:ty
    ) => {
        type $name = RewardModulatedLatticeNetwork<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), f32>,
            spiking_neural_networks::neuron::GridVoltageHistory,
            $spike_train,
            $spike_train_lattice_history,
            spiking_neural_networks::graph::AdjacencyMatrix<
                spiking_neural_networks::graph::GraphPosition, 
                spiking_neural_networks::neuron::RewardModulatedConnection<$reward_mod_weight>
            >,
            $plasticity,
            $reward_mod_plasticity,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            $neurotransitter_kind,
        >;
    };

    (
        $name:ident,
        $plasticity:ty,
        $reward_mod_plasticity:ty,
        $reward_mod_weight:ty,
        $iterate_and_spike:ty,
        $spike_train:ty,
        $neurotransitter_kind:ty,
        lattice_history = $lattice_history:ty,
        spike_train_lattice_history = $spike_train_lattice_history:ty
    ) => {
        type $name = RewardModulatedLatticeNetwork<
            $reward_mod_weight,
            $iterate_and_spike,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), f32>,
            $lattice_history,
            $spike_train,
            $spike_train_lattice_history,
            spiking_neural_networks::graph::AdjacencyMatrix<
                spiking_neural_networks::graph::GraphPosition, 
                spiking_neural_networks::neuron::RewardModulatedConnection<$reward_mod_weight>
            >,
            $plasticity,
            $reward_mod_plasticity,
            spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), $reward_mod_weight>,
            $neurotransitter_kind,
        >;
    };
}

#[doc(inline)]
pub use raw_create_agent_type_for_network as create_agent_type_for_network;
