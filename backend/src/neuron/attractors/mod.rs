//! An set of tools to generate weights for attractors like a Hopfield network
//! as well as a simplified neuron model for very basic testing of attractor dynamics.
//! 
//! Example bipolar autoassociative attractors:
//! ```rust
//! # use spiking_neural_networks::{
//! #     neuron::{
//! #         integrate_and_fire::IzhikevichNeuron,
//! #         plasticity::STDP,
//! #         attractors::{generate_random_patterns, generate_hopfield_network, distort_pattern},
//! #         Lattice, SpikeHistory,
//! #     },
//! #     graph::AdjacencyMatrix,
//! #     error::SpikingNeuralNetworksError,
//! # };
//! #
//! fn main() -> Result<(), SpikingNeuralNetworksError> {
//!     let (num_rows, num_cols) = (7, 7);
//!     let base_neuron = IzhikevichNeuron {
//!         gap_conductance: 5.,
//!         ..IzhikevichNeuron::default_impl()
//!     };
//! 
//!     let mut lattice: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
//!     lattice.parallel = true;
//!     lattice.update_grid_history = true;
//!     lattice.populate(&base_neuron, num_rows, num_cols);
//!     lattice.set_dt(1.);
//! 
//!     // generates random patterns and associated weights
//!     let random_patterns = generate_random_patterns(num_rows, num_cols, 1, 0.5);
//!     let bipolar_connections = generate_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(
//!         0,
//!         &random_patterns,
//!     )?;
//!     lattice.set_graph(bipolar_connections)?;
//! 
//!     // initializes lattice with distorted pattern
//!     let pattern_index = 0;
//!     let input_pattern = distort_pattern(&random_patterns[pattern_index], 0.1);
//!     lattice.apply_given_position(|pos, neuron| {
//!         if input_pattern[pos.0][pos.1] {
//!             neuron.current_voltage = neuron.v_th;
//!         } else {
//!             neuron.current_voltage = neuron.v_init;
//!         }
//!     });
//! 
//!     lattice.run_lattice(1000)?;
//! 
//!     // associates each firing rate to a low and high state
//!     let firing_rates = lattice.grid_history.aggregate();
//!     let firing_threshold: isize = 5;
//!     let predicted_pattern: Vec<Vec<bool>> = firing_rates.iter()
//!         .map(|row| {
//!             row.iter().map(|i| *i >= firing_threshold).collect::<Vec<bool>>()
//!         })
//!         .collect();
//!     
//!     // checks accuracy of recall
//!     let mut accuracy = 0.;
//!     for (row1, row2) in predicted_pattern.iter().zip(random_patterns[pattern_index].iter()) {
//!         for (item1, item2) in row1.iter().zip(row2.iter()) {
//!             if item1 == item2 {
//!                 accuracy += 1.;
//!             }
//!         }
//!     }
//!     
//!     assert!(accuracy / (num_rows * num_cols) as f32 >= 0.9);
//! 
//!     Ok(())
//! }
//! ```
//! 
//! Example binary autoassociative network:
//! ```rust
//! # use rand::Rng;
//! # use spiking_neural_networks::{
//! #     neuron::{
//! #         integrate_and_fire::IzhikevichNeuron,
//! #         plasticity::STDP,
//! #         attractors::{
//! #            generate_random_patterns, generate_binary_hopfield_network, distort_pattern
//! #         },
//! #         Lattice, LatticeNetwork, SpikeHistory,
//! #     },
//! #     graph::AdjacencyMatrix,
//! #     error::SpikingNeuralNetworksError,
//! # };
//! #
//! fn main() -> Result<(), SpikingNeuralNetworksError> {
//!     let (num_rows, num_cols) = (5, 5);
//!     let base_neuron = IzhikevichNeuron {
//!         gap_conductance: 5.,
//!         ..IzhikevichNeuron::default_impl()
//!     };
//! 
//!     let mut inh: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
//!     inh.populate(&base_neuron, 3, 3);
//!     inh.apply(|neuron| 
//!         neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_init..=neuron.v_th)
//!     );
//!     inh.connect(&(|x, y| x != y), Some(&(|_, _| -1.5)));
//! 
//!     let mut exc: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
//!     exc.update_grid_history = true;
//!     exc.populate(&base_neuron, num_rows, num_cols);
//! 
//!     // generates random patterns and associated weights
//!     let random_patterns = generate_random_patterns(num_rows, num_cols, 1, 0.5);
//!     let binary_connections = generate_binary_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(
//!         1,
//!         &random_patterns,
//!         1.,
//!         1.,
//!         0.5,
//!     )?;
//!     exc.set_graph(binary_connections)?;
//!     exc.set_id(1);
//! 
//!     // initializes lattice with distorted pattern
//!     let pattern_index = 0;
//!     let input_pattern = distort_pattern(&random_patterns[pattern_index], 0.1);
//!     exc.apply_given_position(|pos, neuron| {
//!         if input_pattern[pos.0][pos.1] {
//!             neuron.current_voltage = neuron.v_th;
//!         } else {
//!             neuron.current_voltage = neuron.v_init;
//!         }
//!     });
//! 
//!     let mut network = LatticeNetwork::default_impl();
//!     network.add_lattice(exc)?;
//!     network.add_lattice(inh)?;
//!     network.parallel = true;
//!     network.connect(
//!         0, 1, &(|_, _| true), Some(&(|_, _| -2.))
//!     );
//!     network.connect(
//!         1, 0, &(|_, _| true), Some(&(|_, _| 1.))
//!     );
//! 
//!     network.set_dt(1.);
//!     network.run_lattices(1000)?;
//! 
//!     // associates each firing rate to a low and high state
//!     let mut firing_rates = network.get_lattice(&1).expect("Could not retrieve lattice")
//!         .grid_history.aggregate();
//!     let firing_threshold: isize = 10;
//!     let predicted_pattern: Vec<Vec<bool>> = firing_rates.iter()
//!         .map(|row| {
//!             row.iter().map(|i| *i >= firing_threshold).collect::<Vec<bool>>()
//!         })
//!         .collect();
//!     
//!     // checks accuracy of recall
//!     let mut accuracy = 0.;
//!     for (row1, row2) in predicted_pattern.iter().zip(random_patterns[pattern_index].iter()) {
//!         for (item1, item2) in row1.iter().zip(row2.iter()) {
//!             if item1 == item2 {
//!                 accuracy += 1.;
//!             }
//!         }
//!     }
//!     
//!     assert!(accuracy / (num_rows * num_cols) as f32 >= 0.85);
//! 
//!     Ok(())
//! }
//! ```
//! 
//! Example simple ring attractor:
//! ```rust
//! # use rand::Rng;
//! # use spiking_neural_networks::{
//! #     neuron::{
//! #         integrate_and_fire::IzhikevichNeuron,
//! #         spike_train::PoissonNeuron,
//! #         plasticity::STDP,
//! #         Lattice, SpikeTrainLattice, LatticeNetwork, SpikeHistory,
//! #     },
//! #     graph::AdjacencyMatrix,
//! #     error::SpikingNeuralNetworksError,
//! # };
//! # 
//! fn main() -> Result<(), SpikingNeuralNetworksError> {
//!     // setup spike train
//!     let poisson = PoissonNeuron { chance_of_firing : 0.005, ..PoissonNeuron::default_impl() };
//!     let mut spike_train_lattice = SpikeTrainLattice::default_impl();
//!     spike_train_lattice.populate(&poisson, 1, 1);
//!     spike_train_lattice.set_id(0);
//! 
//!     // ring attractor parameters
//!     let n_neurons = 120;
//!     let preferred_direction = n_neurons / 2;
//! 
//!     // ring attractor lattice
//!     let mut lattice: Lattice<_, AdjacencyMatrix<_, _>, SpikeHistory, STDP, _> = Lattice::default();
//!     lattice.populate(&IzhikevichNeuron::default_impl(), n_neurons, 1);
//!     // inhibit far neurons, excite close ones
//!     let ring_distance = |x: isize, y: isize| -> f32 { 
//!         (x - y).abs().min(n_neurons as isize - (x - y).abs()) as f32 
//!     };
//!     lattice.connect(
//!         &(|x, y| x != y),
//!         Some(&(|x, y| 
//!             (-2. * ring_distance(x.0 as isize, y.0 as isize).powf(2.) / 
//!             (n_neurons as f32 * 10.)).exp() - 0.3)
//!         ),
//!     );
//!     lattice.apply(|neuron|
//!         neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_init..=neuron.v_th)
//!     );
//!     lattice.update_grid_history = true;
//!     lattice.set_id(1);
//! 
//!     let mut ring_attractor = LatticeNetwork::default_impl();
//!     ring_attractor.add_lattice(lattice)?;
//!     ring_attractor.add_spike_train_lattice(spike_train_lattice)?;
//!     ring_attractor.connect(
//!         0, 
//!         1, 
//!         &(|_, _| true),
//!         Some(&(|_, y| 
//!             5. * (-2. * 
//!                 (preferred_direction as isize - y.0 as isize).pow(2) as f32 / 
//!                 (n_neurons as f32 * 10.)).exp() - 0.3
//!             )
//!         ),
//!     )?;
//!     ring_attractor.parallel = true;
//!     ring_attractor.set_dt(1.);
//! 
//!     // run ring attractor
//!     ring_attractor.run_lattices(1_000)?;
//!     
//!     let firing_rates = ring_attractor.get_lattice(&1).expect("Could not retrieve lattice")
//!         .grid_history.aggregate();
//!     let maximal_firing_rate = firing_rates.iter().map(|i| i[0])
//!         .max().unwrap_or(0);
//!     
//!     // check if neuron at preferred direction is one of high firing neurons
//!     assert!(firing_rates[preferred_direction][0] as f32 >= maximal_firing_rate as f32 * 0.9);
//! 
//!     Ok(())
//! }
//! ```

use std::result;
use rand::Rng;
use rand_distr::{Binomial, Distribution};
use crate::error::{GraphError, PatternError, SpikingNeuralNetworksError};
use crate::graph::Graph;


/// State of a bipolar discrete neuron
pub enum DiscreteNeuronState {
    /// Active (or `1`)
    Active,
    /// Inactive (or `-1`)
    Inactive,
}

/// Bipolar discrete neuron
pub struct DiscreteNeuron {
    /// Current state of the neuron
    pub state: DiscreteNeuronState
}

impl Default for DiscreteNeuron {
    fn default() -> Self {
        DiscreteNeuron { state: DiscreteNeuronState::Inactive }
    }
}

impl DiscreteNeuron {
    /// Updates state of neuron based on input, if positive the neuron is set to 
    /// [`DiscreteNeuronState::Active`], otherwise it becomes [`DiscreteNeuronState::Inactive`]
    pub fn update(&mut self, input: f32) {
        match input > 0. {
            true => self.state = DiscreteNeuronState::Active,
            false => self.state = DiscreteNeuronState::Inactive,
        }
    }

    /// Translates the state of the neuron to an `isize`, either `1` if 
    /// [`DiscreteNeuronState::Active`] or `-1` if [`DiscreteNeuronState::Inactive`]
    pub fn state_to_numeric(&self) -> isize {
        match &self.state {
            DiscreteNeuronState::Active => 1,
            DiscreteNeuronState::Inactive => -1,
        }
    }

    /// Translates the state of the neuron to an `bool`, either `true` if 
    /// [`DiscreteNeuronState::Active`] or `false` if [`DiscreteNeuronState::Inactive`]
    pub fn state_to_bool(&self) -> bool {
        match &self.state {
            DiscreteNeuronState::Active => true,
            DiscreteNeuronState::Inactive => false,
        }
    }
}

/// Simple lattice of bipolar discrete neurons with a weight matrix
/// 
/// Example discrete bipolar autoassociatve network execution:
/// ```rust
/// # use spiking_neural_networks::{
/// #     neuron::attractors::{
/// #         DiscreteNeuronLattice, generate_hopfield_network, generate_random_patterns, distort_pattern
/// #     },
/// #     graph::AdjacencyMatrix,
/// #     error::SpikingNeuralNetworksError,
/// # };
/// 
/// fn main() -> Result<(), SpikingNeuralNetworksError> {
///     type GraphType = AdjacencyMatrix<(usize, usize), f32>;
/// 
///     let iterations = 10;
///     let num_patterns = 3;
///     let (num_rows, num_cols) = (10, 10);
///     let noise_generation = 0.5;
///     let noise_level = 0.2;
///     // initalize patterns
///     let patterns = generate_random_patterns(num_rows, num_cols, num_patterns, noise_generation);
///     
///     // create network weights
///     let weights = generate_hopfield_network::<GraphType>(0, &patterns)?;
///     let mut discrete_lattice = DiscreteNeuronLattice::<GraphType>::generate_lattice_from_dimension(
///         num_rows, 
///         num_cols,
///     );
///     discrete_lattice.graph = weights;
///     
///     for (n, pattern) in patterns.iter().enumerate() {
///         let distorted_pattern = distort_pattern(&pattern, noise_level);
///     
///         let mut hopfield_history: Vec<Vec<Vec<bool>>> = Vec::new();
///     
///         // setup distorted pattern in lattice
///         discrete_lattice.input_pattern_into_discrete_grid(distorted_pattern);
///         hopfield_history.push(discrete_lattice.convert_to_bools());
///     
///         // execute lattice
///         for _ in 0..iterations {
///             discrete_lattice.iterate()?;
///             hopfield_history.push(discrete_lattice.convert_to_bools());
///         }
///     
///         // check if pattern matches now original
///         assert!(hopfield_history.last().unwrap() == pattern);
///     }
/// 
///     Ok(())
/// }
/// ```
pub struct DiscreteNeuronLattice<T: Graph<K=(usize, usize), V=f32>>{
    /// 2 dimensional grid of discrete neurons
    pub cell_grid: Vec<Vec<DiscreteNeuron>>,
    /// Internal graph weights, position listed in graph must have a corresponding index
    /// in the `cell_grid`, (for example position (0, 1) in graph corresponds to `cell_grid[0][1]`)
    pub graph: T,
}

impl<T: Graph<K=(usize, usize), V=f32>> Default for DiscreteNeuronLattice<T> {
    fn default() -> Self {
        DiscreteNeuronLattice {
            cell_grid: vec![],
            graph: T::default(),
        }
    }
}

impl<T: Graph<K=(usize, usize), V=f32>> DiscreteNeuronLattice<T> {
    /// Generates a lattice with default weights given a number of rows and columns to use
    pub fn generate_lattice_from_dimension(num_rows: usize, num_cols: usize) -> Self {
        let cell_grid: Vec<Vec<DiscreteNeuron>> = (0..num_rows)
            .map(|_| {
                (0..num_cols)
                    .map(|_| {
                        DiscreteNeuron::default()
                    })
                    .collect::<Vec<DiscreteNeuron>>()
            })
            .collect::<Vec<Vec<DiscreteNeuron>>>();

        DiscreteNeuronLattice {
            cell_grid,
            graph: T::default(),
        }
    }

    /// Sets state of given grid of discrete neurons to the given, if value
    /// in pattern is greater than 0 the corressponding state is set to [`DiscreteNeuronState::Active`],
    /// otherwise it is set to [`DiscreteNeuronState::Inactive`]
    pub fn input_pattern_into_discrete_grid(&mut self, pattern: Vec<Vec<bool>>) {
        for (i, pattern_vec) in pattern.iter().enumerate() {
            for (j, value) in pattern_vec.iter().enumerate() {
                if *value {
                    self.cell_grid[i][j].update(1.);
                } else {
                    self.cell_grid[i][j].update(-1.);
                }
            }
        }
    }

    /// Converts the given network of discrete neurons to a grid of `isize` values
    pub fn convert_to_numerics(&self) -> Vec<Vec<isize>> {
        let mut output: Vec<Vec<isize>> = Vec::new();

        for i in self.cell_grid.iter() {
            let mut output_vec: Vec<isize> = Vec::new();
            for j in i.iter() {
                output_vec.push(j.state_to_numeric());
            }

            output.push(output_vec);
        }

        output
    }

    /// Converts the given network of discrete neurons to a grid of `bool` values
    pub fn convert_to_bools(&self) -> Vec<Vec<bool>> {
        let mut output: Vec<Vec<bool>> = Vec::new();

        for i in self.cell_grid.iter() {
            let mut output_vec: Vec<bool> = Vec::new();
            for j in i.iter() {
                output_vec.push(j.state_to_bool());
            }

            output.push(output_vec);
        }

        output
    }

    /// Iterates the discrete network of neurons based on the weights between neurons
    pub fn iterate(&mut self) -> result::Result<(), GraphError> {
        for current_pos in self.graph.get_every_node() {
            let input_positions = self.graph.get_incoming_connections(&current_pos)?;

            let input_value: f32 = input_positions.iter()
                .map(|graph_pos| {
                        let (pos_i, pos_j) = graph_pos;

                        self.graph.lookup_weight(graph_pos, &current_pos).unwrap().unwrap() 
                        * self.cell_grid[*pos_i][*pos_j].state_to_numeric() as f32
                    }
                )
                .sum();

            self.cell_grid[current_pos.0][current_pos.1].update(input_value);
        }

        Ok(())
    }
}

fn outer_product(a: &Vec<isize>, b: &Vec<isize>) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in a {
        let mut vector: Vec<isize> = Vec::new();
        for j in b {
            vector.push(i * j);
        }

        output.push(vector);
    }

    output
}

fn first_dimensional_index_to_position(i: usize, num_cols: usize) -> (usize, usize) {
    ((i / num_cols), (i % num_cols))
}

/// Generates weights for a Hopfield network based on a given set of patterns, and 
/// an id to assign to the graph, assumes the patterns have the same dimensions throughout,
/// also assumes the pattern is completely bipolar (either `-1` or `1`)
pub fn generate_hopfield_network<T: Graph<K=(usize, usize), V=f32> + Default>(
    graph_id: usize,
    data: &Vec<Vec<Vec<bool>>>,
) -> result::Result<T, SpikingNeuralNetworksError> {
    let num_rows = data.first().unwrap_or(&vec![]).len();
    let num_cols = data.first().unwrap_or(&vec![]).first()
        .unwrap_or(&vec![])
        .len();

    for pattern in data {
        if pattern.len() != num_rows {
            return Err(
                SpikingNeuralNetworksError::from(PatternError::PatternDimensionsAreNotEqual)
            );
        }
    
        if pattern.iter().any(|row| row.len() != num_cols) {
            return Err(
                SpikingNeuralNetworksError::from(PatternError::PatternDimensionsAreNotEqual)
            );
        }
    }

    let mut weights = T::default();
    weights.set_id(graph_id);

    for i in 0..num_rows {
        for j in 0..num_cols {
            weights.add_node((i, j));
        }
    }

    for pattern in data {
        let flattened_pattern: Vec<isize> = pattern.iter()
            .flat_map(|v| v.iter().map(|i| {
                if *i { 1 } else { -1 }
            }))
            .collect();

        let weight_changes = outer_product(&flattened_pattern, &flattened_pattern);

        for (i, weight_vec) in weight_changes.iter().enumerate() {
            for (j, value) in weight_vec.iter().enumerate() {
                let coming = first_dimensional_index_to_position(i, num_cols);
                let going = first_dimensional_index_to_position(j, num_cols);

                //   1 2 3 ...
                // 1 . . .
                // 2 . . .
                // 3 . . .
                // ...
                
                //       (0, 0) (0, 1) (0, 2) ...
                // (0, 0)   .      .      .
                // (0, 1)   .      .      .
                // (0, 2)   .      .      .
                // ...

                if coming == going {
                    weights.edit_weight(&coming, &going, None)?;
                    continue;
                }

                let current_weight = (weights.lookup_weight(&coming, &going)?).unwrap_or(0.);

                weights.edit_weight(&coming, &going, Some(current_weight + *value as f32))?;
            }
        }  
    }

    Ok(weights)
}

fn binary_pattern_calculation(flattened_pattern: &Vec<isize>, a: f32, b: f32, scalar: f32) -> Vec<Vec<f32>> {
    let mut output: Vec<Vec<f32>> = Vec::new();

    for i in flattened_pattern {
        let mut vector: Vec<f32> = Vec::new();
        for j in flattened_pattern {
            vector.push((*i as f32 - b) * (*j as f32 - a) * scalar);
        }

        output.push(vector);
    }

    output
}

/// Generates weights for a Hopfield network based on a given set of patterns, and 
/// an id to assign to the graph, assumes the patterns have the same dimensions throughout,
/// also assumes the pattern is completely binary (either `0` or `1`)
pub fn generate_binary_hopfield_network<T: Graph<K=(usize, usize), V=f32> + Default>(
    graph_id: usize,
    data: &Vec<Vec<Vec<bool>>>,
    a: f32,
    b: f32,
    scalar: f32,
) -> result::Result<T, SpikingNeuralNetworksError> {
    let num_rows = data.first().unwrap_or(&vec![]).len();
    let num_cols = data.first().unwrap_or(&vec![]).first().unwrap_or(&vec![])
        .len();

    for pattern in data {
        if pattern.len() != num_rows {
            return Err(
                SpikingNeuralNetworksError::from(PatternError::PatternDimensionsAreNotEqual)
            );
        }
    
        if pattern.iter().any(|row| row.len() != num_cols) {
            return Err(
                SpikingNeuralNetworksError::from(PatternError::PatternDimensionsAreNotEqual)
            );
        }
    }

    let mut weights = T::default();
    weights.set_id(graph_id);

    for i in 0..num_rows {
        for j in 0..num_cols {
            weights.add_node((i, j));
        }
    }

    for pattern in data {
        let flattened_pattern: Vec<isize> = pattern.iter()
            .flat_map(|v| v.iter().map(|i| {
                if *i { 1 } else { 0 }
            }))
            .collect();
        let weight_changes = binary_pattern_calculation(
            &flattened_pattern,
            a,
            b,
            scalar,
        );

        for (i, weight_vec) in weight_changes.iter().enumerate() {
            for (j, value) in weight_vec.iter().enumerate() {
                let coming = first_dimensional_index_to_position(i, num_cols);
                let going = first_dimensional_index_to_position(j, num_cols);

                //   1 2 3 ...
                // 1 . . .
                // 2 . . .
                // 3 . . .
                // ...
                
                //       (0, 0) (0, 1) (0, 2) ...
                // (0, 0)   .      .      .
                // (0, 1)   .      .      .
                // (0, 2)   .      .      .
                // ...

                if coming == going {
                    weights.edit_weight(&coming, &going, None)?;
                    continue;
                }

                let current_weight = (weights.lookup_weight(&coming, &going)?).unwrap_or(0.);

                weights.edit_weight(&coming, &going, Some(current_weight + *value))?;
            }
        }  
    }

    Ok(weights)
}

/// Adds random noise to a given pattern based on a given `noise_level` between `0.` and `1.`
pub fn distort_pattern(pattern: &[Vec<bool>], noise_level: f32) -> Vec<Vec<bool>> {
    let mut output: Vec<Vec<bool>> = Vec::new();

    for i in pattern.iter() {
        let mut output_vec: Vec<bool> = Vec::new();
        for j in i.iter() {
            if rand::thread_rng().gen_range(0.0..=1.0) <= noise_level {
                if *j {
                    output_vec.push(false);
                } else {
                    output_vec.push(true);
                }
            } else {
                output_vec.push(*j)
            }
        }

        output.push(output_vec);
    }

    output
}

/// Generates a set of random patterns based on a given size, number of patterns
/// and degree of noise to use when generating the pattern
pub fn generate_random_patterns(
    num_rows: usize, 
    num_cols: usize, 
    num_patterns: usize, 
    p_zero: f32,
) -> Vec<Vec<Vec<bool>>> {
    let binomial = Binomial::new(1, p_zero.into()).expect("Could not create binomial distribution");
    let mut rng = rand::thread_rng();

    (0..num_patterns)
        .map(|_| {
            let current_pattern: Vec<bool> = binomial.sample_iter(&mut rng).take(num_rows * num_cols).map(|i| {
                let x = i as isize;
                x == 1
            }).collect();

            current_pattern.chunks(num_cols)
                .map(|chunk| chunk.to_vec())
                .collect()
        })
        .collect::<Vec<Vec<Vec<bool>>>>()
}
