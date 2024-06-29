//! An set of tools to generate weights for attractors like a Hopfield network
//! as well as a simplified neuron model for very basic testing of attractor dynamics.

use std::result;
use rand::Rng;
use crate::error::{
    GraphError, PatternErrorKind, SpikingNeuralNetworksError, SpikingNeuralNetworksErrorKind
};
use crate::graph::{Graph, GraphPosition};


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
    pub fn update(&mut self, input: f64) {
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
}

/// Simple lattice of bipolar discrete neurons with a weight matrix
pub struct DiscreteNeuronLattice<T: Graph>{
    /// 2 dimensional grid of discrete neurons
    pub cell_grid: Vec<Vec<DiscreteNeuron>>,
    /// Internal graph weights, position listed in graph must have a corresponding index
    /// in the `cell_grid`, (for example position (0, 1) in graph corresponds to `cell_grid[0][1]`)
    pub graph: T,
}

impl<T: Graph> Default for DiscreteNeuronLattice<T> {
    fn default() -> Self {
        DiscreteNeuronLattice {
            cell_grid: vec![],
            graph: T::default(),
        }
    }
}

impl<T: Graph> DiscreteNeuronLattice<T> {
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
            cell_grid: cell_grid,
            graph: T::default(),
        }
    }

    /// Sets state of given grid of discrete neurons to the given, if value
    /// in pattern is greater than 0 the corressponding state is set to [`DiscreteNeuronState::Active`],
    /// otherwise it is set to [`DiscreteNeuronState::Inactive`]
    pub fn input_pattern_into_discrete_grid(&mut self, pattern: Vec<Vec<isize>>) {
        for (i, pattern_vec) in pattern.iter().enumerate() {
            for (j, value) in pattern_vec.iter().enumerate() {
                self.cell_grid[i][j].update(*value as f64);
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

    /// Iterates the discrete network of neurons based on the weights between neurons
    pub fn iterate(&mut self) -> result::Result<(), GraphError> {
        for current_pos in self.graph.get_every_node() {
            let input_positions = self.graph.get_incoming_connections(&current_pos)?;

            let input_value: f64 = input_positions.iter()
                .map(|graph_pos| {
                        let (pos_i, pos_j) = graph_pos.pos;

                        self.graph.lookup_weight(&graph_pos, &current_pos).unwrap().unwrap() 
                        * self.cell_grid[pos_i][pos_j].state_to_numeric() as f64
                    }
                )
                .sum();

            self.cell_grid[current_pos.pos.0][current_pos.pos.1].update(input_value);
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
pub fn generate_hopfield_network<T: Graph + Default>(
    graph_id: usize,
    data: &Vec<Vec<Vec<isize>>>
) -> result::Result<T, SpikingNeuralNetworksError> {
    let num_rows = data[0].len();
    let num_cols = data[0][0].len();

    for pattern in data {
        for row in pattern {
            if row.iter().any(|i| *i != -1 && *i != 1) {
                return Err(
                    SpikingNeuralNetworksError::new(
                        SpikingNeuralNetworksErrorKind::PatternRelatedError(
                            PatternErrorKind::PatternIsNotBipolar
                        ), file!(), line!()
                    )
                )
            }
        }

        if pattern.len() != num_rows {
            return Err(
                SpikingNeuralNetworksError::new(
                    SpikingNeuralNetworksErrorKind::PatternRelatedError(
                        PatternErrorKind::PatternDimensionsAreNotEqual
                    ), file!(), line!()
                )
            );
        }
    
        if pattern.iter().any(|row| row.len() != num_cols) {
            return Err(
                SpikingNeuralNetworksError::new(
                    SpikingNeuralNetworksErrorKind::PatternRelatedError(
                        PatternErrorKind::PatternDimensionsAreNotEqual
                    ), file!(), line!()
                )
            );
        }
    }

    let mut weights = T::default();
    weights.set_id(graph_id);

    for i in 0..num_rows {
        for j in 0..num_cols {
            weights.add_node(GraphPosition { id: graph_id, pos: (i, j)});
        }
    }

    for pattern in data {
        let flattened_pattern: Vec<isize> = pattern.iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        let weight_changes = outer_product(&flattened_pattern, &flattened_pattern);

        for (i, weight_vec) in weight_changes.iter().enumerate() {
            for (j, value) in weight_vec.iter().enumerate() {
                let coming = first_dimensional_index_to_position(i, num_cols);
                let going = first_dimensional_index_to_position(j, num_cols);

                let coming = GraphPosition { id: graph_id, pos: coming };
                let going = GraphPosition { id: graph_id, pos: going };

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

                let current_weight = match weights.lookup_weight(&coming, &going)? {
                    Some(w) => w,
                    None => 0.
                };

                weights.edit_weight(&coming, &going, Some(current_weight + *value as f64))?;
            }
        }  
    }

    Ok(weights)
}

/// Adds random noise to a given pattern based on a given `noise_level` between `0.` and `1.`
pub fn distort_pattern(pattern: &Vec<Vec<isize>>, noise_level: f64) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in pattern.iter() {
        let mut output_vec: Vec<isize> = Vec::new();
        for j in i.iter() {
            if rand::thread_rng().gen_range(0.0..=1.0) <= noise_level {
                if *j > 0 {
                    output_vec.push(-1);
                } else {
                    output_vec.push(1);
                }
            } else {
                output_vec.push(*j)
            }
        }

        output.push(output_vec);
    }

    output
}

// could try random turing patterns as well
/// Generates a random pattern based on a given size, number of patterns
/// and degree of noise to use when generating the pattern
pub fn generate_random_patterns(
    num_rows: usize, 
    num_cols: usize, 
    num_patterns: usize, 
    noise_level: f64
) -> Vec<Vec<Vec<isize>>> {
    let base_pattern = (0..num_rows).map(|_| {
        (0..num_cols)
            .map(|_| {
                -1
            })
            .collect::<Vec<isize>>()
    })
    .collect::<Vec<Vec<isize>>>();

    (0..num_patterns).map(|_| {
        distort_pattern(&base_pattern, noise_level)
    })
    .collect()
}
