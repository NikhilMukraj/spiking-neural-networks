use std::io::Result;
use rand::Rng;
use crate::graph;
use graph::GraphFunctionality;


pub enum DiscreteNeuronState {
    Active,
    Inactive,
}

pub struct DiscreteNeuron {
    pub state: DiscreteNeuronState
}

impl Default for DiscreteNeuron {
    fn default() -> Self {
        DiscreteNeuron { state: DiscreteNeuronState::Inactive }
    }
}

impl DiscreteNeuron {
    fn update(&mut self, input: f64) {
        match input > 0. {
            true => self.state = DiscreteNeuronState::Active,
            false => self.state = DiscreteNeuronState::Inactive,
        }
    }

    fn state_to_numeric(&self) -> f64 {
        match &self.state {
            DiscreteNeuronState::Active => 1.,
            DiscreteNeuronState::Inactive => -1.,
        }
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

pub fn generate_hopfield_network<T: GraphFunctionality + Default>(
    num_rows: usize, 
    num_cols: usize, 
    data: &Vec<Vec<Vec<isize>>>
) -> Result<T> {
    let mut weights = T::default();

    for i in 0..num_rows {
        for j in 0..num_cols {
            weights.add_vertex((i, j));
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

pub fn input_pattern_into_grid(cell_grid: &mut Vec<Vec<DiscreteNeuron>>, pattern: Vec<Vec<isize>>) {
    for (i, pattern_vec) in pattern.iter().enumerate() {
        for (j, value) in pattern_vec.iter().enumerate() {
            cell_grid[i][j].update(*value as f64);
        }
    }
}

pub fn iterate_hopfield_network<T: GraphFunctionality>(
    cell_grid: &mut Vec<Vec<DiscreteNeuron>>, 
    weights: &T, 
) -> Result<()> {
    for i in 0..cell_grid.len() {
        for j in 0..cell_grid[0].len() {
            let input_positions = weights.get_incoming_connections(&(i, j))?;

            // if there is problem with convergence it is likely this calculation
            let input_value: f64 = input_positions.iter()
                .map(|(pos_i, pos_j)| 
                    weights.lookup_weight(&(*pos_i, *pos_j), &(i, j)).unwrap().unwrap() 
                    * cell_grid[*pos_i][*pos_j].state_to_numeric()
                )
                .sum();

            cell_grid[i][j].update(input_value);
        }
    }

    Ok(())
}

pub fn convert_hopfield_network(cell_grid: &Vec<Vec<DiscreteNeuron>>) -> Vec<Vec<isize>> {
    let mut output: Vec<Vec<isize>> = Vec::new();

    for i in cell_grid.iter() {
        let mut output_vec: Vec<isize> = Vec::new();
        for j in i.iter() {
            output_vec.push(j.state_to_numeric() as isize);
        }

        output.push(output_vec);
    }

    output
}

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
// pub fn generate_random_patterns(
//     num_rows: usize, 
//     num_cols: usize, 
//     num_patterns: usize, 
//     noise_level: f64
// ) -> Vec<Vec<Vec<isize>>> {
//     let base_pattern = (0..num_rows).map(|_| {
//         (0..num_cols)
//             .map(|_| {
//                 -1
//             })
//             .collect::<Vec<isize>>()
//     })
//     .collect::<Vec<Vec<isize>>>();

//     (0..num_patterns).map(|_| {
//         distort_pattern(&base_pattern, noise_level)
//     })
//     .collect()
// }
