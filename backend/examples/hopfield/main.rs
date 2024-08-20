use std::{
    result,
    env::{args, current_dir},
    fs::{read_to_string, File}, 
    io::{BufWriter, Error, ErrorKind, Result, Write}
};
extern crate spiking_neural_networks;
use crate::spiking_neural_networks::{
    neuron::attractors::{
        DiscreteNeuronLattice, generate_hopfield_network, distort_pattern,
    },
    graph::{Graph, AdjacencyMatrix},
    error::SpikingNeuralNetworksError,
};


fn read_pattern(file_contents: &str) -> Result<Vec<Vec<bool>>> {
    let mut matrix = Vec::new();

    for line in file_contents.split("\n") {
        let row: Vec<isize> = line.split(',')
            .map(|s| s.trim().parse().expect("Could not parse pattern"))
            .collect();

        if row.iter().any(|i| *i != -1 && *i != 1) {
            return Err(Error::new(ErrorKind::InvalidData, "Pattern must be bipolar (-1 or 1)"))
        }

        matrix.push(row.iter().map(|i| if *i == 1 { true } else { false }).collect::<Vec<bool>>());
    }

    Ok(matrix)
}

fn test_hopfield_network<T: Graph<K=(usize, usize), V=f32>>(
    patterns: &Vec<Vec<Vec<bool>>>,
    noise_level: f32,
    iterations: usize,
) -> result::Result<(), SpikingNeuralNetworksError> {
    let num_rows = patterns[0].len();
    let num_cols = patterns[0][0].len();

    let weights = generate_hopfield_network::<T>(0, &patterns)?;
    let mut discrete_lattice = DiscreteNeuronLattice::<T>::generate_lattice_from_dimension(num_rows, num_cols);
    discrete_lattice.graph = weights;

    for (n, pattern) in patterns.iter().enumerate() {
        let distorted_pattern = distort_pattern(&pattern, noise_level);

        let mut hopfield_history: Vec<Vec<Vec<isize>>> = Vec::new();

        discrete_lattice.input_pattern_into_discrete_grid(distorted_pattern);
        hopfield_history.push(discrete_lattice.convert_to_numerics());

        for _ in 0..iterations {
            discrete_lattice.iterate()?;
            hopfield_history.push(discrete_lattice.convert_to_numerics());
        }

        let mut hopfield_file = BufWriter::new(File::create(format!("{}_hopfield.txt", n + 1))
            .expect("Could not create file"));

        for grid in hopfield_history {
            for row in grid {
                for value in row {
                    write!(hopfield_file, "{} ", value)
                        .expect("Could not write to file");
                }
                writeln!(hopfield_file)
                    .expect("Could not write to file");
            }
            writeln!(hopfield_file, "-----")
                .expect("Could not write to file"); 
        }
    }

    Ok(())
}

// - Creates a hopfield network from discrete neurons and given patterns
// - Distorts patterns, and inputs pattern into lattice
// then iterates lattice for set amount of iterations to converge 
// - Writes the history of the lattice over time from input to finish

// - List relative to patterns to read, patterns must be bipolar and the same dimensions
// - Bipolar patterns only have values -1 and 1 seperated by spaces and new lines
// - (assumes user is in examples folder and working directory is examples folder) 
// - (outputs will be written to working directory)
// - cargo run --example hopfield examples/hopfield_example/pattern1.txt examples/hopfield_example/pattern2.txt
fn main() -> Result<()> {
    let pattern_files: Vec<String> = args().skip(1).collect();

    if pattern_files.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "Example requires input pattern text files"));
    }

    let current_directory = current_dir()?
        .parent()
        .ok_or_else(|| Error::new(ErrorKind::Other, "Could not determine working directory"))?
        .to_path_buf();

    let pattern_files: Vec<String> = pattern_files.iter()
        .map(|i| {
            let abs_path = current_directory.join(i);
            abs_path.to_string_lossy().into_owned()
        })
        .collect();
    let patterns: Vec<Vec<Vec<bool>>> = pattern_files.iter()
        .map(|i| 
            read_pattern(
                &(read_to_string(i).expect(&format!("Could not read file: {}", i)))
            ).expect("Pattern could not be read, must be bipolar (-1 or 1 for every value)")
        )
        .collect();

    let num_rows = patterns[0].len();
    let num_cols = patterns[0][0].len();

    for pattern in patterns.iter() {
        if pattern.len() != num_rows {
            return Err(Error::new(ErrorKind::InvalidInput, "Patterns must have the same size"));
        }

        if pattern.iter().any(|row| row.len() != num_cols) {
            return Err(Error::new(ErrorKind::InvalidInput, "Patterns must have the same size"));
        }
    }

    let iterations = 10;
    let noise_level = 0.25;

    test_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(&patterns, noise_level, iterations).expect("Error in graph");

    Ok(())
}