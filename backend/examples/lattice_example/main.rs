use std::{
    fs::File,
    result::Result,
    io::{BufWriter, Write}
};
use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        Lattice,
        integrate_and_fire::IzhikevichNeuron,
    }
};


// connect neurons within a radius of 2 with an 80% chance of connection
fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    (((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)) as f64).sqrt() <= 2. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.8
}

/// Runs an example lattice with randomly connected neurons for 500 ms,
/// writes the history of the grid to a file in working directory when finished
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = IzhikevichNeuron {
        gap_conductance: 10.,
        ..IzhikevichNeuron::default_impl()
    };

    let iterations = 5000;
    let (num_rows, num_cols) = (10, 10);
   
    // infers type based on base neuron and default implementation
    let mut lattice = Lattice::default_impl();
    
    lattice.populate(
        &base_neuron, 
        num_rows, 
        num_cols, 
    );

    lattice.connect(&connection_conditional, None);

    lattice.apply(|neuron: &mut _| {
        let mut rng = rand::thread_rng();
        neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
    });

    lattice.update_grid_history = true;

    // iterates internal electrical connections
    lattice.run_lattice(iterations)?;

    let mut voltage_file = BufWriter::new(File::create("lattice_history.txt")
        .expect("Could not create file"));

    // writes history to file
    for grid in lattice.grid_history.history {
        for row in grid {
            for value in row {
                write!(voltage_file, "{} ", value)
                    .expect("Could not write to file");
            }
            writeln!(voltage_file)
                .expect("Could not write to file");
        }
        writeln!(voltage_file, "-----")
            .expect("Could not write to file"); 
    }

    Ok(())
}
