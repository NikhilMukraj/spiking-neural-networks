use std::{
    fs::File,
    io::{BufWriter, Write, Result}
};
use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::neuron::{
    Lattice,
    integrate_and_fire::IzhikevichNeuron,
};


/// Runs an example lattice with randomly connected neurons for 500 ms,
/// writes the history of the grid to a file in working directory when finished
fn main() -> Result<()> {
    let base_neuron = IzhikevichNeuron {
        gap_conductance: 10.,
        ..IzhikevichNeuron::default_impl()
    };

    let iterations = 5000;
    let (num_rows, num_cols, radius) = (10, 10, 2);
   
    // infers type based on base neuron and default implementation
    let mut lattice: Lattice<_, _, _> = Lattice::default_impl();
    
    lattice.populate_and_randomly_connect_lattice(
        &base_neuron, 
        num_rows, 
        num_cols, 
        radius, 
        &None
    );

    let mut rng = rand::thread_rng();

    for row in lattice.cell_grid.iter_mut() {
        for neuron in row {
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        }
    }

    lattice.update_grid_history = true;

    // iterates internal electrical connections
    lattice.run_lattice_electrical_only(iterations)?;

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
