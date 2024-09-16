use std::{fs::File, io::Write};
use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        integrate_and_fire::QuadraticIntegrateAndFireNeuron,
        gpu_lattices::LatticeGPU,
        Lattice,
    }
};


fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
    x != y
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = QuadraticIntegrateAndFireNeuron {
        gap_conductance: 10.,
        ..QuadraticIntegrateAndFireNeuron::default_impl()
    };

    // let iterations = 5000;
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

    let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone());

    lattice.run_lattice(1)?;

    gpu_lattice.run_lattice(1);

    let mut cpu_file = File::create("cpu_lattice.txt").expect("Could not create file");
    let mut gpu_file = File::create("gpu_lattice.txt").expect("Could not create file");

    for row in lattice.cell_grid.iter() {
        for neuron in row {
            write!(cpu_file, "{}\t", neuron.current_voltage).expect("Could not write to file");
        }
        writeln!(cpu_file).expect("Could not write to file");
    }

    for row in gpu_lattice.cell_grid.iter() {
        for neuron in row {
            write!(gpu_file, "{}\t", neuron.current_voltage).expect("Could not write to file");
        }
        writeln!(gpu_file).expect("Could not write to file");
    }

    
    Ok(())
}
