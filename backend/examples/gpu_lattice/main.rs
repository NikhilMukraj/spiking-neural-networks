use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        integrate_and_fire::SimpleLeakyIntegrateAndFire,
        gpu_lattices::LatticeGPU,
        Lattice, RunLattice
    }
};


fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
    x != y
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = SimpleLeakyIntegrateAndFire {
        gap_conductance: 0.1,
        ..SimpleLeakyIntegrateAndFire::default_impl()
    };

    let iterations = 1000;
    let (num_rows, num_cols) = (2, 2);
   
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

    let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;

    lattice.run_lattice(iterations)?;

    gpu_lattice.run_lattice(iterations)?;

    for (row1, row2) in lattice.cell_grid().iter().zip(gpu_lattice.cell_grid().iter()) {
        for (neuron1, neuron2) in row1.iter().zip(row2.iter()) {
            let error = (neuron1.current_voltage - neuron2.current_voltage).abs();
            assert!(
                error <= 2., "error: {}, neuron1: {}, neuron2: {}\n{:#?}\n{:#?}", 
                error,
                neuron1.current_voltage,
                neuron2.current_voltage,
                lattice.cell_grid().iter()
                    .map(|i| i.iter().map(|j| j.current_voltage).collect::<Vec<f32>>())
                    .collect::<Vec<Vec<f32>>>(),
                gpu_lattice.cell_grid().iter()
                    .map(|i| i.iter().map(|j| j.current_voltage).collect::<Vec<f32>>())
                    .collect::<Vec<Vec<f32>>>(),
            );

            let error = (
                neuron1.last_firing_time.unwrap_or(0) as isize - 
                neuron2.last_firing_time.unwrap_or(0) as isize
            ).abs();
            assert!(
                error <= 2, "error: {:#?}, neuron1: {:#?}, neuron2: {:#?}",
                error,
                neuron1.last_firing_time,
                neuron2.last_firing_time,
            );
        }
    }

    for (cpu_cell_grid, gpu_cell_grid) in lattice.grid_history.history.iter()
        .zip(gpu_lattice.grid_history.history.iter()) {
        for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
            for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                let error = (voltage1 - voltage2).abs();
                assert!(
                    error <= 2., "error: {}, voltage1: {}, voltage2: {}", 
                    error,
                    voltage1,
                    voltage2,
                );
            }
        }
    }

    const GREEN: &str = "\x1b[32m";
    const RESET: &str = "\x1b[0m";
    
    println!("{}GPU test passed{}", GREEN, RESET);
    
    Ok(())
}
