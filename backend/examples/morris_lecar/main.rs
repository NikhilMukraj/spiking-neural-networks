use std::{
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use crate::spiking_neural_networks::neuron::{
    iterate_and_spike::IterateAndSpike,
    morris_lecar::MorrisLecarNeuron,
};


// Inputs a static current into the neuron model and tracks the voltage 
// over time, writes the output file to the current working directory
fn main() {
    let mut morris_lecar_neuron = MorrisLecarNeuron::default_impl();

    let iterations = 10000;
    let input = 100.;

    let mut voltages: Vec<f32> = vec![];

    for _ in 0..iterations {
        morris_lecar_neuron.iterate_and_spike(input);

        voltages.push(morris_lecar_neuron.current_voltage);
    }

    let mut file = BufWriter::new(File::create("morris_lecar_static_input.csv")
        .expect("Could not create file"));

    writeln!(file, "voltages").expect("Could not write to file");
    for i in voltages {
        writeln!(file, "{}", i).expect("Could not write to file");
    }
}