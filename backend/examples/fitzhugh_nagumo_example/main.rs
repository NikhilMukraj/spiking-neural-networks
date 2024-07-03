use std::{
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use crate::spiking_neural_networks::neuron::{
    iterate_and_spike::IterateAndSpike,
    fitzhugh_nagumo::FitzHughNagumoNeuron,
};


// Inputs a static current into the neuron model and tracks the voltage 
// and adaptive value over time, writes the output file to the current
// working directory
fn main() {
    let mut fitzhugh_nagumo_neuron = FitzHughNagumoNeuron::default_impl();

    let iterations = 5000;
    let input = 0.5;

    let mut voltages: Vec<f32> = vec![];
    let mut ws: Vec<f32> = vec![];

    for _ in 0..iterations {
        fitzhugh_nagumo_neuron.iterate_and_spike(input);

        voltages.push(fitzhugh_nagumo_neuron.current_voltage);
        ws.push(fitzhugh_nagumo_neuron.w);
    }

    let mut file = BufWriter::new(File::create("fitzhugh_nagumo_static_input.csv")
        .expect("Could not create file"));

    writeln!(file, "voltages,ws").expect("Could not write to file");
    for i in 0..iterations {
        writeln!(file, "{},{}", voltages[i], ws[i]).expect("Could not write to file");
    }
}