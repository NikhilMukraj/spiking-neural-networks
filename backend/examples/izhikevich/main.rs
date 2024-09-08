use std::{
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use spiking_neural_networks::neuron::{
    integrate_and_fire::IzhikevichNeuron, 
    iterate_coupled_spiking_neurons
};


// Couples two Izhikevich neurons and tracks the voltage of the two neurons
// over time, .csv containing the voltages is written to a file in the working 
// directory when the simulation is finished
fn main() {
    let mut presynaptic_neuron = IzhikevichNeuron {
        gap_conductance: 10.,
        ..IzhikevichNeuron::default_impl()
    };
   
    let mut postsynaptic_neuron = presynaptic_neuron.clone();

    let iterations = 10000;
    let input_current = 40.;
    let electrical_synapse = true;
    let chemical_synapse = false;

    let mut presynaptic_voltages: Vec<f32> = Vec::new();
    let mut postsynaptic_voltages: Vec<f32> = Vec::new();

    for _ in 0..iterations {
        iterate_coupled_spiking_neurons(
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron, 
            input_current, 
            electrical_synapse,
            chemical_synapse, 
            None,
        );

        presynaptic_voltages.push(presynaptic_neuron.current_voltage);
        postsynaptic_voltages.push(postsynaptic_neuron.current_voltage);
    }

    let mut file = BufWriter::new(File::create("coupled_izhikevich.csv")
        .expect("Could not create file"));

    writeln!(file, "presynaptic_voltages,postsynaptic_voltages").expect("Could not write to file");
    for i in 0..iterations {
        writeln!(
            file, 
            "{},{}", 
            presynaptic_voltages[i], 
            postsynaptic_voltages[i],
        ).expect("Could not write to file");
    }
}