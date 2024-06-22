use std::{
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use spiking_neural_networks::neuron::{
    iterate_and_spike::{
        AMPADefault, IterateAndSpike, LigandGatedChannel, NeurotransmitterType, 
        ReceptorKinetics, DestexheNeurotransmitter,
    }, 
    hodgkin_huxley::HodgkinHuxleyNeuron, 
    iterate_coupled_spiking_neurons
};


// Couples two Hodgkin Huxley neurons with neurotransmission and tracks relevant 
// history regarding voltages, neurotransmitter values, and receptor values
// which are written to a .csv file at the working directory
fn main() {
    let mut presynaptic_neuron = HodgkinHuxleyNeuron::default_impl();
    presynaptic_neuron.ligand_gates
        .insert(NeurotransmitterType::AMPA, LigandGatedChannel::ampa_default());
    presynaptic_neuron.synaptic_neurotransmitters
        .insert(NeurotransmitterType::AMPA, DestexheNeurotransmitter::ampa_default());

    let mut postsynaptic_neuron = presynaptic_neuron.clone();

    let iterations = 10000;
    let input_current = 50.;
    let do_receptor_kinetics = true;
    let gaussian = false;

    let mut postsynaptic_voltages: Vec<f64> = Vec::new();
    let mut presynaptic_neurotransmitter_concs: Vec<f64> = Vec::new();
    let mut receptor_values: Vec<f64> = Vec::new();

    for _ in 0..iterations {
        iterate_coupled_spiking_neurons(
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron, 
            input_current, 
            do_receptor_kinetics, 
            gaussian
        );

        postsynaptic_voltages.push(postsynaptic_neuron.current_voltage);
        presynaptic_neurotransmitter_concs.push(
            *presynaptic_neuron.get_neurotransmitter_concentrations().get(&NeurotransmitterType::AMPA)
                .expect("Could not find concentration")
        );
        receptor_values.push(
            postsynaptic_neuron.ligand_gates.get(&NeurotransmitterType::AMPA)
                .expect("Could not find ligand gate")
                .receptor
                .get_r()
        );
    }

    let mut file = BufWriter::new(File::create("coupled_hodgkin_huxley.csv")
        .expect("Could not create file"));

    writeln!(file, "voltages,Ts,rs").expect("Could not write to file");
    for i in 0..iterations {
        writeln!(
            file, 
            "{},{},{}", 
            postsynaptic_voltages[i], 
            presynaptic_neurotransmitter_concs[i],
            receptor_values[i],
        ).expect("Could not write to file");
    }
}