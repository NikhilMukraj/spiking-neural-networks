use std::{
    fs::File,
    io::{BufWriter, Write},
};
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError, 
    neuron::{
    hodgkin_huxley::HodgkinHuxleyNeuron, 
    iterate_and_spike::{
        DestexheNeurotransmitter, IonotropicNeurotransmitterType, IonotropicType,
        AMPAReceptor, IterateAndSpike, ReceptorKinetics, Receptors,
    }, 
    iterate_coupled_spiking_neurons
}};


// Couples two Hodgkin Huxley neurons with neurotransmission and tracks relevant 
// history regarding voltages, neurotransmitter values, and receptor values
// which are written to a .csv file at the working directory
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let mut presynaptic_neuron = HodgkinHuxleyNeuron::default_impl();
    presynaptic_neuron.receptors
        .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))?;
    presynaptic_neuron.synaptic_neurotransmitters
        .insert(IonotropicNeurotransmitterType::AMPA, DestexheNeurotransmitter::default());

    let mut postsynaptic_neuron = presynaptic_neuron.clone();

    let iterations = 10000;
    let input_current = 30.;
    let electrical_synapse = false;
    let chemical_synapse = true;

    let mut presynaptic_voltages: Vec<f32> = Vec::new();
    let mut postsynaptic_voltages: Vec<f32> = Vec::new();
    let mut presynaptic_neurotransmitter_concs: Vec<f32> = Vec::new();
    let mut receptor_values: Vec<f32> = Vec::new();

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
        presynaptic_neurotransmitter_concs.push(
            *presynaptic_neuron.get_neurotransmitter_concentrations().get(&IonotropicNeurotransmitterType::AMPA)
                .expect("Could not find concentration")
        );
        let ampa_receptor = postsynaptic_neuron.receptors.get(&IonotropicNeurotransmitterType::AMPA)
            .expect("Could not find ligand gate");
        if let IonotropicType::AMPA(receptor) = ampa_receptor {
            receptor_values.push(receptor.r.get_r());
        }
    }

    let mut file = BufWriter::new(File::create("coupled_hodgkin_huxley.csv")
        .expect("Could not create file"));

    writeln!(file, "presynaptic_voltages,postsynaptic_voltages,Ts,rs").expect("Could not write to file");
    for i in 0..iterations {
        writeln!(
            file, 
            "{},{},{},{}", 
            presynaptic_voltages[i],
            postsynaptic_voltages[i], 
            presynaptic_neurotransmitter_concs[i],
            receptor_values[i],
        ).expect("Could not write to file");
    }

    Ok(())
}