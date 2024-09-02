extern crate spiking_neural_networks;
use spiking_neural_networks::error::SpikingNeuralNetworksError;
// use spiking_neural_networks::{
//     neuron::{
//         iterate_and_spike::{
//             ApproximateNeurotransmitter, LigandGatedChannel, 
//             IonotropicNeurotransmitterType, AMPADefault,
//         }, 
//         integrate_and_fire::IzhikevichNeuron, 
//         Lattice,
//     }, 
//     error::SpikingNeuralNetworksError, 
// };


// fn main() -> Result<(), SpikingNeuralNetworksError> {
fn main() -> Result<(), SpikingNeuralNetworksError> {
    // check if izhikevich ligand gates calculation is correct too
    // remember to remove pub on par get interal electrical inputs and print statements

    // let mut izhikevich_neuron = IzhikevichNeuron::default_impl();
    // let mut ampa = ApproximateNeurotransmitter::ampa_default();
    // ampa.t = 0.5;
    // izhikevich_neuron.synaptic_neurotransmitters
    //     .insert(IonotropicNeurotransmitterType::AMPA, ampa);
    // izhikevich_neuron.ligand_gates
    //     .insert(IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default());
    // let mut lattice = Lattice::default_impl();
    // lattice.populate(&izhikevich_neuron, 2, 2);

    // lattice.connect(&(|_, _| true), None);

    // let inputs = lattice.par_get_internal_neurotransmitter_inputs();

    // println!("{:#?}", inputs);

    Ok(())
}
