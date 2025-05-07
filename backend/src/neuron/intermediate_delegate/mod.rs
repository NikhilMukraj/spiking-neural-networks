//! An intermediate used by neuron models to pass information from the outer neuron
//! to the synaptic neurotransmitters

use iterate_and_spike_traits::{CurrentVoltage, IsSpiking, Timestep};
use crate::neuron::iterate_and_spike::{CurrentVoltage, IsSpiking, Timestep};


/// An intermediate delegate to pass relevant information to the `Neurotransmitters` object
#[derive(CurrentVoltage, IsSpiking, Timestep)]
pub struct NeurotransmittersIntermediate {
    pub current_voltage: f32,
    pub is_spiking: bool,
    pub dt: f32,
}

impl NeurotransmittersIntermediate {
    /// Creates an intermediate given `CurrentVoltage`, `IsSpiking`, and `TimeStep`
    pub fn from_neuron<T: CurrentVoltage + IsSpiking + Timestep>(neuron: &T) -> Self {
        NeurotransmittersIntermediate { 
            current_voltage: neuron.get_current_voltage(), 
            is_spiking: neuron.is_spiking(), 
            dt: neuron.get_dt(), 
        }
    }
}