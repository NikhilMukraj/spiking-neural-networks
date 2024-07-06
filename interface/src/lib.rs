use pyo3::prelude::*;
use spiking_neural_networks::neuron::{
    iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor},
    integrate_and_fire::IzhikevichNeuron,
};


#[pylcass]
pub struct PyApproximateNeurotransmitter {
    neurotransmitter: ApproximateNeurotransmitter
}

#[pyclass]
pub struct PyIzhikevichNeuron {
    model: IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
}

// macro for getter and setter methods on izhikevich neuron
// macro for getter and setter methods on neurotransmitter

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    // pip install target/wheels/lixirnet-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
    // m.add_function(wrap_pyfunction!(func, m)?)?;

    m.add_class::<PyIzhikevichNeuron>()?;

    Ok(())
}