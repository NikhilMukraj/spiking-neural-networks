use spiking_neural_networks::{
    neuron::{
        GridVoltageHistory, LatticeHistory, RunLattice, InternalGraph,
        gpu_lattices::LatticeGPU,
    },
    error::LatticeNetworkError,
    graph::{AdjacencyMatrix, Graph},
};
use nb_macro::neuron_builder;
use pyo3::{types::PyTuple, exceptions::{PyKeyError, PyValueError}};
mod lattices;
use lattices::impl_lattice;


neuron_builder!("
[neurotransmitter_kinetics]
    type: BoundedNeurotransmitterKinetics
    vars: t_max = 1, c = 0.001, conc = 0
    on_iteration:
        [if] is_spiking [then]
            conc = t_max
        [else]
            conc = 0
        [end]

        t = t + dt * -c * t + conc

        t = min(max(t, 0), t_max)
[end]

[receptor_kinetics]
    type: BoundedReceptorKinetics
    vars: r_max = 1
    on_iteration:
        r = min(max(t, 0), r_max)
[end]

[receptors]
    type: TestReceptors
    kinetics: BoundedReceptorKinetics
    neurotransmitter: X
    vars: current = 0, g = 1, e = 0
    on_iteration:
        current = g * r * (v - e)
[end]

[neuron]
    type: BasicIntegrateAndFire
    kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
    receptors: TestReceptors
    vars: g = 1, e = 0, v_reset = -75, v_th = -55
    on_spike: 
        v = v_reset
    spike_detection: v >= v_th
    on_iteration:
        dv/dt = -g * (v - e) + i
[end]"
);

type LatticeAdjacencyMatrix = AdjacencyMatrix<(usize, usize), f32>;

#[pyclass]
#[pyo3(name = "BasicIntegrateAndFireLatticeGPU")]
pub struct PyBasicIntegrateAndFireLatticeGPU {
    lattice: LatticeGPU<
        BasicIntegrateAndFire<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>,
        LatticeAdjacencyMatrix,
        GridVoltageHistory,
        TestReceptorsNeurotransmitterType,
    >
}

impl_lattice!(PyBasicIntegrateAndFireLatticeGPU, PyBasicIntegrateAndFire, "BasicIntegrateAndFireLatticeGPU");

#[pymodule]
fn tests_gpu_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBasicIntegrateAndFire>()?;
    m.add_class::<PyBoundedNeurotransmitterKinetics>()?;
    m.add_class::<PyBoundedReceptorKinetics>()?;
    m.add_class::<PyTestReceptorsNeurotransmitterType>()?;
    m.add_class::<PyXReceptor>()?;
    m.add_class::<PyTestReceptors>()?;
    m.add_class::<PyBasicIntegrateAndFireLatticeGPU>()?;

    Ok(())
}
