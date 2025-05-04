use spiking_neural_networks::{
    neuron::{
        Lattice, RunLattice, InternalGraph, LatticeHistory, GridVoltageHistory,
        plasticity::STDP,
    }, 
    graph::{Graph, AdjacencyMatrix},
    error::LatticeNetworkError,
};
use nb_macro::neuron_builder;
use pyo3::{types::PyTuple, exceptions::{PyKeyError, PyValueError}};
use std::collections::HashSet;
mod lattices;
use lattices::{PySTDP, impl_lattice};


neuron_builder!(r#"
[ion_channel]
    type: TestLeak
    vars: e = 0, g = 1,
    on_iteration:
        current = g * (v - e)
[end]

[ion_channel]
    type: TestChannel
    vars: e = 0, g = 1
    gating_vars: n
    on_iteration:
        current = g * n.alpha * n.beta * n.state * (v - e)
[end]

[ion_channel]
    type: CalciumIonChannel 
    vars: e = 80, g = 0.025,
    gating_vars: s
    on_iteration:
        s.alpha = 1.6 / (1 + exp(-0.072 * (v - 5)))
        s.beta = (0.02 * (v + 8.9)) / ((exp(v + 8.9) / 5) - 1)

        s.update(dt)

        current = g * -(s.state ^ 2) * (v - e)
[end]

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
    vars: e = 0, v_reset = -75, v_th = -55
    on_spike: 
        v = v_reset
    spike_detection: v >= v_th
    on_iteration:
        dv/dt = -(v - e) + i
[end]

[neuron]
    type: IonChannelNeuron
    kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
    receptors: TestReceptors
    ion_channels: l = TestLeak
    vars: v_reset = -75, v_th = -55
    on_spike: 
        v = v_reset
    spike_detection: v >= v_th
    on_iteration:
        l.update_current(v)
        dv/dt = l.current + i
[end]
"#);


type LatticeAdjacencyMatrix = AdjacencyMatrix<(usize, usize), f32>;

#[pyclass]
#[pyo3(name = "BasicIntegrateAndFireLattice")]
#[derive(Clone)]
pub struct PyBasicIntegrateAndFireLattice {
    lattice: Lattice<
        BasicIntegrateAndFire<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>,
        LatticeAdjacencyMatrix,
        GridVoltageHistory,
        STDP,
        TestReceptorsNeurotransmitterType,
    >
}

impl_lattice!(PyBasicIntegrateAndFireLattice, PyBasicIntegrateAndFire, "BasicIntegrateAndFireLattice", PySTDP);

#[pymodule]
fn tests_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBasicIntegrateAndFire>()?;
    m.add_class::<PyBoundedNeurotransmitterKinetics>()?;
    m.add_class::<PyBoundedReceptorKinetics>()?;
    m.add_class::<PyTestReceptorsNeurotransmitterType>()?;
    m.add_class::<PyXReceptor>()?;
    m.add_class::<PyTestReceptors>()?;
    m.add_class::<PyTestLeak>()?;
    m.add_class::<PyBasicGatingVariable>()?;
    m.add_class::<PyTestChannel>()?;
    m.add_class::<PyCalciumIonChannel>()?;
    m.add_class::<PyIonChannelNeuron>()?;
    m.add_class::<PySTDP>()?;
    m.add_class::<PyBasicIntegrateAndFireLattice>()?;

    Ok(())
}

// add tests to determine if iterate and spike works as intended
// and that getter setters also work
// check if neurotransmitters are edited properly
// check receptors methods work as intended (iterate + current related methods)
