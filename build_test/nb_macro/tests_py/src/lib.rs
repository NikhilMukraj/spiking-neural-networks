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

#[cfg(test)]
mod tests {
    use super::*;


    const ITERATIONS: usize = 1000;
    
    #[test]
    fn test_iterate_and_spike() {
        let is = vec![10., 20., 30., 40., 50.];

        pyo3::prepare_freethreaded_python();

        for i in is {
            let mut neuron = BasicIntegrateAndFire::default_impl();
            let mut reference_voltages: Vec<f32> = vec![];

            for _ in 0..ITERATIONS {
                let _ = neuron.iterate_and_spike(i);
                reference_voltages.push(neuron.current_voltage);
            }

            Python::with_gil(|py| {
                let instance = Py::new(py, PyBasicIntegrateAndFire::new()).unwrap();
                let mut voltages: Vec<f32> = vec![];
                for _ in 0..ITERATIONS {
                    let _: bool = instance.call_method1(py, "iterate_and_spike", (i,))
                        .unwrap()
                        .extract(py)
                        .unwrap();
                    voltages.push(
                        instance.getattr(py, "current_voltage")
                            .unwrap()
                            .extract(py)
                            .unwrap()
                    );
                }

                assert_eq!(reference_voltages, voltages);
            });
        }
    }

    #[test]
    fn test_current() {
        let voltages: Vec<f32> = (-10..10).map(|i| i as f32).collect();

        pyo3::prepare_freethreaded_python();

        for voltage in voltages {
            let mut leak = TestLeak::default();
            let mut reference_currents = vec![];

            for _ in 0..ITERATIONS {
                leak.update_current(voltage);
                reference_currents.push(leak.current);
            }

            Python::with_gil(|py| {
                let instance = Py::new(py, PyTestLeak::new()).unwrap();
                let mut currents: Vec<f32> = vec![];
                for _ in 0..ITERATIONS {
                    let _ = instance.call_method1(py, "update_current", (voltage,))
                        .unwrap();
                    currents.push(
                        instance.getattr(py, "current")
                            .unwrap()
                            .extract(py)
                            .unwrap()
                    );
                }

                assert_eq!(reference_currents, currents);
            });
        }
    }

    #[test]
    fn test_calcium_ion_channel() {
        let voltages: Vec<f32> = (-10..10).map(|i| i as f32).collect();
        let dts: Vec<f32> = vec![0.1, 0.5, 1.];

        pyo3::prepare_freethreaded_python();

        for voltage in voltages {
            for dt in &dts {
                let mut ca = CalciumIonChannel::default();
                let mut reference_currents = vec![];

                for _ in 0..ITERATIONS {
                    ca.update_current(voltage, *dt);
                    reference_currents.push(ca.current);
                }

                Python::with_gil(|py| {
                    let instance = Py::new(py, PyCalciumIonChannel::new()).unwrap();
                    let mut currents: Vec<f32> = vec![];
                    for _ in 0..ITERATIONS {
                        let _ = instance.call_method1(py, "update_current", (voltage, *dt,))
                            .unwrap();
                        currents.push(
                            instance.getattr(py, "current")
                                .unwrap()
                                .extract(py)
                                .unwrap()
                        );
                    }

                    assert_eq!(reference_currents, currents);
                });
            }
        }
    }

    #[test]
    fn test_channel_with_gating_vars() {
        let voltages: Vec<f32> = (-10..10).map(|i| i as f32).collect();

        pyo3::prepare_freethreaded_python();

        for voltage in voltages {
            let mut channel = TestChannel::default();
            let mut reference_currents = vec![];
            let mut reference_as = vec![];
            let mut reference_bs = vec![];
            let mut reference_states = vec![];

            for _ in 0..ITERATIONS {
                channel.update_current(voltage);
                reference_currents.push(channel.current);
                reference_as.push(channel.n.alpha);
                reference_bs.push(channel.n.beta);
                reference_states.push(channel.n.state);
            }

            Python::with_gil(|py| {
                let instance = Py::new(py, PyTestChannel::new()).unwrap();
                let mut currents: Vec<f32> = vec![];
                let mut alphas: Vec<f32> = vec![];
                let mut betas: Vec<f32> = vec![];
                let mut states: Vec<f32> = vec![];
                for _ in 0..ITERATIONS {
                    let _ = instance.call_method1(py, "update_current", (voltage,))
                        .unwrap();
                    currents.push(
                        instance.getattr(py, "current")
                            .unwrap()
                            .extract(py)
                            .unwrap()
                    );
                    let gating_var: PyBasicGatingVariable = instance.call_method0(py, "get_n")
                        .unwrap()
                        .extract(py)
                        .unwrap();
                    alphas.push(
                        gating_var.gating_variable.alpha
                    );
                    betas.push(
                        gating_var.gating_variable.beta
                    );
                    states.push(
                        gating_var.gating_variable.state
                    );
                }

                assert_eq!(reference_currents, currents);
                assert_eq!(reference_as, alphas);
                assert_eq!(reference_bs, betas);
                assert_eq!(reference_states, states);
            });
        }
    }

    // test editing receptors and neurotransmitters and both kinetics types
    // test updating methods (iterate, apply changes, etc)
    // test lattice and lattice network methods
}