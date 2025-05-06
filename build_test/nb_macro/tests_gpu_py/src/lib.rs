use spiking_neural_networks::{
    neuron::{
        Lattice, GridVoltageHistory, LatticeHistory, RunLattice, InternalGraph, 
        SpikeTrainLattice, SpikeTrainGrid, SpikeTrainGridHistory, RunSpikeTrainLattice,
        SpikeTrainLatticeHistory, LatticeNetwork, RunNetwork,
        spike_train::{SpikeTrain, PoissonNeuron, NeuralRefractoriness, DeltaDiracRefractoriness},
        plasticity::STDP,
        gpu_lattices::{LatticeGPU, LatticeNetworkGPU},
    },
    error::LatticeNetworkError,
    graph::{AdjacencyMatrix, Graph, GraphPosition},
};
use nb_macro::neuron_builder;
use pyo3::{types::PyTuple, exceptions::{PyKeyError, PyValueError}};
mod lattices;
use lattices::{
    impl_lattice, impl_lattice_gpu, impl_spike_train_lattice,
    impl_network, impl_network_gpu,
};


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

#[pyclass]
#[pyo3(name = "STDP")]
#[derive(Clone)]
pub struct PySTDP {
    plasticity: STDP
}

#[pymethods]
impl PySTDP {
    #[new]
    fn new() -> Self {
        PySTDP { plasticity: STDP::default() }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self.plasticity))
    }

    #[getter]
    fn get_a_plus(&self) -> f32 {
        self.plasticity.a_plus
    }

    #[setter]
    fn set_a_plus(&mut self, new_param: f32) {
        self.plasticity.a_plus = new_param;
    }

    #[getter]
    fn get_a_minus(&self) -> f32 {
        self.plasticity.a_minus
    }

    #[setter]
    fn set_a_minus(&mut self, new_param: f32) {
        self.plasticity.a_minus = new_param;
    }

    #[getter]
    fn get_tau_plus(&self) -> f32 {
        self.plasticity.tau_plus
    }

    #[setter]
    fn set_tau_plus(&mut self, new_param: f32) {
        self.plasticity.tau_plus = new_param;
    }

    #[getter]
    fn get_tau_minus(&self) -> f32 {
        self.plasticity.tau_minus
    }

    #[setter]
    fn set_tau_minus(&mut self, new_param: f32) {
        self.plasticity.tau_minus = new_param;
    }

    #[getter]
    fn get_dt(&self) -> f32 {
        self.plasticity.dt
    }

    #[setter]
    fn set_dt(&mut self, new_param: f32) {
        self.plasticity.dt = new_param;
    }
}

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

impl_lattice_gpu!(
    PyBasicIntegrateAndFireLatticeGPU, PyBasicIntegrateAndFire, 
    PyBasicIntegrateAndFireLattice, "BasicIntegrateAndFireLatticeGPU"
);

#[pyclass]
#[pyo3(name = "DeltaDiracRefractoriness")]
#[derive(Clone)]
pub struct PyDeltaDiracRefractoriness {
    refractoriness: DeltaDiracRefractoriness,
}

#[pymethods]
impl PyDeltaDiracRefractoriness {
    #[new]
    fn new(k: f32) -> Self {
        PyDeltaDiracRefractoriness {
            refractoriness: DeltaDiracRefractoriness { k }
        }
    }

    #[getter]
    fn get_k(&self) -> f32 {
        self.refractoriness.k
    }

    #[setter]
    fn set_k(&mut self, new_param: f32) {
        self.refractoriness.k = new_param;
    }

    fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32 {
        self.refractoriness.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
    }
}

#[pyclass]
#[pyo3(name = "PoissonNeuron")]
#[derive(Clone)]
pub struct PyPoissonNeuron {
    model: PoissonNeuron<TestReceptorsNeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>,
}

#[pymethods]
impl PyPoissonNeuron {
    #[new]
    fn new() -> Self {
        PyPoissonNeuron { model: PoissonNeuron::default() }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self.model))
    }

    #[getter]
    fn get_current_voltage(&self) -> f32 {
        self.model.current_voltage
    }

    #[setter]
    fn set_current_voltage(&mut self, new_param: f32) {
        self.model.current_voltage = new_param;
    }

    #[getter]
    fn get_v_th(&self) -> f32 {
        self.model.v_th
    }

    #[setter]
    fn set_v_th(&mut self, new_param: f32) {
        self.model.v_th = new_param;
    }

    #[getter]
    fn get_v_resting(&self) -> f32 {
        self.model.v_resting
    }

    #[setter]
    fn set_v_resting(&mut self, new_param: f32) {
        self.model.v_resting = new_param;
    }

    #[getter]
    fn get_chance_of_firing(&self) -> f32 {
        self.model.chance_of_firing
    }

    #[setter]
    fn set_chance_of_firing(&mut self, new_param: f32) {
        self.model.chance_of_firing = new_param;
    }

    #[getter]
    fn get_dt(&self) -> f32 {
        self.model.dt
    }

    #[setter]
    fn set_dt(&mut self, new_param: f32) {
        self.model.dt = new_param;
    }

    #[getter]
    fn get_last_firing_time(&self) -> Option<usize> {
        self.model.last_firing_time
    }

    #[setter]
    fn set_last_firing_time(&mut self, new_param: Option<usize>) {
        self.model.last_firing_time = new_param;
    } 

    #[getter]
    fn get_is_spiking(&self) -> bool {
        self.model.is_spiking
    }

    #[setter]
    fn set_is_spiking(&mut self, flag: bool) {
        self.model.is_spiking = flag;
    }

    fn iterate(&mut self) -> bool {
        self.model.iterate()
    }

    fn get_refractoriness(&self) -> PyDeltaDiracRefractoriness {
        PyDeltaDiracRefractoriness { refractoriness: self.model.neural_refractoriness }
    }

    fn set_refractoriness(&mut self, refractoriness: PyDeltaDiracRefractoriness) {
        self.model.neural_refractoriness = refractoriness.refractoriness;
    }

    fn get_synaptic_neurotransmitters<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        for (key, value) in self.model.synaptic_neurotransmitters.iter() {
            let key_py = Py::new(py, key.convert_type_to_py())?;
            let val_py = Py::new(py, PyBoundedNeurotransmitterKinetics {
                neurotransmitter: value.clone(),
            })?;
            dict.set_item(key_py, val_py)?;
        }

        Ok(dict)
    }

    fn set_synaptic_neurotransmitters(&mut self, neurotransmitters: &PyDict) -> PyResult<()> {
        let current_copy = self.model.synaptic_neurotransmitters.clone();
        let keys: Vec<_> = self.model.synaptic_neurotransmitters.keys().cloned().collect();
        for key in keys.iter() {
            self.model.synaptic_neurotransmitters.remove(key).unwrap();
        }

        for (key, value) in neurotransmitters.iter() {
            let current_type = TestReceptorsNeurotransmitterType::convert_from_py(key);
            if current_type.is_none() {
                self.model.synaptic_neurotransmitters = current_copy;
                return Err(PyTypeError::new_err("Incorrect neurotransmitter type"));
            }
            let current_neurotransmitter = value.extract::<PyBoundedNeurotransmitterKinetics>();
            if current_neurotransmitter.is_err() {
                self.model.synaptic_neurotransmitters = current_copy;
                return Err(PyTypeError::new_err("Incorrect neurotransmitter kinetics type"));
            }
            self.model.synaptic_neurotransmitters.insert(
                current_type.unwrap(), 
                current_neurotransmitter.unwrap().neurotransmitter.clone(),
            );
        }

        Ok(())
    }
}

type LatticeSpikeTrain = PoissonNeuron<TestReceptorsNeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>;

#[pyclass]
#[pyo3(name = "PoissonLattice")]
#[derive(Clone)]
pub struct PyPoissonLattice {
    lattice: SpikeTrainLattice<
        TestReceptorsNeurotransmitterType,
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
    >
}

impl_spike_train_lattice!(PyPoissonLattice, PyPoissonNeuron, LatticeSpikeTrain, "PoissonLattice");

#[pyclass]
#[pyo3(name = "GraphPosition")]
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PyGraphPosition {
    graph_position: GraphPosition
}

#[pymethods]
impl PyGraphPosition {
    #[new]
    fn new(id: usize, pos: (usize, usize)) -> PyGraphPosition {
        PyGraphPosition { graph_position: GraphPosition { id, pos } }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self.graph_position))
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.graph_position.id
    }

    #[setter]
    fn set_id(&mut self, id: usize) {
        self.graph_position.id = id;
    }

    #[getter]
    fn get_pos(&self) -> (usize, usize) {
        self.graph_position.pos
    }

    #[setter]
    fn set_pos(&mut self, pos: (usize, usize)) {
        self.graph_position.pos = pos;
    }
}

type ConnectingAdjacencyMatrix = AdjacencyMatrix<GraphPosition, f32>;

#[pyclass]
#[pyo3(name = "BasicIntegrateAndFireNetwork")]
#[derive(Clone)]
pub struct PyBasicIntegrateAndFireNetwork {
    network: LatticeNetwork<
        BasicIntegrateAndFire<BoundedNeurotransmitterKinetics,  BoundedReceptorKinetics>, 
        LatticeAdjacencyMatrix, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        ConnectingAdjacencyMatrix,
        STDP,
        TestReceptorsNeurotransmitterType,
    >
}

impl_network!(
    PyBasicIntegrateAndFireNetwork, PyBasicIntegrateAndFireLattice, PyPoissonLattice, PyBasicIntegrateAndFire,
    PyPoissonNeuron, PySTDP, "BasicIntegrateAndFireLattice", "PoissonLattice", "BasicIntegrateAndFireNetwork",
);

#[pyclass]
#[pyo3(name = "BasicIntegrateAndFireNetworkGPU")]
pub struct PyBasicIntegrateAndFireNetworkGPU {
    network: LatticeNetworkGPU<
        BasicIntegrateAndFire<BoundedNeurotransmitterKinetics,  BoundedReceptorKinetics>, 
        LatticeAdjacencyMatrix, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        STDP,
        TestReceptorsNeurotransmitterType,
        DeltaDiracRefractoriness,
        ConnectingAdjacencyMatrix,
    >
}

impl_network_gpu!(
    PyBasicIntegrateAndFireNetworkGPU, PyBasicIntegrateAndFireLattice, PyPoissonLattice, PyBasicIntegrateAndFire,
    PyPoissonNeuron, PySTDP, "BasicIntegrateAndFireLattice", "PoissonLattice", "BasicIntegrateAndFireNetworkGPU",
);

#[pymodule]
fn tests_gpu_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBasicIntegrateAndFire>()?;
    m.add_class::<PyBoundedNeurotransmitterKinetics>()?;
    m.add_class::<PyBoundedReceptorKinetics>()?;
    m.add_class::<PyTestReceptorsNeurotransmitterType>()?;
    m.add_class::<PyXReceptor>()?;
    m.add_class::<PyTestReceptors>()?;
    m.add_class::<PySTDP>()?;
    m.add_class::<PyBasicIntegrateAndFireLattice>()?;
    m.add_class::<PyBasicIntegrateAndFireLatticeGPU>()?;
    m.add_class::<PyDeltaDiracRefractoriness>()?;
    m.add_class::<PyPoissonNeuron>()?;
    m.add_class::<PyPoissonLattice>()?;
    m.add_class::<PyGraphPosition>()?;
    m.add_class::<PyBasicIntegrateAndFireNetwork>()?;
    m.add_class::<PyBasicIntegrateAndFireNetworkGPU>()?;

    Ok(())
}
