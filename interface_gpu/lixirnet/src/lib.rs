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
    vars: t_max = 1, clearance_constant = 0.001, conc = 0
    on_iteration:
        [if] is_spiking [then]
            conc = t_max
        [else]
            conc = 0
        [end]

        t = t + dt * -clearance_constant * t + conc

        t = min(max(t, 0), t_max)
[end]

[receptor_kinetics]
    type: BoundedReceptorKinetics
    vars: r_max = 1
    on_iteration:
        r = min(max(t, 0), r_max)
[end]

[receptors]
    type: DopaGluGABA
    kinetics: BoundedReceptorKinetics
    vars: inh_modifier = 1, nmda_modifier = 1
    neurotransmitter: Glutamate
    receptors: ampa_r, nmda_r
    vars: current = 0, g_ampa = 1, g_nmda = 0.6, e_ampa = 0, e_nmda = 0, mg = 0.3
    on_iteration:
        current = inh_modifier * g_ampa * ampa_r * (v - e_ampa) + (1 / (1 + exp(-0.062 * v) / mg * 3.57)) * inh_modifier * g_nmda * nmda_r ^ nmda_modifier * (v - e_nmda)
    neurotransmitter: GABA
    vars: current = 0, g = 1.2, e = -80
    on_iteration:
        current = g * r * (v - e)
    neurotransmitter: Dopamine
    receptors: r_d1, r_d2
    vars: s_d2 = 0, s_d1 = 0
    on_iteration:
        inh_modifier = 1 - (r_d2 * s_d2)
        nmda_modifier = 1 - (r_d1 * s_d1)
[end]

[neuron]
    type: IzhikevichNeuron
    kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
    receptors: DopaGluGABA
    vars: u = 30, a = 0.02, b = 0.2, c = -55, d = 8, v_th = 30, tau_m = 1
    on_spike: 
        v = c
        u += d
    spike_detection: v >= v_th
    on_iteration:
        du/dt = (a * (b * v - u)) / tau_m
        dv/dt = (0.04 * v ^ 2 + 5 * v + 140 - u + i) / c_m
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
#[pyo3(name = "IzhikevichNeuronLattice")]
#[derive(Clone)]
pub struct PyIzhikevichNeuronLattice {
    lattice: Lattice<
        IzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>,
        LatticeAdjacencyMatrix,
        GridVoltageHistory,
        STDP,
        DopaGluGABANeurotransmitterType,
    >
}

impl_lattice!(PyIzhikevichNeuronLattice, PyIzhikevichNeuron, "IzhikevichNeuronLattice", PySTDP);

#[pyclass]
#[pyo3(name = "IzhikevichNeuronLatticeGPU")]
pub struct PyIzhikevichNeuronLatticeGPU {
    lattice: LatticeGPU<
        IzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics>,
        LatticeAdjacencyMatrix,
        GridVoltageHistory,
        DopaGluGABANeurotransmitterType,
    >
}

impl_lattice_gpu!(
    PyIzhikevichNeuronLatticeGPU, PyIzhikevichNeuron, 
    PyIzhikevichNeuronLattice, "IzhikevichNeuronLatticeGPU"
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
    model: PoissonNeuron<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>,
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
            let current_type = DopaGluGABANeurotransmitterType::convert_from_py(key);
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

type LatticeSpikeTrain = PoissonNeuron<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>;

#[pyclass]
#[pyo3(name = "PoissonLattice")]
#[derive(Clone)]
pub struct PyPoissonLattice {
    lattice: SpikeTrainLattice<
        DopaGluGABANeurotransmitterType,
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
#[pyo3(name = "IzhikevichNeuronNetwork")]
#[derive(Clone)]
pub struct PyIzhikevichNeuronNetwork {
    network: LatticeNetwork<
        IzhikevichNeuron<BoundedNeurotransmitterKinetics,  BoundedReceptorKinetics>, 
        LatticeAdjacencyMatrix, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        ConnectingAdjacencyMatrix,
        STDP,
        DopaGluGABANeurotransmitterType,
    >
}

impl_network!(
    PyIzhikevichNeuronNetwork, PyIzhikevichNeuronLattice, PyPoissonLattice, PyIzhikevichNeuron,
    PyPoissonNeuron, PySTDP, "IzhikevichNeuronLattice", "PoissonLattice", "IzhikevichNeuronNetwork",
);

#[pyclass]
#[pyo3(name = "IzhikevichNeuronNetworkGPU")]
pub struct PyIzhikevichNeuronNetworkGPU {
    network: LatticeNetworkGPU<
        IzhikevichNeuron<BoundedNeurotransmitterKinetics,  BoundedReceptorKinetics>, 
        LatticeAdjacencyMatrix, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        STDP,
        DopaGluGABANeurotransmitterType,
        DeltaDiracRefractoriness,
        ConnectingAdjacencyMatrix,
    >
}

impl_network_gpu!(
    PyIzhikevichNeuronNetworkGPU, PyIzhikevichNeuronLattice, PyPoissonLattice, PyIzhikevichNeuron,
    PyPoissonNeuron, PySTDP, "IzhikevichNeuronLattice", "PoissonLattice", "IzhikevichNeuronNetworkGPU",
);

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PyBoundedNeurotransmitterKinetics>()?;
    m.add_class::<PyBoundedReceptorKinetics>()?;
    m.add_class::<PyDopaGluGABANeurotransmitterType>()?;
    m.add_class::<PyGlutamateReceptor>()?;
    m.add_class::<PyGABAReceptor>()?;
    m.add_class::<PyDopamineReceptor>()?;
    m.add_class::<PyDopaGluGABA>()?;
    m.add_class::<PySTDP>()?;
    m.add_class::<PyIzhikevichNeuronLattice>()?;
    m.add_class::<PyIzhikevichNeuronLatticeGPU>()?;
    m.add_class::<PyDeltaDiracRefractoriness>()?;
    m.add_class::<PyPoissonNeuron>()?;
    m.add_class::<PyPoissonLattice>()?;
    m.add_class::<PyGraphPosition>()?;
    m.add_class::<PyIzhikevichNeuronNetwork>()?;
    m.add_class::<PyIzhikevichNeuronNetworkGPU>()?;

    Ok(())
}
