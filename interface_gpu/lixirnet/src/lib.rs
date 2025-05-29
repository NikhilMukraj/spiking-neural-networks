use spiking_neural_networks::{
    neuron::{
        Lattice, GridVoltageHistory, LatticeHistory, RunLattice, InternalGraph, 
        SpikeTrainLattice, SpikeTrainGrid, SpikeTrainGridHistory, RunSpikeTrainLattice,
        SpikeTrainLatticeHistory, LatticeNetwork, RunNetwork,
        spike_train::{SpikeTrain, RateSpikeTrain, NeuralRefractoriness, DeltaDiracRefractoriness},
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
        current = inh_modifier * g_ampa * ampa_r * (v - e_ampa) + (1 / (1 + (exp(-0.062 * v) * mg / 3.57))) * inh_modifier * g_nmda * (nmda_r ^ nmda_modifier) * (v - e_nmda)
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
#[pyo3(name = "RateSpikeTrain")]
#[derive(Clone)]
pub struct PyRateSpikeTrain {
    model: RateSpikeTrain<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>,
}

#[pymethods]
impl PyRateSpikeTrain {
    #[new]
    fn new() -> Self {
        PyRateSpikeTrain { model: RateSpikeTrain::default() }
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
    fn get_rate(&self) -> f32 {
        self.model.rate
    }

    #[setter]
    fn set_rate(&mut self, new_param: f32) {
        self.model.rate = new_param;
    }

    #[getter]
    fn get_step(&self) -> f32 {
        self.model.step
    }

    #[setter]
    fn set_step(&mut self, new_param: f32) {
        self.model.step = new_param;
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

type LatticeSpikeTrain = RateSpikeTrain<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness>;

#[pyclass]
#[pyo3(name = "RateSpikeTrainLattice")]
#[derive(Clone)]
pub struct PyRateSpikeTrainLattice {
    lattice: SpikeTrainLattice<
        DopaGluGABANeurotransmitterType,
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
    >
}

impl_spike_train_lattice!(PyRateSpikeTrainLattice, PyRateSpikeTrain, LatticeSpikeTrain, "RateSpikeTrainLattice");

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
    PyIzhikevichNeuronNetwork, PyIzhikevichNeuronLattice, PyRateSpikeTrainLattice, PyIzhikevichNeuron,
    PyRateSpikeTrain, PySTDP, "IzhikevichNeuronLattice", "RateSpikeTrainLattice", "IzhikevichNeuronNetwork",
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
    PyIzhikevichNeuronNetworkGPU, PyIzhikevichNeuronNetwork, PyIzhikevichNeuronLattice, PyRateSpikeTrainLattice, PyIzhikevichNeuron,
    PyRateSpikeTrain, PySTDP, "IzhikevichNeuronLattice", "RateSpikeTrainLattice", "IzhikevichNeuronNetworkGPU",
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
    m.add_class::<PyRateSpikeTrain>()?;
    m.add_class::<PyRateSpikeTrainLattice>()?;
    m.add_class::<PyGraphPosition>()?;
    m.add_class::<PyIzhikevichNeuronNetwork>()?;
    m.add_class::<PyIzhikevichNeuronNetworkGPU>()?;

    Ok(())
}

#[cfg(test)]
mod test {
    use rand::prelude::*;
    use spiking_neural_networks::{
        neuron::{
            iterate_and_spike::Receptors,
            gpu_lattices::LatticeNetworkGPU,
            spike_train::RateSpikeTrain,
            Lattice, SpikeTrainLattice, LatticeNetwork, RunNetwork
        },
    };
    use super::{
        DopaGluGABANeurotransmitterType, DopaGluGABAType, GlutamateReceptor, BoundedNeurotransmitterKinetics, IzhikevichNeuron,
        PyIzhikevichNeuronNetwork, PyIzhikevichNeuronNetworkGPU,
    };


    fn check_history(history: &[Vec<Vec<f32>>], gpu_history: &[Vec<Vec<f32>>], tolerance: f32, msg: &str) {
        assert_eq!(history.len(), gpu_history.len());

        for (cpu_cell_grid, gpu_cell_grid) in history.iter().zip(gpu_history.iter()) {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= tolerance, "{} | error: {}, voltage1: {}, voltage2: {}", 
                        msg,
                        error,
                        voltage1,
                        voltage2,
                    );
                }
            }
        }
    }

    #[test]
    fn test_with_spike_trains() {
        let mut spike_train = RateSpikeTrain { rate: 100., dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(DopaGluGABANeurotransmitterType::Glutamate, BoundedNeurotransmitterKinetics::default());

        let mut spike_train_lattice = SpikeTrainLattice::default();
        spike_train_lattice.set_id(0);
        spike_train_lattice.populate(&spike_train, 3, 3).unwrap();
        spike_train_lattice.apply(|neuron: &mut _| neuron.step = rand::thread_rng().gen_range(0.0..=100.));
        spike_train_lattice.update_grid_history = true;

        let mut base_neuron = IzhikevichNeuron::default_impl();
        base_neuron.gap_conductance = 10.;
        base_neuron.c_m = 25.;

        base_neuron.receptors
            .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(DopaGluGABANeurotransmitterType::Glutamate, BoundedNeurotransmitterKinetics::default());

        let mut lattice: Lattice<IzhikevichNeuron<_, _>, _, _, _, DopaGluGABANeurotransmitterType> = Lattice::default_impl();
        lattice.set_id(1);
        lattice.populate(&base_neuron, 3, 3).unwrap();
        lattice.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.c..=neuron.v_th));
        lattice.connect(&(|x, y| x != y), Some(&(|_, _| 5.0)));
        lattice.update_grid_history = true;

        let lattices: Vec<Lattice<IzhikevichNeuron<_, _>, _, _, _, DopaGluGABANeurotransmitterType>> = vec![lattice];
        let spike_train_lattices: Vec<_> = vec![spike_train_lattice];

        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices).unwrap();
        network.connect(0, 1, &(|x, y| x == y), Some(&(|_, _| 5.))).unwrap();
        network.electrical_synapse = true;
        network.chemical_synapse = false;
        network.parallel = true;
        network.set_dt(1.);

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone()).unwrap();
        let mut py_network = PyIzhikevichNeuronNetwork { network: network.clone() };
        let mut py_gpu_network = PyIzhikevichNeuronNetworkGPU { network: LatticeNetworkGPU::from_network(network.clone()).unwrap() };
        
        network.run_lattices(1000).unwrap();
        gpu_network.run_lattices(1000).unwrap();
        py_network.run_lattices(1000).unwrap();
        py_gpu_network.run_lattices(1000).unwrap();
        
        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let py_history = &py_network.network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let py_gpu_history = &py_gpu_network.network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), 1000);
        assert_eq!(gpu_history.len(), 1000);
        assert_eq!(py_history.len(), 1000);
        assert_eq!(py_gpu_history.len(), 1000);

        check_history(cpu_history, gpu_history, 1., "spike_train");
        check_history(cpu_history, py_history, 1., "spike_train");
        check_history(cpu_history, py_gpu_history, 1., "spike_train");

        let cpu_history = &network.get_lattice(&1).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&1).unwrap().grid_history.history;
        let py_history = &py_network.network.get_lattice(&1).unwrap().grid_history.history;
        let py_gpu_history = &py_gpu_network.network.get_lattice(&1).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), 1000);
        assert_eq!(gpu_history.len(), 1000);
        assert_eq!(py_history.len(), 1000);
        assert_eq!(py_gpu_history.len(), 1000);

        check_history(cpu_history, gpu_history, 3., "gpu");
        check_history(cpu_history, py_history, 3., "py");
        check_history(cpu_history, py_gpu_history, 3., "py-gpu");
    }
}
