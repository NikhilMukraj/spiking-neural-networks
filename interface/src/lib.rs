use std::{collections::{hash_map::DefaultHasher, HashMap, HashSet}, hash::{Hash, Hasher}};
use pyo3::{exceptions::{PyKeyError, PyValueError}, prelude::*, types::{PyDict, PyList, PyTuple}};
use spiking_neural_networks::{
    error::LatticeNetworkError, graph::{AdjacencyMatrix, Graph, GraphPosition}, neuron::{
    integrate_and_fire::IzhikevichNeuron, iterate_and_spike::{
        AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, GABAaDefault, 
        GABAbDefault, IterateAndSpike, LastFiringTime, LigandGatedChannel, 
        LigandGatedChannels, NMDADefault, NeurotransmitterConcentrations, 
        NeurotransmitterType, Neurotransmitters, 
    }, spike_train::{DeltaDiracRefractoriness, NeuralRefractoriness, PoissonNeuron, SpikeTrain},
    GridVoltageHistory, Lattice, LatticeHistory, LatticeNetwork, SpikeTrainGridHistory,
    SpikeTrainLattice, SpikeTrainLatticeHistory
}};


macro_rules! impl_repr {
    ($name:ident, $field:ident) => {
        #[pymethods]
        impl $name {
            fn __repr__(&self) -> PyResult<String> {
                Ok(format!("{:#?}", self.$field))
            }
        }
    };
}

#[pyclass]
#[pyo3(name = "NeurotransmitterType")]
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub enum PyNeurotransmitterType {
    Basic,
    AMPA,
    GABAa,
    GABAb,
    NMDA,
}

impl PyNeurotransmitterType {
    pub fn convert_type(&self) -> NeurotransmitterType {
        match self {
            PyNeurotransmitterType::Basic => NeurotransmitterType::Basic,
            PyNeurotransmitterType::AMPA => NeurotransmitterType::AMPA,
            PyNeurotransmitterType::GABAa => NeurotransmitterType::GABAa,
            PyNeurotransmitterType::GABAb => NeurotransmitterType::GABAb,
            PyNeurotransmitterType::NMDA => NeurotransmitterType::NMDA,
        }
    }
}

#[pymethods]
impl PyNeurotransmitterType {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

macro_rules! implement_basic_getter_and_setter {
    ($name:ident, $field:ident, $($param:ident, $get_name:ident, $set_name:ident),+) => {
        #[pymethods]
        impl $name {
            $(
                #[getter]
                fn $get_name(&self) -> f32 {
                    self.$field.$param
                }

                #[setter]
                fn $set_name(&mut self, new_param: f32) {
                    self.$field.$param = new_param;
                }
            )+
        }
    };
}

macro_rules! implement_nested_getter_and_setter {
    ($name:ident, $field:ident, $nested_field:ident, $($param:ident, $py_name:ident, $get_name:ident, $set_name:ident),+) => {
        #[pymethods]
        impl $name {
            $(
                #[getter($py_name)]
                fn $get_name(&self) -> f32 {
                    self.$field.$nested_field.$param
                }

                #[setter($py_name)]
                fn $set_name(&mut self, new_param: f32) {
                    self.$field.$nested_field.$param = new_param;
                }
            )+
        }
    };
}

#[pyclass]
#[pyo3(name = "ApproximateNeurotransmitter")]
#[derive(Clone, Copy)]
pub struct PyApproximateNeurotransmitter {
    neurotransmitter: ApproximateNeurotransmitter,
}

implement_basic_getter_and_setter!(
    PyApproximateNeurotransmitter, 
    neurotransmitter,
    t_max, get_t_max, set_t_max, 
    t, get_t, set_t, 
    v_th, get_v_th, set_v_th, 
    dt, get_dt, set_dt, 
    clearance_constant, get_clearance_constant, set_clearance_constant
);

#[pymethods]
impl PyApproximateNeurotransmitter {
    #[new]
    #[pyo3(signature = (t_max=1., t=0., v_th=30., dt=0.1, clearance_constant=0.1))]
    fn new(t_max: f32, t: f32, v_th: f32, dt: f32, clearance_constant: f32) -> Self {
        PyApproximateNeurotransmitter {
            neurotransmitter: ApproximateNeurotransmitter {
                t_max: t_max,
                t: t,
                v_th: v_th,
                dt: dt,
                clearance_constant: clearance_constant,
            }
        }
    }
}

impl_repr!(PyApproximateNeurotransmitter, neurotransmitter);

#[pyclass]
#[pyo3(name = "ApproximateNeurotransmitters")]
#[derive(Clone)]
pub struct PyApproximateNeurotransmitters {
    neurotransmitters: Neurotransmitters<ApproximateNeurotransmitter>
}

impl_repr!(PyApproximateNeurotransmitters, neurotransmitters);

#[pymethods]
impl PyApproximateNeurotransmitters {
    #[new]
    #[pyo3(signature = (neurotransmitter_types=None))]
    fn new(neurotransmitter_types: Option<&PyList>) -> PyResult<Self> {
        let mut neurotransmitters: HashMap<NeurotransmitterType, ApproximateNeurotransmitter> = HashMap::new();

        match neurotransmitter_types {
            Some(values) => {
                for i in values.iter() {
                    let current_type = i.extract::<PyNeurotransmitterType>()?.convert_type();
                    let neurotransmitter = match current_type {
                        NeurotransmitterType::Basic => ApproximateNeurotransmitter::default(),
                        NeurotransmitterType::AMPA => ApproximateNeurotransmitter::ampa_default(),
                        NeurotransmitterType::GABAa => ApproximateNeurotransmitter::gabaa_default(),
                        NeurotransmitterType::GABAb => ApproximateNeurotransmitter::gabab_default(),
                        NeurotransmitterType::NMDA => ApproximateNeurotransmitter::nmda_default(),
                    };
        
                    neurotransmitters.insert(current_type, neurotransmitter);
                }
            },
            None => {}
        };

        Ok(
            PyApproximateNeurotransmitters {
                neurotransmitters: Neurotransmitters { neurotransmitters: neurotransmitters }
            }
        )
    }

    fn __getitem__(&self, neurotransmitter_type: PyNeurotransmitterType) -> PyResult<PyApproximateNeurotransmitter> {
        if let Some(value) = self.neurotransmitters.get(&neurotransmitter_type.convert_type()) {
            Ok(
                PyApproximateNeurotransmitter { 
                    neurotransmitter: *value 
                }
            )
        } else {
            Err(PyKeyError::new_err(format!("{:#?} not found", neurotransmitter_type)))
        }
    }

    fn set_neurotransmitter(
        &mut self, neurotransmitter_type: PyNeurotransmitterType, neurotransmitter: PyApproximateNeurotransmitter
    ) {
        self.neurotransmitters.neurotransmitters.insert(
            neurotransmitter_type.convert_type(), neurotransmitter.neurotransmitter
        );
    }

    fn apply_t_changes(&mut self, voltage: f32) {
        self.neurotransmitters.apply_t_changes(voltage);
    }
}

// #[pyclass]
// #[pyo3(name = "ApproximateReceptor")]
// #[derive(Clone)]
// pub struct PyApproximateReceptor {
//     receptor: ApproximateReceptor
// }

// implement_basic_getter_and_setter!(
//     PyApproximateReceptor, 
//     receptor,
//     r, get_r, set_r 
// );

// #[pymethods]
// impl PyApproximateReceptor {
//     #[new]
//     #[pyo3(signature = (r=0.))]
//     fn new(r: f32) -> Self {
//         PyApproximateReceptor {
//             receptor: ApproximateReceptor {
//                 r: r
//             }
//         }
//     }
// }

#[pyclass]
#[pyo3(name = "ApproximateLigandGatedChannel")]
#[derive(Clone)]
pub struct PyApproximateLigandGatedChannel {
    ligand_gate: LigandGatedChannel<ApproximateReceptor>
}

implement_basic_getter_and_setter!(
    PyApproximateLigandGatedChannel, 
    ligand_gate,
    g, get_g, set_g,
    reversal, get_reversal, set_reversal,
    current, get_current, set_current
);

impl_repr!(PyApproximateLigandGatedChannel, ligand_gate);

#[pymethods]
impl PyApproximateLigandGatedChannel {
    #[new]
    #[pyo3(signature = (receptor_type=PyNeurotransmitterType::Basic))]
    fn new(receptor_type: PyNeurotransmitterType) -> Self {
        let ligand_gate = match receptor_type.convert_type() {
            NeurotransmitterType::Basic => LigandGatedChannel::default(),
            NeurotransmitterType::AMPA => LigandGatedChannel::ampa_default(),
            NeurotransmitterType::GABAa => LigandGatedChannel::gabaa_default(),
            NeurotransmitterType::GABAb => LigandGatedChannel::gabab_default(),
            NeurotransmitterType::NMDA => LigandGatedChannel::nmda_default(),
        };

        PyApproximateLigandGatedChannel {
            ligand_gate: ligand_gate
        }
    }

    #[getter]
    fn get_r(&self) -> f32 {
        self.ligand_gate.receptor.r
    }

    #[setter]
    fn set_r(&mut self, new_r: f32) {
        self.ligand_gate.receptor.r = new_r;
    }
}

fn pydict_to_neurotransmitters_concentration(dict: &PyDict) -> PyResult<NeurotransmitterConcentrations> {
    let mut neurotransmitter_concs: HashMap<NeurotransmitterType, f32> = HashMap::new();

    for (key, value) in dict.iter() {
        let current_type = key.extract::<PyNeurotransmitterType>()?.convert_type();
        let conc = value.extract::<f32>()?;

        neurotransmitter_concs.insert(current_type, conc);
    }

    Ok(
        neurotransmitter_concs
    )
}


#[pyclass]
#[pyo3(name = "ApproximateLigandGatedChannels")]
#[derive(Clone)]
pub struct PyApproximateLigandGatedChannels {
    ligand_gates: LigandGatedChannels<ApproximateReceptor>
}

impl_repr!(PyApproximateLigandGatedChannels, ligand_gates);

#[pymethods]
impl PyApproximateLigandGatedChannels {
    #[new]
    #[pyo3(signature = (neurotransmitter_types=None))]
    fn new(neurotransmitter_types: Option<&PyList>) -> PyResult<Self> {
        let mut ligand_gates: HashMap<NeurotransmitterType, LigandGatedChannel<ApproximateReceptor>> = HashMap::new();

        match neurotransmitter_types {
            Some(values) => {
                for i in values.iter() {
                    let current_type = i.extract::<PyNeurotransmitterType>()?.convert_type();
                    let neurotransmitter = match current_type {
                        NeurotransmitterType::Basic => LigandGatedChannel::default(),
                        NeurotransmitterType::AMPA => LigandGatedChannel::ampa_default(),
                        NeurotransmitterType::GABAa => LigandGatedChannel::gabaa_default(),
                        NeurotransmitterType::GABAb => LigandGatedChannel::gabab_default(),
                        NeurotransmitterType::NMDA => LigandGatedChannel::nmda_default(),
                    };
        
                    ligand_gates.insert(current_type, neurotransmitter);
                }
            },
            None => {}
        };

        Ok(
            PyApproximateLigandGatedChannels {
                ligand_gates: LigandGatedChannels { ligand_gates: ligand_gates }
            }
        )
    }

    fn __getitem__(&self, neurotransmitter_type: PyNeurotransmitterType) -> PyResult<PyApproximateLigandGatedChannel> {
        if let Some(value) = self.ligand_gates.get(&neurotransmitter_type.convert_type()) {
            Ok(
                PyApproximateLigandGatedChannel { 
                    ligand_gate: value.clone() 
                }
            )
        } else {
            Err(PyKeyError::new_err(format!("{:#?} not found", neurotransmitter_type)))
        }
    }

    fn set_ligand_gate(
        &mut self, neurotransmitter_type: PyNeurotransmitterType, ligand_gate: PyApproximateLigandGatedChannel
    ) {
        self.ligand_gates.ligand_gates.insert(
            neurotransmitter_type.convert_type(), ligand_gate.ligand_gate
        );
    }

    fn update_receptor_kinetics(&mut self, neurotransmitter_concs: &PyDict) -> PyResult<()> {
        let neurotransmitter_concs = pydict_to_neurotransmitters_concentration(neurotransmitter_concs)?;

        self.ligand_gates.update_receptor_kinetics(&neurotransmitter_concs);

        Ok(())
    }
}

#[pyclass]
#[pyo3(name = "IzhikevichNeuron")]
#[derive(Clone)]
pub struct PyIzhikevichNeuron {
    // could try dyn neurotransmitter kinetics
    model: IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
}

implement_basic_getter_and_setter!(
    PyIzhikevichNeuron, 
    model,
    current_voltage, get_current_voltage, set_current_voltage,
    a, get_a, set_a, 
    b, get_b, set_b, 
    c, get_c, set_c, 
    d, get_d, set_d, 
    dt, get_dt, set_dt, 
    v_th, get_v_th, set_v_th, 
    v_init, get_v_init, set_v_init, 
    w_value, get_w, set_w,
    w_init, get_w_init, set_w_init,
    gap_conductance, get_gap_conductance, set_gap_conductance,
    tau_m, get_tau_m, set_tau_m,
    c_m, get_c_m, set_c_m
);

implement_nested_getter_and_setter!(
    PyIzhikevichNeuron,
    model,
    gaussian_params,
    mean, gaussian_mean, get_gaussian_mean, set_gaussian_mean,
    std, gaussian_std, get_gaussian_std, set_gaussian_std,
    min, gaussian_min, get_gaussian_min, set_gaussian_min,
    max, gaussian_max, get_gaussian_max, set_gaussian_max
);

implement_nested_getter_and_setter!(
    PyIzhikevichNeuron,
    model,
    stdp_params,
    a_plus, a_plus, get_a_plis, set_a_plis,
    a_minus, a_minus, get_a_minus, set_a_minus,
    tau_minus, tau_minus, get_tau_minus, set_tau_minus,
    tau_plus, tau_plus, get_tau_plus, set_tau_plus
);

impl_repr!(PyIzhikevichNeuron, model);

#[pymethods]
impl PyIzhikevichNeuron {
    #[new]
    #[pyo3(signature = (
        a=0.02, b=0.2, c=-55., d=8., v_th=30., dt=0.1, current_voltage=-65., 
        v_init=-65., w_value=30., w_init=30., gap_conductance=10., tau_m=1., c_m=100.,
    ))]
    fn new(
        a: f32, b: f32, c: f32, d: f32, v_th: f32, dt: f32, current_voltage: f32, v_init: f32, 
        w_value: f32, w_init: f32, gap_conductance: f32, tau_m: f32, c_m: f32
    ) -> Self {
        PyIzhikevichNeuron {
            model: IzhikevichNeuron {
                a: a,
                b: b,
                c: c,
                d: d,
                current_voltage: current_voltage,
                v_init: v_init,
                v_th: v_th,
                dt: dt,
                w_value: w_value,
                w_init: w_init,
                gap_conductance: gap_conductance,
                tau_m: tau_m,
                c_m: c_m,
                ..IzhikevichNeuron::default()
            }
        }
    }

    fn iterate_and_spike(&mut self, i: f32) -> bool {
        self.model.iterate_and_spike(i)
    }

    #[pyo3(signature = (i, neurotransmitter_concs))]
    fn iterate_with_neurotransmitter_and_spike(&mut self, i: f32, neurotransmitter_concs: &PyDict) -> PyResult<bool> {
        let neurotransmitter_concs = pydict_to_neurotransmitters_concentration(neurotransmitter_concs)?;

        Ok(self.model.iterate_with_neurotransmitter_and_spike(i, &neurotransmitter_concs))
    }

    fn get_neurotransmitters(&self) -> PyApproximateNeurotransmitters {
        PyApproximateNeurotransmitters { neurotransmitters: self.model.get_neurotransmitters().clone() }
    }

    fn set_neurotransmitters(&mut self, neurotransmitters: PyApproximateNeurotransmitters) {
        self.model.synaptic_neurotransmitters = neurotransmitters.neurotransmitters;
    }

    fn get_ligand_gates(&self) -> PyApproximateLigandGatedChannels {
        PyApproximateLigandGatedChannels { ligand_gates: self.model.get_ligand_gates().clone() }
    }

    fn set_ligand_gates(&mut self, ligand_gates: PyApproximateLigandGatedChannels) {
        self.model.ligand_gates = ligand_gates.ligand_gates;
    }

    #[getter(last_firing_time)]
    fn get_last_firing_time(&self) -> Option<usize> {
        self.model.get_last_firing_time()
    }

    #[setter(last_firing_time)]
    fn set_last_firing_time(&mut self, timestep: Option<usize>) {
        self.model.set_last_firing_time(timestep);
    }
}

#[pyclass]
#[pyo3(name = "DeltaDiracRefractoriness")]
#[derive(Clone)]
pub struct PyDeltaDiracRefractoriness {
    refractoriness: DeltaDiracRefractoriness,
}

implement_basic_getter_and_setter!(
    PyDeltaDiracRefractoriness, 
    refractoriness,
    k, get_k, set_k
);

#[pymethods]
impl PyDeltaDiracRefractoriness {
    #[new]
    fn new(k: f32) -> Self {
        PyDeltaDiracRefractoriness {
            refractoriness: DeltaDiracRefractoriness { k: k }
        }
    }

    fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32 {
        self.refractoriness.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
    }
}

#[pyclass]
#[pyo3(name = "PoissonNeuron")]
#[derive(Clone)]
pub struct PyPoissonNeuron {
    model: PoissonNeuron<ApproximateNeurotransmitter, DeltaDiracRefractoriness>,
}

implement_basic_getter_and_setter!(
    PyPoissonNeuron, 
    model,
    current_voltage, get_current_voltage, set_current_voltage,
    v_th, get_v_th, set_v_th,
    v_resting, get_v_resting, set_v_resting,
    chance_of_firing, get_chance_of_firing, set_chance_of_firing,
    refractoriness_dt, get_refractoriness_dt, set_refractoriness_dt
);

impl_repr!(PyPoissonNeuron, model);

#[pymethods]
impl PyPoissonNeuron {
    #[new]
    #[pyo3(signature = (current_voltage=0., v_th=30., v_resting=0., chance_of_firing=0.01, refactoriness_dt=0.1))]
    fn new(
        current_voltage: f32, v_th: f32, v_resting: f32, chance_of_firing: f32, refactoriness_dt: f32
    ) -> Self {
        PyPoissonNeuron {
            model: PoissonNeuron { 
                current_voltage: current_voltage, 
                v_th: v_th, 
                v_resting: v_resting, 
                last_firing_time: None, 
                synaptic_neurotransmitters: Neurotransmitters::default(), 
                neural_refractoriness: DeltaDiracRefractoriness::default(), 
                chance_of_firing: chance_of_firing, 
                refractoriness_dt: refactoriness_dt, 
            }
        }
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

    fn get_neurotransmitters(&self) -> PyApproximateNeurotransmitters {
        PyApproximateNeurotransmitters { neurotransmitters: self.model.synaptic_neurotransmitters.clone() }
    }

    fn set_neurotransmitters(&mut self, neurotransmitters: PyApproximateNeurotransmitters) {
        self.model.synaptic_neurotransmitters = neurotransmitters.neurotransmitters;
    }
}

// eventually use macro to generate lattices for each neuronal type
// could have user precompile or dynamically dispatch neuron lattice type
// depending on user input and wrap relevant functions
type LatticeNeuron = IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
type PyLatticeNeuron = PyIzhikevichNeuron;

#[pyclass]
#[pyo3(name = "IzhikevichLattice")]
#[derive(Clone)]
pub struct PyIzhikevichLattice {
    lattice: Lattice<
        LatticeNeuron,
        AdjacencyMatrix<(usize, usize)>,
        GridVoltageHistory,
    >
}

#[pymethods]
impl PyIzhikevichLattice {
    #[new]
    fn new() -> Self {
        PyIzhikevichLattice { lattice: Lattice::default() }
    }

    fn populate(&mut self, neuron: PyLatticeNeuron, num_rows: usize, num_cols: usize) {
        self.lattice.populate(&neuron.model, num_rows, num_cols);
    }

    #[pyo3(signature = (connection_conditional, weight_logic=None))]
    fn connect(&mut self, py: Python, connection_conditional: &PyAny, weight_logic: Option<&PyAny>) {
        let py_callable = connection_conditional.to_object(connection_conditional.py());

        let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> bool {
            let args = PyTuple::new(py, &[a, b]);
            py_callable.call1(py, args).unwrap().extract::<bool>(py).unwrap()
        };

        let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> f32>> = match weight_logic {
            Some(value) => {
                let py_callable = value.to_object(value.py()); 

                let closure = move |a: (usize, usize), b: (usize, usize)| -> f32 {
                    let args = PyTuple::new(py, &[a, b]);
                    py_callable.call1(py, args).unwrap().extract::<f32>(py).unwrap()
                };

                Some(Box::new(closure))
            },
            None => None,
        };

        self.lattice.connect(&connection_closure, weight_closure.as_deref());
    }

    fn get_every_node(&self) -> HashSet<(usize, usize)> {
        self.lattice.graph.get_every_node()
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.lattice.get_id()
    }

    #[setter]
    fn set_id(&mut self, id: usize) {
        self.lattice.set_id(id)
    }

    fn get_neuron(&self, row: usize, col: usize) -> PyResult<PyLatticeNeuron> {
        let neuron = match self.lattice.cell_grid.get(row) {
            Some(row_cells) => match row_cells.get(col) {
                Some(neuron) => neuron.clone(),
                None => {
                    return Err(PyKeyError::new_err(format!("Column at {} not found", col)));
                }
            },
            None => {
                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
            }
        };

        Ok(
            PyIzhikevichNeuron { 
                model: neuron
            }
        )
    }

    fn set_neuron(&mut self, row: usize, col: usize, neuron: PyLatticeNeuron) -> PyResult<()> {
        let row_cells = match self.lattice.cell_grid.get_mut(row) {
            Some(row_cells) => row_cells,
            None => {
                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
            }
        };

        if let Some(existing_neuron) = row_cells.get_mut(col) {
            *existing_neuron = neuron.model.clone();

            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("Column at {} not found", col)))
        }
    }

    fn get_weight(&self, presynaptic: (usize, usize), postsynaptic: (usize, usize)) -> PyResult<f32> {
        match self.lattice.graph.lookup_weight(&presynaptic, &postsynaptic) {
            Ok(value) => Ok(value.unwrap_or(0.)),
            Err(_) => Err(PyKeyError::new_err(
                format!("Weight at ({:#?}, {:#?}) not found", presynaptic, postsynaptic))
            )
        }
    }

    // fn set_weight(&mut self, presynaptic: (usize, usize), postsynaptic: (usize, usize), weight: f32) -> PyResult<()> {
    //     let weight = if weight == 0. {
    //         None
    //     } else {
    //         Some(weight)
    //     };
        
    //     match self.lattice.graph.edit_weight(&presynaptic, &postsynaptic, weight) {
    //         Ok(_) => Ok(()),
    //         Err(_) => Err(PyKeyError::new_err(
    //             format!("Connection at ({:#?}, {:#?}) not found", presynaptic, postsynaptic))
    //         ),
    //     }
    // }

    fn reset_timing(&mut self) {
        self.lattice.reset_timing();
    }

    fn run_lattice(&mut self, iterations: usize) -> PyResult<()> {
        match self.lattice.run_lattice(iterations) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
        }
    }

    fn run_lattice_chemical_synapse_only(&mut self, iterations: usize) -> PyResult<()> {
        match self.lattice.run_lattice_chemical_synapses_only(iterations) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
        }
    }

    fn run_lattice_chemical_and_electrical_synapses(&mut self, iterations: usize) -> PyResult<()> {
        match self.lattice.run_lattice_with_electrical_and_chemical_synapses(iterations) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
        }
    }

    #[getter]
    fn get_history(&self) -> Vec<Vec<Vec<f32>>> {
        self.lattice.grid_history.history.clone()
    }

    #[getter]
    fn get_update_grid_history(&self) -> bool {
        self.lattice.update_grid_history
    }

    #[setter]
    fn set_update_grid_history(&mut self, flag: bool) {
        self.lattice.update_grid_history = flag;
    }

    #[getter]
    fn get_do_stdp(&self) -> bool {
        self.lattice.update_grid_history
    }

    #[setter]
    fn set_do_stdp(&mut self, flag: bool) {
        self.lattice.do_stdp = flag;
    }

    fn reset_history(&mut self) {
        self.lattice.grid_history.reset();
    }

    #[getter]
    fn get_weights(&self) -> Vec<Vec<f32>> {
        self.lattice.graph.matrix.clone()
            .into_iter()
            .map(|inner_vec| {
                inner_vec.into_iter()
                    .map(|opt| opt.unwrap_or(0.))
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        let rows = self.lattice.cell_grid.len();
        let cols = self.lattice.cell_grid.get(0).unwrap_or(&vec![]).len();

        Ok(
            format!(
                "IzhikevichLattice {{ ({}x{}), id: {}, do_stdp: {}, update_grid_history: {} }}", 
                rows,
                cols,
                self.lattice.get_id(),
                self.lattice.do_stdp,
                self.lattice.update_grid_history,
            )
        )
    }
}

type LatticeSpikeTrain = PoissonNeuron<ApproximateNeurotransmitter, DeltaDiracRefractoriness>;

#[pyclass]
#[pyo3(name = "PoissonLattice")]
#[derive(Clone)]
pub struct PyPoissonLattice {
    lattice: SpikeTrainLattice<
        LatticeSpikeTrain,
        SpikeTrainGridHistory
    >
}

#[pymethods]
impl PyPoissonLattice {
    #[new]
    fn new() -> Self {
        PyPoissonLattice { lattice: SpikeTrainLattice::default() }
    }

    fn populate(&mut self, neuron: PyPoissonNeuron, num_rows: usize, num_cols: usize) {
        self.lattice.populate(&neuron.model, num_rows, num_cols);
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.lattice.get_id()
    }

    #[setter]
    fn set_id(&mut self, id: usize) {
        self.lattice.set_id(id)
    }

    fn get_neuron(&self, row: usize, col: usize) -> PyResult<PyPoissonNeuron> {
        let neuron = match self.lattice.cell_grid.get(row) {
            Some(row_cells) => match row_cells.get(col) {
                Some(neuron) => neuron.clone(),
                None => {
                    return Err(PyKeyError::new_err(format!("Column at {} not found", col)));
                }
            },
            None => {
                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
            }
        };

        Ok(
            PyPoissonNeuron { 
                model: neuron
            }
        )
    }

    fn set_neuron(&mut self, row: usize, col: usize, neuron: PyPoissonNeuron) -> PyResult<()> {
        let row_cells = match self.lattice.cell_grid.get_mut(row) {
            Some(row_cells) => row_cells,
            None => {
                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
            }
        };

        if let Some(existing_neuron) = row_cells.get_mut(col) {
            *existing_neuron = neuron.model.clone();

            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("Column at {} not found", col)))
        }
    }

    fn reset_timing(&mut self) {
        self.lattice.reset_timing();
    }

    fn reset_history(&mut self) {
        self.lattice.grid_history.reset();
    }

    fn run_lattice(&mut self, iterations: usize) {
        self.lattice.run_lattice(iterations);
    }

    #[getter]
    fn get_history(&self) -> Vec<Vec<Vec<f32>>> {
        self.lattice.grid_history.history.clone()
    }

    #[getter]
    fn get_update_grid_history(&self) -> bool {
        self.lattice.update_grid_history
    }

    #[setter]
    fn set_update_grid_history(&mut self, flag: bool) {
        self.lattice.update_grid_history = flag;
    }

    fn __repr__(&self) -> PyResult<String> {
        let rows = self.lattice.cell_grid.len();
        let cols = self.lattice.cell_grid.get(0).unwrap_or(&vec![]).len();

        Ok(
            format!(
                "PoissonLattice {{ ({}x{}), id: {}, update_grid_history: {} }}", 
                rows,
                cols,
                self.lattice.get_id(),
                self.lattice.update_grid_history,
            )
        )
    }
}

#[pyclass]
#[pyo3(name = "GraphPosition")]
#[derive(Clone)]
pub struct PyGraphPosition {
    graph_position: GraphPosition
}

#[pymethods]
impl PyGraphPosition {
    #[new]
    fn new(id: usize, pos: (usize, usize)) -> PyGraphPosition {
        PyGraphPosition { graph_position: GraphPosition { id: id, pos: pos } }
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

#[pyclass]
#[pyo3(name = "IzhikevichNetwork")]
#[derive(Clone)]
pub struct PyIzhikevichNetwork {
    network: LatticeNetwork<
        LatticeNeuron, 
        AdjacencyMatrix<(usize, usize)>, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition>
    >
}

#[pymethods]
impl PyIzhikevichNetwork {
    #[new]
    fn new() -> Self {
        PyIzhikevichNetwork { network: LatticeNetwork::default() }
    }

    fn add_lattice(&mut self, lattice: PyIzhikevichLattice) -> PyResult<()> {
        match self.network.add_lattice(lattice.lattice) {
            Ok(_) => Ok(()),
            Err(_) => Err(PyValueError::new_err("Id already in network")),
        }
    }

    fn add_spike_train_lattice(&mut self, spike_train_lattice: PyPoissonLattice) -> PyResult<()> {
        match self.network.add_spike_train_lattice(spike_train_lattice.lattice) {
            Ok(_) => Ok(()),
            Err(_) => Err(PyValueError::new_err("Id already in network")),
        }
    }

    #[pyo3(signature = (id, connection_conditional, weight_logic=None))]
    fn connect_internally(
        &mut self, py: Python, id: usize, connection_conditional: &PyAny, weight_logic: Option<&PyAny>,
    ) -> PyResult<()> {
        let py_callable = connection_conditional.to_object(connection_conditional.py());

        let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> bool {
            let args = PyTuple::new(py, &[a, b]);
            py_callable.call1(py, args).unwrap().extract::<bool>(py).unwrap()
        };

        let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> f32>> = match weight_logic {
            Some(value) => {
                let py_callable = value.to_object(value.py()); 

                let closure = move |a: (usize, usize), b: (usize, usize)| -> f32 {
                    let args = PyTuple::new(py, &[a, b]);
                    py_callable.call1(py, args).unwrap().extract::<f32>(py).unwrap()
                };

                Some(Box::new(closure))
            },
            None => None,
        };

        match self.network.connect_innterally(id, &connection_closure, weight_closure.as_deref()) {
            Ok(_) => Ok(()),
            Err(_) => Err(PyValueError::new_err("Id not found in network")),
        }
    }

    #[pyo3(signature = (presynaptic_id, postsynaptic_id, connection_conditional, weight_logic=None))]
    fn connect(
        &mut self, 
        py: Python, 
        presynaptic_id: usize,  
        postsynaptic_id: usize, 
        connection_conditional: &PyAny, 
        weight_logic: Option<&PyAny>,
    ) -> PyResult<()> {
        let py_callable = connection_conditional.to_object(connection_conditional.py());

        let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> bool {
            let args = PyTuple::new(py, &[a, b]);
            py_callable.call1(py, args).unwrap().extract::<bool>(py).unwrap()
        };

        let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> f32>> = match weight_logic {
            Some(value) => {
                let py_callable = value.to_object(value.py()); 

                let closure = move |a: (usize, usize), b: (usize, usize)| -> f32 {
                    let args = PyTuple::new(py, &[a, b]);
                    py_callable.call1(py, args).unwrap().extract::<f32>(py).unwrap()
                };

                Some(Box::new(closure))
            },
            None => None,
        };

        match self.network.connect(
            presynaptic_id, 
            postsynaptic_id, 
            &connection_closure, 
            weight_closure.as_deref()
        ) {
            Ok(_) => Ok(()),
            Err(e) => match e {
                LatticeNetworkError::PresynapticIDNotFound(id) => Err(PyValueError::new_err(
                    format!("Presynaptic id ({}) not found", id)
                )),
                LatticeNetworkError::PostsynapticIDNotFound(id) => Err(PyValueError::new_err(
                    format!("Postsynaptic id ({}) not found", id)
                )),
                LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain => Err(PyValueError::new_err(
                    format!("Postsynaptic lattice cannot be spike train")
                )),
                _ => unreachable!(),
            },
        }
    }

    // #[getter]
    // fn get_connecting_weights(&self) -> Vec<Vec<f32>> {
    //     self.network.get_connecting_graph().matrix
    // }

    fn get_weight(&self, presynaptic: PyGraphPosition, postsynaptic: PyGraphPosition) -> PyResult<f32> {
        let presynaptic = presynaptic.graph_position;
        let postsynaptic = postsynaptic.graph_position;

        if presynaptic.id == postsynaptic.id {
            let current_lattice = match self.network.get_lattice(&presynaptic.id) {
                Some(lattice) => lattice,
                None => { return Err(PyValueError::new_err("Id not found in lattice")); },
            };
                
            match current_lattice.graph.lookup_weight(&presynaptic.pos, &postsynaptic.pos) {
                Ok(Some(value)) => Ok(value),
                Ok(None) => Ok(0.),
                Err(e) => Err(
                    PyValueError::new_err(format!("{}", e))
                )
            }
        } else {
            match self.network.get_connecting_graph().lookup_weight(&presynaptic, &postsynaptic) {
                Ok(Some(value)) => Ok(value),
                Ok(None) => Ok(0.),
                Err(e) => Err(
                    PyValueError::new_err(format!("{}", e))
                )
            }
        }
    }

    // fn set_weight(&mut self, presynaptic: PyGraphPosition, postsynaptic: PyGraphPosition, weight: f32) -> PyResult<()> {
    //     let presynaptic = presynaptic.graph_position;
    //     let postsynaptic = postsynaptic.graph_position;

    //     let weight = match weight {
    //         0. => None,
    //         value => Some(value)
    //     };

    //     if presynaptic.id == postsynaptic.id {
    //         let current_lattice = match self.network.get_mut_lattice(&presynaptic.id) {
    //             Some(lattice) => lattice,
    //             None => { return Err(PyValueError::new_err("Id not found in lattice")); },
    //         };
                
    //         match current_lattice.graph.edit_weight(&presynaptic.pos, &postsynaptic.pos, weight) {
    //             Ok(_) => Ok(()),
    //             Err(e) => Err(
    //                 PyValueError::new_err(format!("{}", e))
    //             )
    //         }
    //     } else {
    //         match self.network.get_connecting_graph().edit_weight(&presynaptic, &postsynaptic, weight) {
    //             Ok(_) => Ok(()),
    //             Err(e) => Err(
    //                 PyValueError::new_err(format!("{}", e))
    //             )
    //         }
    //     }
    // }

    fn get_neuron(&self, id: usize, row: usize, col: usize) -> PyResult<PyIzhikevichNeuron> {
        match self.network.get_lattice(&id) {
            Some(lattice) => {
                let neuron = match lattice.cell_grid.get(row) {
                    Some(row_cells) => match row_cells.get(col) {
                        Some(neuron) => neuron.clone(),
                        None => {
                            return Err(PyKeyError::new_err(format!("Column at {} not found", col)));
                        }
                    },
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };
        
                Ok(
                    PyIzhikevichNeuron { 
                        model: neuron
                    }
                )
            },
            None => Err(PyValueError::new_err("Id not found")),
        }
    }

    fn set_neuron(&mut self, id: usize, row: usize, col: usize, neuron: PyLatticeNeuron) -> PyResult<()> {
        match self.network.get_mut_lattice(&id) {
            Some(lattice) => {
                let row_cells = match lattice.cell_grid.get_mut(row) {
                    Some(row_cells) => row_cells,
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };
        
                if let Some(existing_neuron) = row_cells.get_mut(col) {
                    *existing_neuron = neuron.model.clone();
        
                    Ok(())
                } else {
                    Err(PyKeyError::new_err(format!("Column at {} not found", col)))
                }
            },
            None => Err(PyValueError::new_err("Id not found")),
        }
    }

    fn get_spike_train(&self, id: usize, row: usize, col: usize) -> PyResult<PyPoissonNeuron> {
        match self.network.get_spike_train_lattice(&id) {
            Some(lattice) => {
                let neuron = match lattice.cell_grid.get(row) {
                    Some(row_cells) => match row_cells.get(col) {
                        Some(neuron) => neuron.clone(),
                        None => {
                            return Err(PyKeyError::new_err(format!("Column at {} not found", col)));
                        }
                    },
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };
        
                Ok(
                    PyPoissonNeuron { 
                        model: neuron
                    }
                )
            },
            None => Err(PyValueError::new_err("Id not found")),
        }
    }

    fn set_spike_train(&mut self, id: usize, row: usize, col: usize, neuron: PyPoissonNeuron) -> PyResult<()> {
        match self.network.get_mut_spike_train_lattice(&id) {
            Some(lattice) => {
                let row_cells = match lattice.cell_grid.get_mut(row) {
                    Some(row_cells) => row_cells,
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };
        
                if let Some(existing_neuron) = row_cells.get_mut(col) {
                    *existing_neuron = neuron.model.clone();
        
                    Ok(())
                } else {
                    Err(PyKeyError::new_err(format!("Column at {} not found", col)))
                }
            },
            None => Err(PyValueError::new_err("Id not found")),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let lattice_strings = self.network.lattices_values()
            .map(|i| {
                let rows = i.cell_grid.len();
                let cols = i.cell_grid.get(0).unwrap_or(&vec![]).len();

                format!(
                    "IzhikevichLattice {{ ({}x{}), id: {}, do_stdp: {}, update_grid_history: {} }}", 
                    rows,
                    cols,
                    i.get_id(),
                    i.do_stdp,
                    i.update_grid_history,
                )
            })
            .collect::<Vec<String>>()
            .join("\n");

        let spike_train_strings = self.network.spike_trains_values()
            .map(|i| {
                let rows = i.cell_grid.len();
                let cols = i.cell_grid.get(0).unwrap_or(&vec![]).len();

                format!(
                    "PoissonLattice {{ ({}x{}), id: {}, update_grid_history: {} }}", 
                    rows,
                    cols,
                    i.get_id(),
                    i.update_grid_history,
                )
            })
            .collect::<Vec<String>>()
            .join(",\n");

        Ok(format!("IzhikevichNetwork {{ \n[{}],\n[{}], }}", lattice_strings, spike_train_strings))
    }
}

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeurotransmitterType>()?;
    m.add_class::<PyApproximateNeurotransmitter>()?;
    m.add_class::<PyApproximateNeurotransmitters>()?;
    m.add_class::<PyApproximateLigandGatedChannel>()?;
    m.add_class::<PyApproximateLigandGatedChannels>()?;
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PyIzhikevichLattice>()?;
    m.add_class::<PyDeltaDiracRefractoriness>()?;
    m.add_class::<PyPoissonNeuron>()?;
    m.add_class::<PyPoissonLattice>()?;
    m.add_class::<PyIzhikevichNetwork>()?;

    // RUN LATTICE METHODS
    
    // view weights
    // eventually work with graph history
    // connecting graph history should be updated
    
    // in python wrapper for pyo3, connect conditional errors could be caught and made more readable

    // temp env variable for building pyo3 with custom models
    // impl neuron macro for arbitrary neuron (separate one for neurons with ion channels)

    Ok(())
}
