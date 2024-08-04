use std::{collections::{hash_map::DefaultHasher, HashMap, HashSet}, hash::{Hash, Hasher}};
use pyo3::{exceptions::{PyKeyError, PyValueError}, prelude::*, types::{PyDict, PyList, PyTuple}};
use spiking_neural_networks::{
    error::LatticeNetworkError, graph::{AdjacencyMatrix, Graph, GraphPosition}, neuron::{
    hodgkin_huxley::HodgkinHuxleyNeuron, integrate_and_fire::IzhikevichNeuron, ion_channels::{BasicGatingVariable, IonChannel, 
        KIonChannel, KLeakChannel, NaIonChannel, TimestepIndependentIonChannel
    }, iterate_and_spike::{
        AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, DestexheNeurotransmitter, 
        DestexheReceptor, GABAaDefault, GABAbDefault, IterateAndSpike, LastFiringTime, 
        LigandGatedChannel, LigandGatedChannels, 
        NMDADefault, NeurotransmitterConcentrations, NeurotransmitterKinetics, 
        NeurotransmitterType, Neurotransmitters, ReceptorKinetics 
    }, plasticity::STDP, spike_train::{
        DeltaDiracRefractoriness, NeuralRefractoriness, PoissonNeuron, SpikeTrain
    }, GridVoltageHistory, Lattice, LatticeHistory, LatticeNetwork, SpikeTrainGridHistory, SpikeTrainLattice, SpikeTrainLatticeHistory
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

    fn apply_t_change(&mut self, voltage: f32) {
        self.neurotransmitter.apply_t_change(voltage);
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

#[pyclass]
#[pyo3(name = "ApproximateReceptor")]
#[derive(Clone)]
pub struct PyApproximateReceptor {
    receptor: ApproximateReceptor
}

implement_basic_getter_and_setter!(
    PyApproximateReceptor, 
    receptor,
    r, get_r, set_r 
);

#[pymethods]
impl PyApproximateReceptor {
    #[new]
    #[pyo3(signature = (r=0.))]
    fn new(r: f32) -> Self {
        PyApproximateReceptor {
            receptor: ApproximateReceptor {
                r: r
            }
        }
    }

    fn apply_r_change(&mut self, neurotransmitter_conc: f32) {
        self.receptor.apply_r_change(neurotransmitter_conc);
    }
}

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

    fn get_receptor(&self) -> PyApproximateReceptor {
        PyApproximateReceptor { receptor: self.ligand_gate.receptor }
    }

    fn set_receptor(&mut self, receptor: PyApproximateReceptor) {
        self.ligand_gate.receptor = receptor.receptor;
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

macro_rules! impl_default_neuron_methods {
    ($neuron_kind:ident, $neurotransmitter_kind:ident, $ligand_gates_kind:ident) => {
        #[pymethods]
        impl $neuron_kind {
            fn iterate_and_spike(&mut self, i: f32) -> bool {
                self.model.iterate_and_spike(i)
            }
        
            #[pyo3(signature = (i, neurotransmitter_concs))]
            fn iterate_with_neurotransmitter_and_spike(&mut self, i: f32, neurotransmitter_concs: &PyDict) -> PyResult<bool> {
                let neurotransmitter_concs = pydict_to_neurotransmitters_concentration(neurotransmitter_concs)?;
        
                Ok(self.model.iterate_with_neurotransmitter_and_spike(i, &neurotransmitter_concs))
            }
        
            fn get_neurotransmitters(&self) -> $neurotransmitter_kind {
                $neurotransmitter_kind { neurotransmitters: self.model.synaptic_neurotransmitters.clone() }
            }
        
            fn set_neurotransmitters(&mut self, neurotransmitters: $neurotransmitter_kind) {
                self.model.synaptic_neurotransmitters = neurotransmitters.neurotransmitters;
            }
        
            fn get_ligand_gates(&self) -> $ligand_gates_kind {
                $ligand_gates_kind { ligand_gates: self.model.ligand_gates.clone() }
            }
        
            fn set_ligand_gates(&mut self, ligand_gates: $ligand_gates_kind) {
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
        
            #[getter]
            fn is_spiking(&self) -> bool {
                self.model.is_spiking
            }
        }
    };
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
impl_repr!(PyIzhikevichNeuron, model);
impl_default_neuron_methods!(
    PyIzhikevichNeuron, 
    PyApproximateNeurotransmitters, 
    PyApproximateLigandGatedChannels
);

#[pymethods]
impl PyIzhikevichNeuron {
    #[new]
    #[pyo3(signature = (
        a=0.02, b=0.2, c=-55., d=8., v_th=30., dt=0.1, current_voltage=-65., 
        v_init=-65., w_value=30., w_init=30., gap_conductance=10., tau_m=1., c_m=100.,
        synaptic_neurotransmitters=PyApproximateNeurotransmitters { 
            neurotransmitters: Neurotransmitters::<ApproximateNeurotransmitter>::default() 
        },
        ligand_gates=PyApproximateLigandGatedChannels {
            ligand_gates: LigandGatedChannels::<ApproximateReceptor>::default()
        }
    ))]
    fn new(
        a: f32, b: f32, c: f32, d: f32, v_th: f32, dt: f32, current_voltage: f32, v_init: f32, 
        w_value: f32, w_init: f32, gap_conductance: f32, tau_m: f32, c_m: f32,
        synaptic_neurotransmitters: PyApproximateNeurotransmitters, ligand_gates: PyApproximateLigandGatedChannels,
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
                synaptic_neurotransmitters: synaptic_neurotransmitters.neurotransmitters,
                ligand_gates: ligand_gates.ligand_gates,
                ..IzhikevichNeuron::default()
            }
        }
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

#[pyclass]
#[pyo3(name = "STDP")]
#[derive(Clone)]
pub struct PySTDP {
    plasticity: STDP
}

implement_basic_getter_and_setter!(
    PySTDP,
    plasticity,
    a_plus, get_a_plus, set_a_plus,
    a_minus, get_a_minus, set_a_minus,
    tau_plus, get_tau_plus, set_tau_plus,
    tau_minus, get_tau_minus, set_tau_minus,
    dt, get_dt, set_dt
);

macro_rules! impl_lattice {
    ($lattice_kind:ident, $lattice_neuron:ident, $name:literal, $plasticity_kind:ident) => {
        #[pymethods]
        impl $lattice_kind {
            #[new]
            #[pyo3(signature = (id=0))]
            fn new(id: usize) -> Self {
                let mut lattice = $lattice_kind { lattice: Lattice::default() };
                lattice.set_id(id);

                lattice
            }

            fn populate(&mut self, neuron: $lattice_neuron, num_rows: usize, num_cols: usize) {
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

            fn get_neuron(&self, row: usize, col: usize) -> PyResult<$lattice_neuron> {
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
                    $lattice_neuron { 
                        model: neuron
                    }
                )
            }

            fn set_neuron(&mut self, row: usize, col: usize, neuron: $lattice_neuron) -> PyResult<()> {
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

            fn get_incoming_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.graph.get_incoming_connections(&position) {
                    Ok(value) => Ok(value),
                    Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                }
            }

            fn get_outgoing_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.graph.get_outgoing_connections(&position) {
                    Ok(value) => Ok(value),
                    Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                }
            }

            fn apply(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                self.lattice.apply(|neuron| {
                    let py_neuron = $lattice_neuron {
                        model: neuron.clone(),
                    };
                    let result = py_callable.call1(py, (py_neuron,)).unwrap();
                    let updated_py_neuron: $lattice_neuron = result.extract(py).unwrap();
                    *neuron = updated_py_neuron.model;
                });

                Ok(())
            }

            fn apply_function_given_position(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                self.lattice.apply_given_position(|(i, j), neuron| {
                    let py_neuron = $lattice_neuron {
                        model: neuron.clone(),
                    };
                    let result = py_callable.call1(py, ((i, j), py_neuron,)).unwrap();
                    let updated_py_neuron: $lattice_neuron = result.extract(py).unwrap();
                    *neuron = updated_py_neuron.model;
                });

                Ok(())
            }

            fn reset_timing(&mut self) {
                self.lattice.reset_timing();
            }

            fn run_lattice(&mut self, iterations: usize) -> PyResult<()> {
                match self.lattice.run_lattice(iterations) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
                }
            }

            fn run_lattice_chemical_synapses_only(&mut self, iterations: usize) -> PyResult<()> {
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
            fn get_plasticity(&self) -> $plasticity_kind {
                $plasticity_kind { plasticity: self.lattice.plasticity }
            }

            #[setter]
            fn set_plasticity(&mut self, plasticity: $plasticity_kind) {
                self.lattice.plasticity = plasticity.plasticity
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
            fn get_gaussian(&self) -> bool {
                self.lattice.gaussian
            }

            #[setter]
            fn set_gaussian(&mut self, flag: bool) {
                self.lattice.gaussian = flag;
            }

            #[getter]
            fn weights_history(&self) -> Vec<Vec<Vec<f32>>> {
                self.lattice.graph.history.clone()
                    .iter()
                    .map(|grid| {
                        grid.iter()
                            .map(|row| {
                                row.iter().map(|i| {
                                    i.unwrap_or(0.)
                                })
                                .collect()
                            })
                            .collect()
                    })
                    .collect()
            }

            #[getter]
            fn get_update_graph_history(&self) -> bool {
                self.lattice.update_graph_history
            }

            #[setter]
            fn set_update_graph_history(&mut self, flag: bool) {
                self.lattice.update_graph_history = flag;
            }

            #[getter]
            fn get_do_plasticity(&self) -> bool {
                self.lattice.update_grid_history
            }

            #[setter]
            fn set_do_plasticity(&mut self, flag: bool) {
                self.lattice.do_plasticity = flag;
            }

            fn reset_history(&mut self) {
                self.lattice.grid_history.reset();
            }

            #[getter(weights)]
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

            #[getter(position_to_index)]
            fn get_position_to_index_for_weights(&self) -> HashMap<(usize, usize), usize> {
                self.lattice.graph.position_to_index.clone()
            }

            fn __repr__(&self) -> PyResult<String> {
                let rows = self.lattice.cell_grid.len();
                let cols = self.lattice.cell_grid.get(0).unwrap_or(&vec![]).len();

                Ok(
                    format!(
                        "{} {{ ({}x{}), id: {}, do_plasticity: {}, update_grid_history: {} }}", 
                        $name,
                        rows,
                        cols,
                        self.lattice.get_id(),
                        self.lattice.do_plasticity,
                        self.lattice.update_grid_history,
                    )
                )
            }
        }
    };
}

#[pyclass]
#[pyo3(name = "IzhikevichLattice")]
#[derive(Clone)]
pub struct PyIzhikevichLattice {
    lattice: Lattice<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        GridVoltageHistory,
        STDP,
    >
}

impl_lattice!(PyIzhikevichLattice, PyIzhikevichNeuron, "IzhikevichLattice", PySTDP);

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
    #[pyo3(signature = (id=0))]
    fn new(id: usize) -> Self {
        let mut lattice = PyPoissonLattice { lattice: SpikeTrainLattice::default() };

        lattice.set_id(id);

        lattice
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
    
    fn apply_function(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
        let py_callable = function.to_object(py);

        self.lattice.apply(|neuron| {
            let py_neuron = PyPoissonNeuron {
                model: neuron.clone(),
            };
            let result = py_callable.call1(py, (py_neuron,)).unwrap();
            let updated_py_neuron: PyPoissonNeuron = result.extract(py).unwrap();
            *neuron = updated_py_neuron.model;
        });

        Ok(())
    }

    fn apply_function_given_position(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
        let py_callable = function.to_object(py);

        self.lattice.apply_given_position(|(i, j), neuron| {
            let py_neuron = PyPoissonNeuron {
                model: neuron.clone(),
            };
            let result = py_callable.call1(py, ((i, j), py_neuron,)).unwrap();
            let updated_py_neuron: PyPoissonNeuron = result.extract(py).unwrap();
            *neuron = updated_py_neuron.model;
        });

        Ok(())
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
#[derive(Clone, Hash, PartialEq, Eq)]
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

macro_rules! impl_network {
    (
        $network_kind:ident, $lattice_kind:ident, $spike_train_lattice_kind:ident,
        $lattice_neuron_kind:ident, $spike_train_kind:ident, $plasticity_kind:ident,
        $lattice_neuron_name:literal, $spike_train_name:literal, $network_name:literal,
    ) => {        
        #[pymethods]
        impl $network_kind {
            #[new]
            fn new() -> Self {
                $network_kind { network: LatticeNetwork::default() }
            }

            fn add_lattice(&mut self, lattice: $lattice_kind) -> PyResult<()> {
                let id = lattice.lattice.get_id();

                match self.network.add_lattice(lattice.lattice) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(
                        PyKeyError::new_err(
                            format!("Id ({}) already in network", id)
                        )
                    ),
                }
            }

            fn add_spike_train_lattice(&mut self, spike_train_lattice: $spike_train_lattice_kind) -> PyResult<()> {
                let id = spike_train_lattice.lattice.get_id();
                
                match self.network.add_spike_train_lattice(spike_train_lattice.lattice) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(
                        PyKeyError::new_err(
                            format!("Id ({}) already in network", id)
                        )
                    ),
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

                match self.network.connect_interally(id, &connection_closure, weight_closure.as_deref()) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(PyKeyError::new_err("Id not found in network")),
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
                        LatticeNetworkError::PresynapticIDNotFound(id) => Err(PyKeyError::new_err(
                            format!("Presynaptic id ({}) not found", id)
                        )),
                        LatticeNetworkError::PostsynapticIDNotFound(id) => Err(PyKeyError::new_err(
                            format!("Postsynaptic id ({}) not found", id)
                        )),
                        LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain => Err(PyValueError::new_err(
                            format!("Postsynaptic lattice cannot be spike train")
                        )),
                        _ => unreachable!(),
                    },
                }
            }

            #[getter(connecting_weights)]
            fn get_connecting_weights(&self) -> Vec<Vec<f32>> {
                self.network.get_connecting_graph().matrix
                    .iter()
                    .map(|row|
                        row.iter()
                            .map(|i| i.unwrap_or(0.) )
                            .collect::<Vec<f32>>()
                    )
                    .collect()
            }

            #[getter(connecting_position_to_index)]
            fn get_connecting_position_to_index(&self) -> HashMap<PyGraphPosition, usize> {
                self.network.get_connecting_graph().position_to_index
                    .iter()
                    .map(|(key, value)| 
                        (PyGraphPosition { graph_position: *key }, *value)
                    )
                    .collect()
            }           

            fn get_weight(&self, presynaptic: PyGraphPosition, postsynaptic: PyGraphPosition) -> PyResult<f32> {
                let presynaptic = presynaptic.graph_position;
                let postsynaptic = postsynaptic.graph_position;

                if presynaptic.id == postsynaptic.id {
                    let current_lattice = match self.network.get_lattice(&presynaptic.id) {
                        Some(lattice) => lattice,
                        None => { return Err(PyKeyError::new_err("Id not found in lattice")); },
                    };
                        
                    match current_lattice.graph.lookup_weight(&presynaptic.pos, &postsynaptic.pos) {
                        Ok(Some(value)) => Ok(value),
                        Ok(None) => Ok(0.),
                        Err(e) => Err(
                            PyKeyError::new_err(format!("{}", e))
                        )
                    }
                } else {
                    match self.network.get_connecting_graph().lookup_weight(&presynaptic, &postsynaptic) {
                        Ok(Some(value)) => Ok(value),
                        Ok(None) => Ok(0.),
                        Err(e) => Err(
                            PyKeyError::new_err(format!("{}", e))
                        )
                    }
                }
            }

            fn get_incoming_connections_within_lattice(&self, id: usize, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.network.get_lattice(&id) {
                    Some(lattice) => {
                        match lattice.graph.get_incoming_connections(&position) {
                            Ok(value) => Ok(value),
                            Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                        }
                    },
                    None => {
                        Err(PyKeyError::new_err(format!("Lattice {} not found in network", id)))
                    }
                }
            }

            fn get_outgoing_connections_within_lattice(&self, id: usize, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.network.get_lattice(&id) {
                    Some(lattice) => {
                        match lattice.graph.get_outgoing_connections(&position) {
                            Ok(value) => Ok(value),
                            Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                        }
                    },
                    None => {
                        Err(PyKeyError::new_err(format!("Lattice {} not found in network", id)))
                    }
                }
            }

            fn get_incoming_connectings_across_lattices(&self, id: usize, position: (usize, usize)) -> PyResult<HashSet<PyGraphPosition>> {
                match self.network.get_lattice(&id) {
                    Some(_) => {
                        let graph_pos = GraphPosition { id: id, pos: position };
                        let connections = self.network.get_connecting_graph().get_incoming_connections(&graph_pos);

                        match connections {
                            Ok(connections_value) => {
                                Ok(
                                    connections_value.iter()
                                        .map(|i| PyGraphPosition { graph_position: *i })
                                        .collect()
                                )
                            },
                            Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                        }
                    },
                    None => {
                        Err(PyKeyError::new_err(format!("Lattice {} not found in network", id)))
                    }
                }
            }

            fn get_outgoing_connectings_across_lattices(&self, id: usize, position: (usize, usize)) -> PyResult<HashSet<PyGraphPosition>> {
                match self.network.get_lattice(&id) {
                    Some(_) => {
                        let graph_pos = GraphPosition { id: id, pos: position };
                        let connections = self.network.get_connecting_graph().get_outgoing_connections(&graph_pos);

                        match connections {
                            Ok(connections_value) => {
                                Ok(
                                    connections_value.iter()
                                        .map(|i| PyGraphPosition { graph_position: *i })
                                        .collect()
                                )
                            },
                            Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                        }
                    },
                    None => {
                        Err(PyKeyError::new_err(format!("Lattice {} not found in network", id)))
                    }
                }
            }

            fn get_neuron(&self, id: usize, row: usize, col: usize) -> PyResult<$lattice_neuron_kind> {
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
                            $lattice_neuron_kind { 
                                model: neuron
                            }
                        )
                    },
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            }

            fn set_neuron(&mut self, id: usize, row: usize, col: usize, neuron: $lattice_neuron_kind) -> PyResult<()> {
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
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            }

            fn get_spike_train(&self, id: usize, row: usize, col: usize) -> PyResult<$spike_train_kind> {
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
                            $spike_train_kind { 
                                model: neuron
                            }
                        )
                    },
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            }

            fn set_spike_train(&mut self, id: usize, row: usize, col: usize, neuron: $spike_train_kind) -> PyResult<()> {
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
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            }

            fn get_lattice(&self, id: usize) -> PyResult<$lattice_kind> {
                match self.network.get_lattice(&id) {
                    Some(value) => Ok($lattice_kind { lattice: value.clone() }),
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            } 

            fn get_spike_train_lattice(&self, id: usize) -> PyResult<$spike_train_lattice_kind> {
                match self.network.get_spike_train_lattice(&id) {
                    Some(value) => Ok($spike_train_lattice_kind { lattice: value.clone() }),
                    None => Err(PyKeyError::new_err("Id not found")),
                }
            } 

            fn set_lattice(&mut self, id: usize, lattice: $lattice_kind) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    *current_lattice = lattice.lattice.clone();

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }
            
            fn set_spike_train_lattice(&mut self, id: usize, lattice: $spike_train_lattice_kind) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                    *current_lattice = lattice.lattice.clone();

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }

            fn get_do_plasticity(&self, id: usize) -> PyResult<bool> {
                if let Some(current_lattice) = self.network.get_lattice(&id) {
                    Ok(current_lattice.do_plasticity)
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn set_do_plasticity(&mut self, id: usize, flag: bool) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.do_plasticity = flag;

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn get_gaussian(&self, id: usize) -> PyResult<bool> {
                if let Some(current_lattice) = self.network.get_lattice(&id) {
                    Ok(current_lattice.gaussian)
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn set_gaussian(&mut self, id: usize, flag: bool) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.gaussian = flag;

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn get_plasticity(&self, id: usize) -> PyResult<$plasticity_kind> {
                if let Some(current_lattice) = self.network.get_lattice(&id) {
                    Ok($plasticity_kind { plasticity: current_lattice.plasticity })
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn set_plasticity(&mut self, id: usize, plasticity: $plasticity_kind) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.plasticity = plasticity.plasticity;

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn reset_timing(&mut self, id: usize) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.reset_timing();

                    Ok(())
                } else {
                    if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                        current_lattice.reset_timing();

                        Ok(())
                    } else {
                        Err(PyKeyError::new_err("Id not found"))
                    }
                }
            }

            fn get_update_grid_history(&self, id: usize) -> PyResult<bool> {
                if let Some(current_lattice) = self.network.get_lattice(&id) {
                    Ok(current_lattice.update_grid_history)
                } else {
                    if let Some(current_lattice) = self.network.get_spike_train_lattice(&id) {
                        Ok(current_lattice.update_grid_history)
                    } else {
                        Err(PyKeyError::new_err("Id not found"))
                    }
                }
            }

            fn set_update_grid_history(&mut self, id: usize, flag: bool) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.update_grid_history = flag;

                    Ok(())
                } else {
                    if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                        current_lattice.update_grid_history = flag;

                        Ok(())
                    } else {
                        Err(PyKeyError::new_err("Id not found"))
                    }
                }
            }

            fn reset_history(&mut self, id: usize) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.grid_history.reset();

                    Ok(())
                } else {
                    if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                        current_lattice.grid_history.reset();

                        Ok(())
                    } else {
                        Err(PyKeyError::new_err("Id not found"))
                    }
                }
            }

            fn get_update_graph_history(&self, id: usize) -> PyResult<bool> {
                if let Some(current_lattice) = self.network.get_lattice(&id) {
                    Ok(current_lattice.update_graph_history)
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            fn set_update_graph_history(&mut self, id: usize, flag: bool) -> PyResult<()> {
                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.update_graph_history = flag;

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found (in non spike train lattices)"))
                }
            }

            #[getter]
            fn get_connecting_graph_history(&self) -> Vec<Vec<Vec<f32>>> {
                self.network.get_connecting_graph().history.clone()
                    .iter()
                    .map(|grid| {
                        grid.iter()
                            .map(|row| {
                                row.iter().map(|i| {
                                    i.unwrap_or(0.)
                                })
                                .collect()
                            })
                            .collect()
                    })
                    .collect()
            }

            fn apply_lattice(&mut self, py: Python, id: usize, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.apply(|neuron| {
                        let py_neuron = $lattice_neuron_kind {
                            model: neuron.clone(),
                        };
                        let result = py_callable.call1(py, (py_neuron,)).unwrap();
                        let updated_py_neuron: $lattice_neuron_kind = result.extract(py).unwrap();
                        *neuron = updated_py_neuron.model;
                    });

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }

            fn apply_spike_train_lattice(&mut self, py: Python, id: usize, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                    current_lattice.apply(|neuron| {
                        let py_neuron = $spike_train_kind {
                            model: neuron.clone(),
                        };
                        let result = py_callable.call1(py, (py_neuron,)).unwrap();
                        let updated_py_neuron: $spike_train_kind = result.extract(py).unwrap();
                        *neuron = updated_py_neuron.model;
                    });

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }

            fn apply_lattice_given_position(&mut self, py: Python, id: usize, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                if let Some(current_lattice) = self.network.get_mut_lattice(&id) {
                    current_lattice.apply_given_position(|(i, j), neuron| {
                        let py_neuron = $lattice_neuron_kind {
                            model: neuron.clone(),
                        };
                        let result = py_callable.call1(py, ((i, j), py_neuron,)).unwrap();
                        let updated_py_neuron: $lattice_neuron_kind = result.extract(py).unwrap();
                        *neuron = updated_py_neuron.model;
                    });

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }

            fn apply_spike_train_lattice_given_position(&mut self, py: Python, id: usize, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);

                if let Some(current_lattice) = self.network.get_mut_spike_train_lattice(&id) {
                    current_lattice.apply_given_position(|(i, j), neuron| {
                        let py_neuron = $spike_train_kind {
                            model: neuron.clone(),
                        };
                        let result = py_callable.call1(py, ((i, j), py_neuron,)).unwrap();
                        let updated_py_neuron: $spike_train_kind = result.extract(py).unwrap();
                        *neuron = updated_py_neuron.model;
                    });

                    Ok(())
                } else {
                    Err(PyKeyError::new_err("Id not found"))
                }
            }

            fn run_lattices(&mut self, iterations: usize) -> PyResult<()> {
                match self.network.run_lattices(iterations) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
                }
            }

            fn run_lattices_chemical_synapses_only(&mut self, iterations: usize) -> PyResult<()> {
                match self.network.run_lattices_chemical_synapses_only(iterations) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
                }
            }

            fn run_lattices_chemical_and_electrical_synapses(&mut self, iterations: usize) -> PyResult<()> {
                match self.network.run_lattices_with_electrical_and_chemical_synapses(iterations) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyKeyError::new_err(format!("Graph error occured in execution: {:#?}", e)))
                }
            }

            fn __repr__(&self) -> PyResult<String> {
                let lattice_strings = self.network.lattices_values()
                    .map(|i| {
                        let rows = i.cell_grid.len();
                        let cols = i.cell_grid.get(0).unwrap_or(&vec![]).len();

                        format!(
                            "{} {{ ({}x{}), id: {}, do_plasticity: {}, update_grid_history: {} }}", 
                            $lattice_neuron_name,
                            rows,
                            cols,
                            i.get_id(),
                            i.do_plasticity,
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
                            "{} {{ ({}x{}), id: {}, update_grid_history: {} }}", 
                            $spike_train_name,
                            rows,
                            cols,
                            i.get_id(),
                            i.update_grid_history,
                        )
                    })
                    .collect::<Vec<String>>()
                    .join(",");

                Ok(format!("{} {{ \n    [{}],\n    [{}],\n}}", $network_name, lattice_strings, spike_train_strings))
            }
        }
    };
}

#[pyclass]
#[pyo3(name = "IzhikevichNetwork")]
#[derive(Clone)]
pub struct PyIzhikevichNetwork {
    network: LatticeNetwork<
        IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
        AdjacencyMatrix<(usize, usize), f32>, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition, f32>,
        STDP,
    >
}

impl_network!(
    PyIzhikevichNetwork, PyIzhikevichLattice, PyPoissonLattice, PyIzhikevichNeuron,
    PyPoissonNeuron, PySTDP, "IzhikevichLattice", "PoissonLattice", "IzhikevichNetwork",
);

#[pyclass]
#[pyo3(name = "DestexheNeurotransmitter")]
#[derive(Clone)]
pub struct PyDestexheNeurotransmitter {
    neurotransmitter: DestexheNeurotransmitter
}

implement_basic_getter_and_setter!(
    PyDestexheNeurotransmitter,
    neurotransmitter,
    t_max, get_t_max, set_t_max,
    t, get_t, set_t,
    v_p, get_v_p, set_v_p,
    k_p, get_k_p, set_k_p
);

#[pymethods]
impl PyDestexheNeurotransmitter {
    #[new]
    #[pyo3(signature = (t_max=1., t=0., v_p=5.0, k_p=2.0))]
    fn new(t_max: f32, t: f32, v_p: f32, k_p: f32) -> Self {
        PyDestexheNeurotransmitter {
            neurotransmitter: DestexheNeurotransmitter {
                t_max: t_max,
                t: t,
                v_p: v_p,
                k_p: k_p,
            }
        }
    }

    fn apply_t_change(&mut self, voltage: f32) {
        self.neurotransmitter.apply_t_change(voltage);
    }
}

impl_repr!(PyDestexheNeurotransmitter, neurotransmitter);

#[pyclass]
#[pyo3(name = "DestexheNeurotransmitters")]
#[derive(Clone)]
pub struct PyDestexheNeurotransmitters {
    neurotransmitters: Neurotransmitters<DestexheNeurotransmitter>
}

impl_repr!(PyDestexheNeurotransmitters, neurotransmitters);

#[pymethods]
impl PyDestexheNeurotransmitters {
    #[new]
    #[pyo3(signature = (neurotransmitter_types=None))]
    fn new(neurotransmitter_types: Option<&PyList>) -> PyResult<Self> {
        let mut neurotransmitters: HashMap<NeurotransmitterType, DestexheNeurotransmitter> = HashMap::new();

        match neurotransmitter_types {
            Some(values) => {
                for i in values.iter() {
                    let current_type = i.extract::<PyNeurotransmitterType>()?.convert_type();
                    let neurotransmitter = match current_type {
                        NeurotransmitterType::Basic => DestexheNeurotransmitter::default(),
                        NeurotransmitterType::AMPA => DestexheNeurotransmitter::ampa_default(),
                        NeurotransmitterType::GABAa => DestexheNeurotransmitter::gabaa_default(),
                        NeurotransmitterType::GABAb => DestexheNeurotransmitter::gabab_default(),
                        NeurotransmitterType::NMDA => DestexheNeurotransmitter::nmda_default(),
                    };
        
                    neurotransmitters.insert(current_type, neurotransmitter);
                }
            },
            None => {}
        };

        Ok(
            PyDestexheNeurotransmitters {
                neurotransmitters: Neurotransmitters { neurotransmitters: neurotransmitters }
            }
        )
    }

    fn __getitem__(&self, neurotransmitter_type: PyNeurotransmitterType) -> PyResult<PyDestexheNeurotransmitter> {
        if let Some(value) = self.neurotransmitters.get(&neurotransmitter_type.convert_type()) {
            Ok(
                PyDestexheNeurotransmitter { 
                    neurotransmitter: *value 
                }
            )
        } else {
            Err(PyKeyError::new_err(format!("{:#?} not found", neurotransmitter_type)))
        }
    }

    fn set_neurotransmitter(
        &mut self, neurotransmitter_type: PyNeurotransmitterType, neurotransmitter: PyDestexheNeurotransmitter
    ) {
        self.neurotransmitters.neurotransmitters.insert(
            neurotransmitter_type.convert_type(), neurotransmitter.neurotransmitter
        );
    }

    fn apply_t_changes(&mut self, voltage: f32) {
        self.neurotransmitters.apply_t_changes(voltage);
    }
}

#[pyclass]
#[pyo3(name = "DestexheReceptor")]
#[derive(Clone)]
pub struct PyDestexheReceptor {
    receptor: DestexheReceptor
}

implement_basic_getter_and_setter!(
    PyDestexheReceptor,
    receptor,
    r, get_r, set_r,
    alpha, get_alpha, set_alpha,
    beta, get_beta, set_beta,
    dt, get_dt, set_dt
);

#[pymethods]
impl PyDestexheReceptor {
    #[new]
    #[pyo3(signature = (r=1., alpha=1., beta=1.0, dt=0.1))]
    fn new(r: f32, alpha: f32, beta: f32, dt: f32) -> Self {
        PyDestexheReceptor { 
            receptor: DestexheReceptor {
                r: r,
                alpha: alpha,
                beta: beta,
                dt: dt,
            } 
        }
    }

    fn apply_r_change(&mut self, neurotransmitter_conc: f32) {
        self.receptor.apply_r_change(neurotransmitter_conc);
    }
}

#[pyclass]
#[pyo3(name = "DestexheLigandGatedChannel")]
#[derive(Clone)]
pub struct PyDestexheLigandGatedChannel {
    ligand_gate: LigandGatedChannel<DestexheReceptor>
}

implement_basic_getter_and_setter!(
    PyDestexheLigandGatedChannel, 
    ligand_gate,
    g, get_g, set_g,
    reversal, get_reversal, set_reversal,
    current, get_current, set_current
);

impl_repr!(PyDestexheLigandGatedChannel, ligand_gate);

#[pyclass]
#[pyo3(name = "DestexheLigandGatedChannels")]
#[derive(Clone)]
pub struct PyDestexheLigandGatedChannels {
    ligand_gates: LigandGatedChannels<DestexheReceptor>
}

#[pymethods]
impl PyDestexheLigandGatedChannels {
    #[new]
    #[pyo3(signature = (neurotransmitter_types=None))]
    fn new(neurotransmitter_types: Option<&PyList>) -> PyResult<Self> {
        let mut ligand_gates: HashMap<NeurotransmitterType, LigandGatedChannel<DestexheReceptor>> = HashMap::new();

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
            PyDestexheLigandGatedChannels {
                ligand_gates: LigandGatedChannels { ligand_gates: ligand_gates }
            }
        )
    }

    fn __getitem__(&self, neurotransmitter_type: PyNeurotransmitterType) -> PyResult<PyDestexheLigandGatedChannel> {
        if let Some(value) = self.ligand_gates.get(&neurotransmitter_type.convert_type()) {
            Ok(
                PyDestexheLigandGatedChannel { 
                    ligand_gate: value.clone() 
                }
            )
        } else {
            Err(PyKeyError::new_err(format!("{:#?} not found", neurotransmitter_type)))
        }
    }

    fn set_ligand_gate(
        &mut self, neurotransmitter_type: PyNeurotransmitterType, ligand_gate: PyDestexheLigandGatedChannel
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

#[pymethods]
impl PyDestexheLigandGatedChannel {
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

        PyDestexheLigandGatedChannel {
            ligand_gate: ligand_gate
        }
    }

    fn get_receptor(&self) -> PyDestexheReceptor {
        PyDestexheReceptor { receptor: self.ligand_gate.receptor }
    }

    fn set_receptor(&mut self, receptor: PyDestexheReceptor) {
        self.ligand_gate.receptor = receptor.receptor;
    }
}

#[pyclass]
#[pyo3(name = "BasicGatingVariable")]
#[derive(Clone)]
pub struct PyBasicGatingVariable {
    gating_variable: BasicGatingVariable
}

implement_basic_getter_and_setter!(
    PyBasicGatingVariable,
    gating_variable,
    alpha, get_alpha, set_alpha,
    beta, get_beta, set_beta,
    state, get_state, set_state
);

impl_repr!(PyBasicGatingVariable, gating_variable);

#[pymethods]
impl PyBasicGatingVariable {
    #[new]
    #[pyo3(signature = (alpha=0., beta=0., state=0.))]
    fn new(alpha: f32, beta: f32, state: f32) -> Self {
        PyBasicGatingVariable { 
            gating_variable: BasicGatingVariable { 
                alpha: alpha, 
                beta: beta, 
                state: state, 
            } 
        }
    }

    fn init_state(&mut self) {
        self.gating_variable.init_state();
    }

    fn update(&mut self, dt: f32) {
        self.gating_variable.update(dt);
    }
}

#[pyclass]
#[pyo3(name = "NaIonChannel")]
#[derive(Clone)]
pub struct PyNaIonChannel {
    ion_channel: NaIonChannel
}

implement_basic_getter_and_setter!(
    PyNaIonChannel,
    ion_channel,
    g_na, get_g_na, set_g_na,
    e_na, get_e_na, set_e_na,
    current, get_current, set_current
);

impl_repr!(PyNaIonChannel, ion_channel);

#[pymethods]
impl PyNaIonChannel {
    #[new]
    #[pyo3(signature = (
        g_na=120., 
        e_na=115., 
        m=PyBasicGatingVariable { gating_variable: BasicGatingVariable::default() }, 
        h=PyBasicGatingVariable { gating_variable: BasicGatingVariable::default() }, 
        current=0.
    ))]
    fn new(g_na: f32, e_na: f32, m: PyBasicGatingVariable, h: PyBasicGatingVariable, current: f32) -> Self {
        PyNaIonChannel {
            ion_channel: NaIonChannel { 
                g_na: g_na, 
                e_na: e_na, 
                m: m.gating_variable, 
                h: h.gating_variable, 
                current: current,
            }
        }
    }

    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.ion_channel.update_current(voltage, dt);
    }

    fn get_m(&self) -> PyBasicGatingVariable {
        PyBasicGatingVariable { gating_variable: self.ion_channel.m }
    }

    fn set_m(&mut self, m: PyBasicGatingVariable) {
        self.ion_channel.m = m.gating_variable;
    }

    fn get_h(&self) -> PyBasicGatingVariable {
        PyBasicGatingVariable { gating_variable: self.ion_channel.h }
    }

    fn set_h(&mut self, h: PyBasicGatingVariable) {
        self.ion_channel.h = h.gating_variable;
    }
}

#[pyclass]
#[pyo3(name = "KIonChannel")]
#[derive(Clone)]
pub struct PyKIonChannel {
    ion_channel: KIonChannel
}

implement_basic_getter_and_setter!(
    PyKIonChannel,
    ion_channel,
    g_k, get_g_k, set_g_k,
    e_k, get_e_k, set_e_k,
    current, get_current, set_current
);

impl_repr!(PyKIonChannel, ion_channel);

#[pymethods]
impl PyKIonChannel {
    #[new]
    #[pyo3(signature = (
        g_k=36., 
        e_k=-12., 
        n=PyBasicGatingVariable { gating_variable: BasicGatingVariable::default() }, 
        current=0.
    ))]
    fn new(g_k: f32, e_k: f32, n: PyBasicGatingVariable, current: f32) -> Self {
        PyKIonChannel {
            ion_channel: KIonChannel { 
                g_k: g_k, 
                e_k: e_k, 
                n: n.gating_variable, 
                current: current 
            }
        }
    }

    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.ion_channel.update_current(voltage, dt);
    }

    fn get_n(&self) -> PyBasicGatingVariable {
        PyBasicGatingVariable { gating_variable: self.ion_channel.n }
    }

    fn set_n(&mut self, n: PyBasicGatingVariable) {
        self.ion_channel.n = n.gating_variable;
    }
}

#[pyclass]
#[pyo3(name = "KLeakChannel")]
#[derive(Clone)]
pub struct PyKLeakChannel {
    ion_channel: KLeakChannel
}

implement_basic_getter_and_setter!(
    PyKLeakChannel,
    ion_channel,
    g_k_leak, get_g_k_leak, set_g_k_leak,
    e_k_leak, get_e_k_leak, set_e_k_leak,
    current, get_current, set_current
);

impl_repr!(PyKLeakChannel, ion_channel);

#[pymethods]
impl PyKLeakChannel {
    #[new]
    #[pyo3(signature = (g_k_leak=0.3, e_k_leak=10.6, current=0.))]
    fn new(g_k_leak: f32, e_k_leak: f32, current: f32) -> Self {
        PyKLeakChannel {
            ion_channel: KLeakChannel { 
                g_k_leak: g_k_leak, 
                e_k_leak: e_k_leak, 
                current: current 
            }
        }
    }

    fn update_current(&mut self, voltage: f32) {
        self.ion_channel.update_current(voltage);
    }
}

#[pyclass]
#[pyo3(name = "HodgkinHuxleyNeuron")]
#[derive(Clone)]
pub struct PyHodgkinHuxleyNeuron {
    model: HodgkinHuxleyNeuron<DestexheNeurotransmitter, DestexheReceptor>,
}

implement_basic_getter_and_setter!(
    PyHodgkinHuxleyNeuron,
    model,
    current_voltage, get_current_voltage, set_current_voltage,
    gap_conductance, get_gap_conductance, set_gap_conductance,
    dt, get_dt, set_dt,
    c_m, get_c_m, set_c_m,
    v_th, get_v_th, set_v_th
);
impl_repr!(PyHodgkinHuxleyNeuron, model);
impl_default_neuron_methods!(
    PyHodgkinHuxleyNeuron,
    PyDestexheNeurotransmitters,
    PyDestexheLigandGatedChannels
);

#[pymethods]
impl PyHodgkinHuxleyNeuron {
    #[new]
    #[pyo3(signature = (
        current_voltage=0., gap_conductance=10., dt=0.01, c_m=1.0, v_th=60.,
        na_channel=PyNaIonChannel { ion_channel: NaIonChannel::default() },
        k_channel=PyKIonChannel { ion_channel: KIonChannel::default() },
        k_leak_channel=PyKLeakChannel { ion_channel: KLeakChannel::default() },
        synaptic_neurotransmitters=PyDestexheNeurotransmitters { 
            neurotransmitters: Neurotransmitters::<DestexheNeurotransmitter>::default() 
        },
        ligand_gates=PyDestexheLigandGatedChannels {
            ligand_gates: LigandGatedChannels::<DestexheReceptor>::default()
        }
    ))]
    fn new(
        current_voltage: f32, gap_conductance: f32, dt: f32, c_m: f32, v_th: f32,
        na_channel: PyNaIonChannel, k_channel: PyKIonChannel, k_leak_channel: PyKLeakChannel, 
        synaptic_neurotransmitters: PyDestexheNeurotransmitters, ligand_gates: PyDestexheLigandGatedChannels
    ) -> Self {
        PyHodgkinHuxleyNeuron { 
            model: HodgkinHuxleyNeuron { 
                current_voltage: current_voltage, 
                gap_conductance: gap_conductance, 
                dt: dt, 
                c_m: c_m, 
                na_channel: na_channel.ion_channel, 
                k_channel: k_channel.ion_channel,
                k_leak_channel: k_leak_channel.ion_channel, 
                v_th: v_th, 
                synaptic_neurotransmitters: synaptic_neurotransmitters.neurotransmitters,
                ligand_gates: ligand_gates.ligand_gates,
                ..HodgkinHuxleyNeuron::default()
            } 
        }
    }

    fn get_na_channel(&self) -> PyNaIonChannel {
        PyNaIonChannel { ion_channel: self.model.na_channel }
    }

    fn set_na_channel(&mut self, na_channel: PyNaIonChannel) {
        self.model.na_channel = na_channel.ion_channel;
    }

    fn get_k_channel(&self) -> PyKIonChannel {
        PyKIonChannel { ion_channel: self.model.k_channel }
    }

    fn set_k_channel(&mut self, k_channel: PyKIonChannel) {
        self.model.k_channel = k_channel.ion_channel;
    }

    fn get_k_leak_channel(&self) -> PyKLeakChannel {
        PyKLeakChannel { ion_channel: self.model.k_leak_channel }
    }

    fn set_k_leak_channel(&mut self, na_channel: PyKLeakChannel) {
        self.model.k_leak_channel = na_channel.ion_channel;
    }

    #[getter]
    fn get_was_increasing(&self) -> bool {
        self.model.was_increasing
    }

    #[setter]
    fn set_was_increasing(&mut self, was_increasing: bool) {
        self.model.was_increasing = was_increasing;
    }
}

#[pyclass]
#[pyo3(name = "HodgkinHuxleyLattice")]
#[derive(Clone)]
pub struct PyHodgkinHuxleyLattice {
    lattice: Lattice<
        HodgkinHuxleyNeuron<DestexheNeurotransmitter, DestexheReceptor>,
        AdjacencyMatrix<(usize, usize), f32>,
        GridVoltageHistory,
        STDP,
    >
}

impl_lattice!(PyHodgkinHuxleyLattice, PyHodgkinHuxleyNeuron, "HodgkinHuxleyLattice", PySTDP);

#[pyclass]
#[pyo3(name = "HodgkinHuxleyNetwork")]
#[derive(Clone)]
pub struct PyHodgkinHuxleyNetwork {
    network: LatticeNetwork<
        HodgkinHuxleyNeuron<DestexheNeurotransmitter, DestexheReceptor>, 
        AdjacencyMatrix<(usize, usize), f32>, 
        GridVoltageHistory, 
        LatticeSpikeTrain,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition, f32>,
        STDP,
    >
}

impl_network!(
    PyHodgkinHuxleyNetwork, PyHodgkinHuxleyLattice, PyPoissonLattice, PyHodgkinHuxleyNeuron,
    PyPoissonNeuron, PySTDP, "HodgkinHuxleyLattice", "PoissonLattice", "HodgkinHuxleyNetwork",
);

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeurotransmitterType>()?;
    m.add_class::<PyApproximateNeurotransmitter>()?;
    m.add_class::<PyApproximateNeurotransmitters>()?;
    m.add_class::<PyDestexheNeurotransmitter>()?;
    m.add_class::<PyDestexheNeurotransmitters>()?;
    m.add_class::<PyApproximateReceptor>()?;
    m.add_class::<PyApproximateLigandGatedChannel>()?;
    m.add_class::<PyApproximateLigandGatedChannels>()?;
    m.add_class::<PyDestexheReceptor>()?;
    m.add_class::<PyDestexheLigandGatedChannel>()?;
    m.add_class::<PyDestexheLigandGatedChannels>()?;
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PyIzhikevichLattice>()?;
    m.add_class::<PyDeltaDiracRefractoriness>()?;
    m.add_class::<PyPoissonNeuron>()?;
    m.add_class::<PyPoissonLattice>()?;
    m.add_class::<PyIzhikevichNetwork>()?;
    m.add_class::<PyBasicGatingVariable>()?;
    m.add_class::<PyNaIonChannel>()?;
    m.add_class::<PyKIonChannel>()?;
    m.add_class::<PyKLeakChannel>()?;
    m.add_class::<PyHodgkinHuxleyNeuron>()?;
    m.add_class::<PyHodgkinHuxleyLattice>()?;
    m.add_class::<PyHodgkinHuxleyNetwork>()?;

    // isspiking trait spike train in pyo3
    // falliable connect (specifically for pyo3)

    // generate network method

    // reset timing and history across network

    // reward modulation

    // in python wrapper for pyo3, connect conditional errors could be caught and made more readable
    // python could automatically generate wrappers given the __dir__ of the module
    // python wrapper should do f = lambda x: bool(x) to try and autoconvert before passing to rust
    // universal lattice and latticenetwork type that dynamically dispatch to a certain neuron type
    // depending on initialization

    // verbose option that prints progress in running simulation
    // should be printed from rust

    // spike train lattice macro

    // temp env variable for building pyo3 with custom models
    // builtin models can be listed in a separate file associated with this crate
    // use pest to generate models from file
    // macros for building receptors, ligand gates, neurotransmitters, and neurons
    // if env variable does not work when installing package, use a seperate build script
    // to generate new models that is associated with this package but accessible from the command line
    // or maybe wheel is built separately with a setup py and config is done through there
    // (could use setuptools rust for that)
    // impl neuron macro for arbitrary neuron (separate one for neurons with ion channels)

    Ok(())
}
