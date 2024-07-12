use std::{collections::{hash_map::DefaultHasher, HashMap}, hash::{Hash, Hasher}};
use pyo3::{exceptions::{PyKeyError, PyValueError}, types::{PyList, PyDict}, prelude::*};
use spiking_neural_networks::{graph::AdjacencyMatrix, neuron::{
    integrate_and_fire::IzhikevichNeuron, iterate_and_spike::{
        AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, GABAaDefault, 
        GABAbDefault, IterateAndSpike, LastFiringTime, LigandGatedChannel, 
        LigandGatedChannels, NMDADefault, NeurotransmitterConcentrations, 
        NeurotransmitterType, Neurotransmitters, 
        PotentiationType
    }, GridVoltageHistory, Lattice
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
#[pyo3(name = "PotentiationType")]
#[derive(Clone, Copy)]
pub struct PyPotentiationType {
    potentiation: PotentiationType
}

impl_repr!(PyPotentiationType, potentiation);

#[pymethods]
impl PyPotentiationType {
    #[new]
    fn new(potentiation_type: String) -> PyResult<Self> {
        match potentiation_type.to_ascii_lowercase().as_str() {
            "excitatory" => Ok(PyPotentiationType { potentiation: PotentiationType::Excitatory }),
            "inhibitory" => Ok(PyPotentiationType { potentiation: PotentiationType::Inhibitory }),
            _ => Err(PyValueError::new_err("Potentation type must be inhibitory or excitatory"))
        }
    }

    #[staticmethod]
    fn from_bool(potentiation_type: bool) -> Self {
        match potentiation_type {
            true => PyPotentiationType { potentiation: PotentiationType::Excitatory },
            false => PyPotentiationType { potentiation: PotentiationType::Inhibitory },
        }
    }

    fn is_excitatory(&self) -> bool {
        match self.potentiation {
            PotentiationType::Excitatory => true,
            PotentiationType::Inhibitory => false,
        }
    }

    fn __bool__(&self) -> bool {
        self.is_excitatory()
    }

    fn is_inhibitory(&self) -> bool {
        match self.potentiation {
            PotentiationType::Excitatory => false,
            PotentiationType::Inhibitory => true,
        }
    }
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

    fn update_receptor_kinetics(&mut self, neurotransmitter_concs: Option<&PyDict>) -> PyResult<()> {
        let concs = match neurotransmitter_concs {
            Some(ref value) => Some(pydict_to_neurotransmitters_concentration(value)?),
            None => None,
        };

        self.ligand_gates.update_receptor_kinetics(concs.as_ref());

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
        potentiation=PyPotentiationType { potentiation: PotentiationType::Excitatory }
    ))]
    fn new(
        a: f32, b: f32, c: f32, d: f32, v_th: f32, dt: f32, current_voltage: f32, v_init: f32, 
        w_value: f32, w_init: f32, gap_conductance: f32, tau_m: f32, c_m: f32, potentiation: PyPotentiationType,
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
                potentiation_type: potentiation.potentiation,
                ..IzhikevichNeuron::default()
            }
        }
    }

    fn iterate_and_spike(&mut self, i: f32) -> bool {
        self.model.iterate_and_spike(i)
    }

    #[pyo3(signature = (i, neurotransmitter_concs=None))]
    fn iterate_with_neurotransmitter_and_spike(&mut self, i: f32, neurotransmitter_concs: Option<&PyDict>) -> PyResult<bool> {
        let concs = match neurotransmitter_concs {
            Some(ref value) => Some(pydict_to_neurotransmitters_concentration(value)?),
            None => None,
        };

        Ok(self.model.iterate_with_neurotransmitter_and_spike(i, concs.as_ref()))
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

// eventually use macro to generate lattices for each neuronal type
// could have user precompile or dynamically dispatch neuron lattice type
// depending on user input and wrap relevant functions
type LatticeNeuron = IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
type PyLatticeNeuron = PyIzhikevichNeuron;

#[pyclass]
#[pyo3(name = "IzhikevichLattice")]
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

    // fn connect(&mut self, connection_conditional: PyAny, weight_logc: PyAny) -> PyResult<()> {

    // }

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
}

// #[pyclass]
// #[pyo3(name = "IzhikevichNetwork")]
// pub struct PyIzhikevichNetwork {
//     network: LatticeNetwork<LatticeNeuron, >
// }

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPotentiationType>()?;
    m.add_class::<PyNeurotransmitterType>()?;
    m.add_class::<PyApproximateNeurotransmitter>()?;
    m.add_class::<PyApproximateNeurotransmitters>()?;
    m.add_class::<PyApproximateLigandGatedChannel>()?;
    m.add_class::<PyApproximateLigandGatedChannels>()?;
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PyIzhikevichLattice>()?;
    // m.add_class::<PyIzhikevichNetwork>()?;

    Ok(())
}
