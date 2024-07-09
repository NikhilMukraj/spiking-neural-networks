use std::collections::HashMap;
use pyo3::{exceptions::{PyKeyError, PyValueError}, types::PyList, prelude::*};
use spiking_neural_networks::neuron::{
    integrate_and_fire::IzhikevichNeuron, 
    iterate_and_spike::{
        ApproximateNeurotransmitter, ApproximateReceptor, IterateAndSpike, 
        NeurotransmitterType, Neurotransmitters, PotentiationType,
        AMPADefault, GABAaDefault, GABAbDefault, NMDADefault,
    }
};


#[pyclass]
#[pyo3(name = "PotentiationType")]
#[derive(Clone, Copy)]
pub struct PyPotentiationType {
    potentiation: PotentiationType
}

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
#[derive(Debug, Clone, Copy)]
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

#[pyclass]
#[pyo3(name = "ApproximateNeurotransmitters")]
#[derive(Clone)]
pub struct PyApproximateNeurotransmitters {
    neurotransmitters: Neurotransmitters<ApproximateNeurotransmitter>
}

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

    // fn iterate_with_neurotransmitter_and_spike(&mut self, i: f32, neurotransmitter_conc: Option<PyDict>) -> bool {

    // }
}

#[pymodule]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeurotransmitterType>()?;
    m.add_class::<PyApproximateNeurotransmitter>()?;
    m.add_class::<PyApproximateNeurotransmitters>()?;
    m.add_class::<PyIzhikevichNeuron>()?;

    Ok(())
}
