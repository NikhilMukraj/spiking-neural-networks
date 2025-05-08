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

            fn set_dt(&mut self, dt: f32) {
                self.lattice.set_dt(dt);
            } 

            fn populate(&mut self, neuron: $lattice_neuron, num_rows: usize, num_cols: usize) -> PyResult<()> {
                match self.lattice.populate(&neuron.model, num_rows, num_cols) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(PyValueError::new_err("Input dimensions do not match current dimensions")),
                }
            }

            #[pyo3(signature = (connection_conditional, weight_logic=None))]
            fn connect(&mut self, py: Python, connection_conditional: &PyAny, weight_logic: Option<&PyAny>) -> PyResult<()> {
                let py_callable = connection_conditional.to_object(connection_conditional.py());

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.lattice.falliable_connect(&connection_closure, weight_closure.as_deref()) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }
            }

            fn get_every_node(&self) -> HashSet<(usize, usize)> {
                self.lattice.internal_graph().get_every_node()
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
                let neuron = match self.lattice.cell_grid().get(row) {
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
                let mut current_copy: Vec<Vec<_>> = self.lattice.cell_grid().clone().to_vec();

                let row_cells = match current_copy.get_mut(row) {
                    Some(row_cells) => row_cells,
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };

                if let Some(existing_neuron) = row_cells.get_mut(col) {
                    *existing_neuron = neuron.model.clone();
                    self.lattice.set_cell_grid(current_copy.to_vec()).unwrap();

                    Ok(())
                } else {
                    Err(PyKeyError::new_err(format!("Column at {} not found", col)))
                }
            }

            fn get_weight(&self, presynaptic: (usize, usize), postsynaptic: (usize, usize)) -> PyResult<f32> {
                match self.lattice.internal_graph().lookup_weight(&presynaptic, &postsynaptic) {
                    Ok(value) => Ok(value.unwrap_or(0.)),
                    Err(_) => Err(PyKeyError::new_err(
                        format!("Weight at ({:#?}, {:#?}) not found", presynaptic, postsynaptic))
                    )
                }
            }

            fn get_incoming_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.internal_graph().get_incoming_connections(&position) {
                    Ok(value) => Ok(value),
                    Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                }
            }

            fn get_outgoing_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.internal_graph().get_outgoing_connections(&position) {
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

            fn apply_given_position(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
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
            fn get_parallel(&self) -> bool {
                self.lattice.parallel
            }

            #[setter]
            fn set_parallel(&mut self, flag: bool) {
                self.lattice.parallel = flag;
            }

            #[getter]
            fn get_electrical_synapse(&self) -> bool {
                self.lattice.electrical_synapse
            }

            #[setter]
            fn set_electrical_synapse(&mut self, flag: bool) {
                self.lattice.electrical_synapse = flag;
            }

            #[getter]
            fn get_chemical_synapse(&self) -> bool {
                self.lattice.chemical_synapse
            }

            #[setter]
            fn set_chemical_synapse(&mut self, flag: bool) {
                self.lattice.chemical_synapse = flag;
            }

            #[getter]
            fn weights_history(&self) -> Vec<Vec<Vec<f32>>> {
                self.lattice.internal_graph().history.clone()
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
                self.lattice.internal_graph().matrix.clone()
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
                self.lattice.internal_graph().position_to_index.clone()
            }

            fn __repr__(&self) -> PyResult<String> {
                let rows = self.lattice.cell_grid().len();
                let cols = self.lattice.cell_grid().get(0).unwrap_or(&vec![]).len();

                Ok(
                    format!(
                        "{} {{ ({}x{}), id: {}, do_plasticity: {}, update_grid_history: {}, update_graph_history: {} }}", 
                        $name,
                        rows,
                        cols,
                        self.lattice.get_id(),
                        self.lattice.do_plasticity,
                        self.lattice.update_grid_history,
                        self.lattice.update_graph_history
                    )
                )
            }
        }
    };
}

pub(super) use impl_lattice;

macro_rules! impl_lattice_gpu {
    ($lattice_kind:ident, $lattice_neuron:ident, $from_type:ident, $name:literal) => {
        #[pymethods]
        impl $lattice_kind {
            #[new]
            #[pyo3(signature = (id=0))]
            fn new(id: usize) -> Self {
                let mut lattice = $lattice_kind { lattice: LatticeGPU::try_default().unwrap() };
                lattice.set_id(id);

                lattice
            }

            #[staticmethod]
            fn from_lattice(lattice: &$from_type) -> Self {
                $lattice_kind { lattice: LatticeGPU::from_lattice(lattice.lattice.clone()).unwrap() }
            }

            fn set_dt(&mut self, dt: f32) {
                self.lattice.set_dt(dt);
            } 

            fn populate(&mut self, neuron: $lattice_neuron, num_rows: usize, num_cols: usize) {
                self.lattice.populate(&neuron.model, num_rows, num_cols)
            }

            #[pyo3(signature = (connection_conditional, weight_logic=None))]
            fn connect(&mut self, py: Python, connection_conditional: &PyAny, weight_logic: Option<&PyAny>) -> PyResult<()> {
                let py_callable = connection_conditional.to_object(connection_conditional.py());

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.lattice.falliable_connect(&connection_closure, weight_closure.as_deref()) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyValueError::new_err(e.to_string())),
                }
            }

            fn get_every_node(&self) -> HashSet<(usize, usize)> {
                self.lattice.internal_graph().get_every_node()
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
                let neuron = match self.lattice.cell_grid().get(row) {
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
                let mut current_copy: Vec<Vec<_>> = self.lattice.cell_grid().clone().to_vec();

                let row_cells = match current_copy.get_mut(row) {
                    Some(row_cells) => row_cells,
                    None => {
                        return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                    }
                };

                if let Some(existing_neuron) = row_cells.get_mut(col) {
                    *existing_neuron = neuron.model.clone();
                    self.lattice.set_cell_grid(current_copy.to_vec()).unwrap();

                    Ok(())
                } else {
                    Err(PyKeyError::new_err(format!("Column at {} not found", col)))
                }
            }

            fn get_weight(&self, presynaptic: (usize, usize), postsynaptic: (usize, usize)) -> PyResult<f32> {
                match self.lattice.internal_graph().lookup_weight(&presynaptic, &postsynaptic) {
                    Ok(value) => Ok(value.unwrap_or(0.)),
                    Err(_) => Err(PyKeyError::new_err(
                        format!("Weight at ({:#?}, {:#?}) not found", presynaptic, postsynaptic))
                    )
                }
            }

            fn get_incoming_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.internal_graph().get_incoming_connections(&position) {
                    Ok(value) => Ok(value),
                    Err(_) => Err(PyKeyError::new_err(format!("Position {:#?} not found in lattice", position))), 
                }
            }

            fn get_outgoing_connections(&self, position: (usize, usize)) -> PyResult<HashSet<(usize, usize)>> {
                match self.lattice.internal_graph().get_outgoing_connections(&position) {
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

            fn apply_given_position(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
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
            fn get_electrical_synapse(&self) -> bool {
                self.lattice.electrical_synapse
            }

            #[setter]
            fn set_electrical_synapse(&mut self, flag: bool) {
                self.lattice.electrical_synapse = flag;
            }

            #[getter]
            fn get_chemical_synapse(&self) -> bool {
                self.lattice.chemical_synapse
            }

            #[setter]
            fn set_chemical_synapse(&mut self, flag: bool) {
                self.lattice.chemical_synapse = flag;
            }

            fn reset_history(&mut self) {
                self.lattice.grid_history.reset();
            }

            #[getter(weights)]
            fn get_weights(&self) -> Vec<Vec<f32>> {
                self.lattice.internal_graph().matrix.clone()
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
                self.lattice.internal_graph().position_to_index.clone()
            }

            fn __repr__(&self) -> PyResult<String> {
                let rows = self.lattice.cell_grid().len();
                let cols = self.lattice.cell_grid().get(0).unwrap_or(&vec![]).len();

                Ok(
                    format!(
                        "{} {{ ({}x{}), id: {}, update_grid_history: {} }}", 
                        $name,
                        rows,
                        cols,
                        self.lattice.get_id(),
                        self.lattice.update_grid_history,
                    )
                )
            }
        }
    };
}

pub(super) use impl_lattice_gpu;

macro_rules! impl_spike_train_lattice {
    ($pyclass_name:ident, $spike_train_type:ident, $spike_train_lattice_type:ty, $repr_name:expr) => {
        #[pymethods]
        impl $pyclass_name {
            #[new]
            #[pyo3(signature = (id=0))]
            fn new(id: usize) -> Self {
                let mut lattice = $pyclass_name { lattice: SpikeTrainLattice::default() };
                lattice.set_id(id);
                lattice
            }

            fn set_dt(&mut self, dt: f32) {
                self.lattice.set_dt(dt);
            }

            fn populate(&mut self, neuron: $spike_train_type, num_rows: usize, num_cols: usize) -> PyResult<()> {
                match self.lattice.populate(&neuron.model, num_rows, num_cols) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(PyValueError::new_err("Input dimensions do not match current dimensions")),
                }
            }

            #[getter]
            fn get_id(&self) -> usize {
                self.lattice.get_id()
            }

            #[setter]
            fn set_id(&mut self, id: usize) {
                self.lattice.set_id(id)
            }

            fn get_neuron(&self, row: usize, col: usize) -> PyResult<$spike_train_type> {
                let neuron = self.lattice.spike_train_grid().get(row)
                    .and_then(|row_cells| row_cells.get(col))
                    .cloned()
                    .ok_or_else(|| PyKeyError::new_err(format!("Position ({}, {}) not found", row, col)))?;

                Ok($spike_train_type { model: neuron })
            }

            fn set_neuron(&mut self, row: usize, col: usize, neuron: $spike_train_type) -> PyResult<()> {
                let mut current_copy: Vec<Vec<_>> = self.lattice.spike_train_grid().clone().to_vec();
                current_copy.get_mut(row)
                    .and_then(|row_cells| row_cells.get_mut(col))
                    .map(|existing_neuron| *existing_neuron = neuron.model.clone())
                    .ok_or_else(|| return PyKeyError::new_err(format!("Position ({}, {}) not found", row, col)))?;

                self.lattice.set_spike_train_grid(current_copy).unwrap();

                Ok(())
            }

            fn apply(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);
                self.lattice.apply(|neuron| {
                    let py_neuron = $spike_train_type { model: neuron.clone() };
                    let result = py_callable.call1(py, (py_neuron,)).unwrap();
                    let updated_py_neuron: $spike_train_type = result.extract(py).unwrap();
                    *neuron = updated_py_neuron.model;
                });
                Ok(())
            }

            fn apply_given_position(&mut self, py: Python, function: &PyAny) -> PyResult<()> {
                let py_callable = function.to_object(py);
                self.lattice.apply_given_position(|(i, j), neuron| {
                    let py_neuron = $spike_train_type { model: neuron.clone() };
                    let result = py_callable.call1(py, ((i, j), py_neuron)).unwrap();
                    let updated_py_neuron: $spike_train_type = result.extract(py).unwrap();
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

            fn run_lattice(&mut self, iterations: usize) -> PyResult<()> {
                match self.lattice.run_lattice(iterations) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(PyValueError::new_err("Could not run lattices")),
                }
            }

            #[getter]
            fn get_history(&self) -> Vec<Vec<Vec<f32>>> {
                self.lattice.grid_history.history.clone()
            }

            fn __repr__(&self) -> PyResult<String> {
                let rows = self.lattice.spike_train_grid().len();
                let cols = self.lattice.spike_train_grid().first().unwrap_or(&vec![]).len();

                Ok(
                    format!(
                        "{} {{ ({}x{}), id: {}, update_grid_history: {} }}", 
                        $repr_name,
                        rows,
                        cols,
                        self.lattice.get_id(),
                        self.lattice.update_grid_history,
                    )
                )
            }
        }
    };
}

pub(super) use impl_spike_train_lattice;

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

            #[staticmethod]
            #[pyo3(signature = (lattices=Vec::new(), spike_train_lattices=Vec::new()))]
            fn generate_network(
                lattices: Vec<$lattice_kind>, spike_train_lattices: Vec<$spike_train_lattice_kind>
            ) -> PyResult<Self> {
                let mut network = $network_kind { network: LatticeNetwork::default() };

                for i in lattices {
                    let id = i.lattice.get_id();
                    match network.network.add_lattice(i.lattice) {
                        Ok(_) => {},
                        Err(_) => {
                            return Err(
                                PyValueError::new_err(
                                    format!("Id ({}) already in network", id)
                                )
                            )
                        },
                    };
                }

                for i in spike_train_lattices {
                    let id = i.lattice.get_id();
                    match network.network.add_spike_train_lattice(i.lattice) {
                        Ok(_) => {},
                        Err(_) => {
                            return Err(
                                PyValueError::new_err(
                                    format!("Id ({}) already in network", id)
                                )
                            )
                        },
                    };
                }

                Ok(network)
            }

            fn set_dt(&mut self, dt: f32) {
                self.network.set_dt(dt);
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

            fn get_all_ids(&self) -> HashSet<usize> {
                self.network.get_all_ids()     
            }

            #[pyo3(signature = (id, connection_conditional, weight_logic=None))]
            fn connect_internally(
                &mut self, py: Python, id: usize, connection_conditional: &PyAny, weight_logic: Option<&PyAny>,
            ) -> PyResult<()> {
                let py_callable = connection_conditional.to_object(connection_conditional.py());

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.network.falliable_connect_interally(id, &connection_closure, weight_closure.as_deref()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        LatticeNetworkError::IDNotFoundInLattices(id) => Err(PyKeyError::new_err(
                            format!("Presynaptic id ({}) not found", id)
                        )),
                        LatticeNetworkError::ConnectionFailure(_) => Err(PyValueError::new_err(
                            e.to_string()
                        )),
                        _ => unreachable!(),
                    },
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

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.network.falliable_connect(
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
                        LatticeNetworkError::ConnectionFailure(_) => Err(PyValueError::new_err(
                            e.to_string()
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
                        
                    match current_lattice.graph().lookup_weight(&presynaptic.pos, &postsynaptic.pos) {
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
                        match lattice.graph().get_incoming_connections(&position) {
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
                        match lattice.graph().get_outgoing_connections(&position) {
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
                        let graph_pos = GraphPosition { id, pos: position };
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
                        let graph_pos = GraphPosition { id, pos: position };
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
                        let neuron = match lattice.cell_grid().get(row) {
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
                        let mut current_copy: Vec<Vec<_>> = lattice.cell_grid().clone().to_vec();

                        let row_cells = match current_copy.get_mut(row) {
                            Some(row_cells) => row_cells,
                            None => {
                                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                            }
                        };
                
                        if let Some(existing_neuron) = row_cells.get_mut(col) {
                            *existing_neuron = neuron.model.clone();

                            lattice.set_cell_grid(current_copy).unwrap();
                
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
                        let neuron = match lattice.spike_train_grid().get(row) {
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
                        let mut current_copy: Vec<Vec<_>> = lattice.spike_train_grid().clone().to_vec();

                        let row_cells = match current_copy.get_mut(row) {
                            Some(row_cells) => row_cells,
                            None => {
                                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                            }
                        };
                
                        if let Some(existing_neuron) = row_cells.get_mut(col) {
                            *existing_neuron = neuron.model.clone();

                            lattice.set_spike_train_grid(current_copy).unwrap();
                
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

            #[getter]
            fn get_update_connecting_graph_history(&self) -> bool {
                self.network.update_connecting_graph_history
            }

            #[setter]
            fn set_update_connecting_graph_history(&mut self, flag: bool) {
                self.network.update_connecting_graph_history = flag;
            }

            #[getter]
            fn get_parallel(&self) -> bool {
                self.network.parallel
            }

            #[setter]
            fn set_parallel(&mut self, flag: bool) {
                self.network.parallel = flag;
            }

            #[getter]
            fn get_electrical_synapse(&self) -> bool {
                self.network.electrical_synapse
            }

            #[setter]
            fn set_electrical_synapse(&mut self, flag: bool) {
                self.network.electrical_synapse = flag;
            }

            #[getter]
            fn get_chemical_synapse(&self) -> bool {
                self.network.chemical_synapse
            }

            #[setter]
            fn set_chemical_synapse(&mut self, flag: bool) {
                self.network.chemical_synapse = flag;
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

            fn __repr__(&self) -> PyResult<String> {
                let lattice_strings = self.network.lattices_values()
                    .map(|i| {
                        let rows = i.cell_grid().len();
                        let cols = i.cell_grid().get(0).unwrap_or(&vec![]).len();

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
                        let rows = i.spike_train_grid().len();
                        let cols = i.spike_train_grid().get(0).unwrap_or(&vec![]).len();

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

pub(super) use impl_network;

macro_rules! impl_network_gpu {
    (
        $network_kind:ident, $lattice_kind:ident, $spike_train_lattice_kind:ident,
        $lattice_neuron_kind:ident, $spike_train_kind:ident, $plasticity_kind:ident,
        $lattice_neuron_name:literal, $spike_train_name:literal, $network_name:literal,
    ) => {        
        #[pymethods]
        impl $network_kind {
            #[new]
            fn new() -> Self {
                $network_kind { network: LatticeNetworkGPU::try_default().unwrap() }
            }

            #[staticmethod]
            #[pyo3(signature = (lattices=Vec::new(), spike_train_lattices=Vec::new()))]
            fn generate_network(
                lattices: Vec<$lattice_kind>, spike_train_lattices: Vec<$spike_train_lattice_kind>
            ) -> PyResult<Self> {
                let mut network = LatticeNetworkGPU::try_default().unwrap();

                for i in lattices {
                    let id = i.lattice.get_id();
                    match network.add_lattice(i.lattice) {
                        Ok(_) => {},
                        Err(_) => {
                            return Err(
                                PyValueError::new_err(
                                    format!("Id ({}) already in network", id)
                                )
                            )
                        },
                    };
                }

                for i in spike_train_lattices {
                    let id = i.lattice.get_id();
                    match network.add_spike_train_lattice(i.lattice) {
                        Ok(_) => {},
                        Err(_) => {
                            return Err(
                                PyValueError::new_err(
                                    format!("Id ({}) already in network", id)
                                )
                            )
                        },
                    };
                }

                Ok($network_kind { network })
            }

            fn set_dt(&mut self, dt: f32) {
                self.network.set_dt(dt);
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

            fn get_all_ids(&self) -> HashSet<usize> {
                self.network.get_all_ids()     
            }

            #[pyo3(signature = (id, connection_conditional, weight_logic=None))]
            fn connect_internally(
                &mut self, py: Python, id: usize, connection_conditional: &PyAny, weight_logic: Option<&PyAny>,
            ) -> PyResult<()> {
                let py_callable = connection_conditional.to_object(connection_conditional.py());

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.network.falliable_connect_interally(id, &connection_closure, weight_closure.as_deref()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        LatticeNetworkError::IDNotFoundInLattices(id) => Err(PyKeyError::new_err(
                            format!("Presynaptic id ({}) not found", id)
                        )),
                        LatticeNetworkError::ConnectionFailure(_) => Err(PyValueError::new_err(
                            e.to_string()
                        )),
                        _ => unreachable!(),
                    },
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

                let connection_closure = move |a: (usize, usize), b: (usize, usize)| -> Result<bool, LatticeNetworkError> {
                    let args = PyTuple::new(py, &[a, b]);
                    match py_callable.call1(py, args).unwrap().extract::<bool>(py) {
                        Ok(value) => Ok(value),
                        Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                    }
                };

                let weight_closure: Option<Box<dyn Fn((usize, usize), (usize, usize)) -> Result<f32, LatticeNetworkError>>> = match weight_logic {
                    Some(value) => {
                        let py_callable = value.to_object(value.py()); 

                        let closure = move |a: (usize, usize), b: (usize, usize)| -> Result<f32, LatticeNetworkError> {
                            let args = PyTuple::new(py, &[a, b]);
                            match py_callable.call1(py, args).unwrap().extract::<f32>(py) {
                                Ok(value) => Ok(value),
                                Err(e) => Err(LatticeNetworkError::ConnectionFailure(e.to_string())),
                            }
                        };

                        Some(Box::new(closure))
                    },
                    None => None,
                };

                match self.network.falliable_connect(
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
                        LatticeNetworkError::ConnectionFailure(_) => Err(PyValueError::new_err(
                            e.to_string()
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
                        
                    match current_lattice.graph().lookup_weight(&presynaptic.pos, &postsynaptic.pos) {
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
                        match lattice.graph().get_incoming_connections(&position) {
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
                        match lattice.graph().get_outgoing_connections(&position) {
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
                        let graph_pos = GraphPosition { id, pos: position };
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
                        let graph_pos = GraphPosition { id, pos: position };
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
                        let neuron = match lattice.cell_grid().get(row) {
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
                        let mut current_copy: Vec<Vec<_>> = lattice.cell_grid().clone().to_vec();

                        let row_cells = match current_copy.get_mut(row) {
                            Some(row_cells) => row_cells,
                            None => {
                                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                            }
                        };
                
                        if let Some(existing_neuron) = row_cells.get_mut(col) {
                            *existing_neuron = neuron.model.clone();

                            lattice.set_cell_grid(current_copy).unwrap();
                
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
                        let neuron = match lattice.spike_train_grid().get(row) {
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
                        let mut current_copy: Vec<Vec<_>> = lattice.spike_train_grid().clone().to_vec();

                        let row_cells = match current_copy.get_mut(row) {
                            Some(row_cells) => row_cells,
                            None => {
                                return Err(PyKeyError::new_err(format!("Row at {} not found", row)));
                            }
                        };
                
                        if let Some(existing_neuron) = row_cells.get_mut(col) {
                            *existing_neuron = neuron.model.clone();

                            lattice.set_spike_train_grid(current_copy).unwrap();
                
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

            #[getter]
            fn get_electrical_synapse(&self) -> bool {
                self.network.electrical_synapse
            }

            #[setter]
            fn set_electrical_synapse(&mut self, flag: bool) {
                self.network.electrical_synapse = flag;
            }

            #[getter]
            fn get_chemical_synapse(&self) -> bool {
                self.network.chemical_synapse
            }

            #[setter]
            fn set_chemical_synapse(&mut self, flag: bool) {
                self.network.chemical_synapse = flag;
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

            fn __repr__(&self) -> PyResult<String> {
                let lattice_strings = self.network.lattices_values()
                    .map(|i| {
                        let rows = i.cell_grid().len();
                        let cols = i.cell_grid().get(0).unwrap_or(&vec![]).len();

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
                        let rows = i.spike_train_grid().len();
                        let cols = i.spike_train_grid().get(0).unwrap_or(&vec![]).len();

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

pub(super) use impl_network_gpu;
