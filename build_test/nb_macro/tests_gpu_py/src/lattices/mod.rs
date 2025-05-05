macro_rules! impl_lattice {
    ($lattice_kind:ident, $lattice_neuron:ident, $name:literal) => {
        #[pymethods]
        impl $lattice_kind {
            #[new]
            #[pyo3(signature = (id=0))]
            fn new(id: usize) -> Self {
                let mut lattice = $lattice_kind { lattice: LatticeGPU::try_default().unwrap() };
                lattice.set_id(id);

                lattice
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

pub(super) use impl_lattice;
