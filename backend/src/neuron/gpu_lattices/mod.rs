use std::collections::{hash_map::{Values, ValuesMut}, HashMap, HashSet};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_READ_WRITE}, platform::get_platforms, program::Program, types::{cl_float, CL_NON_BLOCKING}
};
use crate::{
    error::{GPUError, GraphError, LatticeNetworkError, SpikingNeuralNetworksError}, 
    graph::{
        ConnectingGraphGPU, ConnectingGraphToGPU, Graph, GraphGPU, 
        GraphPosition, GraphToGPU, InterleavingGraphGPU
    },
};
use super::{
    check_position, 
    impl_apply, impl_reset_timing,
    iterate_and_spike::{
        generate_unique_prefix, AvailableBufferType, BufferGPU, 
        IterateAndSpike, IterateAndSpikeGPU, KernelFunction, NeurotransmitterTypeGPU
    },
    plasticity::Plasticity, 
    spike_train::{NeuralRefractorinessGPU, SpikeTrainGPU}, 
    GridVoltageHistory, Lattice, LatticeHistory, LatticeNetwork, CellGrid,
    Position, RunLattice, RunNetwork, SpikeTrainGrid, SpikeTrainGridHistory, 
    SpikeTrainLattice, SpikeTrainLatticeHistory, InternalGraph,
};
use std::ptr;


// convert graph to gpu
// may need to use gpu graph outside of trait first

// macro_rules! impl_apply {
//     () => {
//         /// Applies a function across the entire cell grid to each neuron
//         pub fn apply<F>(&mut self, f: F)
//         where
//             F: Fn(&mut T),
//         {
//             for row in self.cell_grid.iter_mut() {
//                 for neuron in row {
//                     f(neuron);
//                 }
//             }
//         }

//         /// Applies a function across the entire cell grid to each neuron
//         /// given the position, `(usize, usize)`, of the neuron and the neuron itself
//         pub fn apply_given_position<F>(&mut self, f: F)
//         where
//             F: Fn((usize, usize), &mut T),
//         {
//             for (i, row) in self.cell_grid.iter_mut().enumerate() {
//                 for (j, neuron) in row.iter_mut().enumerate() {
//                     f((i, j), neuron);
//                 }
//             }
//         }
//     };
// }

const INPUTS_KERNEL: &str = r#"
__kernel void calculate_internal_electrical_inputs(
    __global const uint *connections, 
    __global const float *weights, 
    __global const uint *index_to_position,
    __global const float *gap_conductances,
    __global const float *voltages,
    uint n, 
    __global float *res
) {
    int gid = get_global_id(0);

    float sum = 0.0f;
    float count = 0.0f;
    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i];
            int postsynaptic_index = index_to_position[gid];
            float gap_junction = gap_conductances[postsynaptic_index] * (voltages[presynaptic_index] - voltages[postsynaptic_index]);
            sum += weights[i * n + gid] * gap_junction;
            count++;
        }
    }
    
    if (count != 0.0f) {
        res[gid] = sum / count;
    } else {
        res[gid] = 0;
    }
}
"#;

const INPUTS_KERNEL_NAME: &str = "calculate_internal_electrical_inputs";

const NEUROTRANSMITTER_INPUTS_KERNEL: &str = r#"
__kernel void get_neurotransmitter_inputs(
    __global const uint *connections, 
    __global const float *weights, 
    __global const uint *index_to_position,
    __global const uint *flags,
    __global const float *t,
    uint n, 
    uint number_of_types,
    __global float *counts,
    __global float *res
) {
    int gid = get_global_id(0);

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        int idx = gid * number_of_types + t_index;
        res[idx] = 0.0f;
        counts[idx] = 0.0f;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i] * number_of_types;
            // int postsynaptic_index = index_to_position[gid]; // maybe use this instead of just gid
            for (int t_index = 0; t_index < number_of_types; t_index++) {
                if (flags[presynaptic_index + t_index] == 1) {
                    res[gid * number_of_types + t_index] += weights[i * n + gid] * t[presynaptic_index + t_index];
                    counts[gid * number_of_types + t_index]++;
                }
            }
        }
    }

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        if (counts[gid * number_of_types + t_index] != 0.0f) {
            res[gid * number_of_types + t_index] /= counts[gid  * number_of_types + t_index];
        } else {
            res[gid * number_of_types + t_index] = 0.0f;
        }
    }
}
"#;

const NEUROTRANSMITTER_INPUTS_KERNEL_NAME: &str = "get_neurotransmitter_inputs";

const GRID_VOLTAGE_HISTORY_KERNEL: &str = r#"
__kernel void add_grid_voltage_history(
    __global const uint *index_to_position,
    __global const float *current_voltage,
    __global float *history,
    int iteration,
    int size,
    int skip_index
) {
    int gid = get_global_id(0);
    int index = index_to_position[gid + skip_index];

    history[iteration * size + index_to_position[gid]] = current_voltage[index]; 
}
"#;

const GRID_VOLTAGE_HISTORY_KERNEL_NAME: &str = "add_grid_voltage_history";

// const SPIKE_HISTORY_KERNEL: &str = r#"
// __kernel void add_spike_history(
//     __global const uint *index_to_position,
//     __global const uint *is_spiking,
//     __global uint *history,
//     int iteration,
//     int size,
//     int skip_index
// ) {
//     int gid = get_global_id(0);
//     int index = index_to_position[gid + skip_index];

//     history[iteration * size + index_to_position[gid]] = is_spiking[index]; 
// }
// "#;

// const SPIKE_HISTORY_KERNEL_NAME: &str = "add_spike_history";

pub trait LatticeHistoryGPU: LatticeHistory {
    fn get_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
    fn to_gpu(&self, context: &Context, iterations: usize, size: (usize, usize)) -> Result<HashMap<String, BufferGPU>, GPUError>;
    fn add_from_gpu(
        &mut self, queue: &CommandQueue, buffers: HashMap<String, BufferGPU>, iterations: usize, size: (usize, usize)
    ) -> Result<(), GPUError>;  
}

impl LatticeHistoryGPU for GridVoltageHistory {
    fn get_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let history_program = match Program::create_and_build_from_source(context, GRID_VOLTAGE_HISTORY_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let history_kernel = match Kernel::create(&history_program, GRID_VOLTAGE_HISTORY_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let argument_names = vec![
            String::from("index_to_position"), String::from("current_voltage"), String::from("history"),
            String::from("iteration"), String::from("size"), String::from("skip_index")
        ];

        Ok(
            KernelFunction { 
                kernel: history_kernel, 
                program_source: String::from(GRID_VOLTAGE_HISTORY_KERNEL), 
                kernel_name: String::from(GRID_VOLTAGE_HISTORY_KERNEL_NAME), 
                argument_names
            }
        )
    }

    fn to_gpu(
        &self, context: &Context, iterations: usize, size: (usize, usize)
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        let history_buffer = unsafe {
            match Buffer::<cl_float>::create(
                context, CL_MEM_READ_WRITE, iterations * size.0 * size.1, ptr::null_mut()
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let mut buffers = HashMap::new();

        buffers.insert(String::from("history"), BufferGPU::Float(history_buffer));

        Ok(buffers)
    }

    fn add_from_gpu(
        &mut self, 
        queue: &CommandQueue, 
        buffers: HashMap<String, BufferGPU>, 
        iterations: usize, 
        size: (usize, usize)
    ) -> Result<(), GPUError> {
        let mut results = vec![0.0; iterations * size.0 * size.1];
        let results_read_event = unsafe {
            let event = queue.enqueue_read_buffer(
                match buffers.get("history").expect("Could not get history") {
                    BufferGPU::Float(value) => value,
                    BufferGPU::UInt(_) => unreachable!("History is not unsigned integer"),
                    BufferGPU::OptionalUInt(_) => unreachable!("History is not optional unsigned integer"),
                }, 
                CL_NON_BLOCKING, 
                0, 
                &mut results, 
                &[]
            );

            match event {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        results_read_event.wait().expect("Could not wait");

        let mut nested_vector: Vec<Vec<Vec<f32>>> = Vec::with_capacity(iterations);

        for i in 0..iterations {
            let mut grid: Vec<Vec<f32>> = Vec::with_capacity(size.0);
            for j in 0..size.0 {
                let start_idx = i * size.0 * size.1 + j * size.1;
                let end_idx = start_idx + size.1;
                grid.push(results[start_idx..end_idx].to_vec());
            }

            nested_vector.push(grid);
        }

        self.history.extend(nested_vector);

        Ok(())
    }
}

const LAST_FIRING_TIME_KERNEL: &str = r#"
__kernel void set_last_firing_time(
    __global const uint *index_to_position,
    __global const uint *is_spiking,
    __global int *last_firing_time,
    int iteration
) {
    int gid = get_global_id(0);
    int index = index_to_position[gid];

    if (is_spiking[index] != 0) {
        last_firing_time[index] = iteration;
    }
} 
"#;

const LAST_FIRING_TIME_KERNEL_NAME: &str = "set_last_firing_time";

fn create_and_write_buffer<T>(
    context: &Context,
    queue: &CommandQueue,
    size: usize,
    init_value: T,
) -> Result<Buffer<cl_float>, GPUError>
where
    T: Clone + Into<f32>,
{
    let mut buffer = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
            .map_err(|_| GPUError::BufferCreateError)?
    };

    let initial_data = vec![init_value.into(); size];
    let write_event = unsafe {
        queue
            .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &initial_data, &[])
            .map_err(|_| GPUError::BufferWriteError)?
    };

    write_event.wait().map_err(|_| GPUError::WaitError)?;

    Ok(buffer)
}

/// An implementation of a lattice that can run on the GPU
pub struct LatticeGPU<
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
> {
    cell_grid: Vec<Vec<T>>,
    graph: U,
    electrical_incoming_connections_kernel: Kernel,
    chemical_incoming_connections_kernel: Kernel,
    last_firing_time_kernel: Kernel,
    context: Context,
    queue: CommandQueue,
    pub grid_history: V,
    grid_history_kernel: KernelFunction,
    pub update_grid_history: bool,
    pub electrical_synapse: bool,
    pub chemical_synapse: bool,
    internal_clock: usize,
}

impl<T, U, V, N> LatticeGPU<T, U, V, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
{
    impl_apply!();
    impl_reset_timing!();

    /// Retrieves an immutable reference to the grid of cells
    pub fn cell_grid(&self) -> &[Vec<T>] {
        &self.cell_grid
    }

    /// Sets a cell grid (cell grid must be the same dimensions as exsting grid)
    pub fn set_cell_grid(&mut self, cell_grid: Vec<Vec<T>>) -> Result<(), GraphError> {
        for (row, expected_row) in cell_grid.iter().zip(self.cell_grid.iter()) {
            if row.len() != expected_row.len() {
                return Err(GraphError::PositionNotFound(String::from("Unmatched positions in new grid")));
            }
        }

        if cell_grid.len() != self.cell_grid.len() {
            return Err(GraphError::PositionNotFound(String::from("Unmatched positions in new grid")));
        }

        self.cell_grid = cell_grid;

        Ok(())
    }

    fn generate_cell_grid(base_neuron: &T, num_rows: usize, num_cols: usize) -> Vec<Vec<T>> {
        (0..num_rows)
            .map(|_| {
                (0..num_cols)
                    .map(|_| {
                        base_neuron.clone()
                    })
                    .collect::<Vec<T>>()
            })
            .collect()
    }

    /// Populates the cell grid with a given base neuron
    pub fn populate(&mut self, base_neuron: &T, num_rows: usize, num_cols: usize) {
        let id = self.get_id();

        self.graph = U::default();
        self.graph.set_id(id);
        self.cell_grid = Self::generate_cell_grid(base_neuron, num_rows, num_cols);

        for i in 0..num_rows {
            for j in 0..num_cols {
                self.graph.add_node((i, j))
            }
        }
    }

    /// Gets id of lattice [`Graph`]
    pub fn get_id(&self) -> usize {
        self.graph.get_id()
    }

    /// Sets id of lattice [`Graph`] given an id
    pub fn set_id(&mut self, id: usize) {
        self.graph.set_id(id);
    }

    fn generate_electrical_kernel(context: &Context) -> Result<Kernel, GPUError> {
        let electrical_incoming_connections_program = match Program::create_and_build_from_source(context, INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        match Kernel::create(&electrical_incoming_connections_program, INPUTS_KERNEL_NAME) {
            Ok(value) => Ok(value),
            Err(_) => Err(GPUError::KernelCompileFailure),
        }
    }

    fn generate_chemical_kernel(context: &Context) -> Result<Kernel, GPUError> {
        let chemical_incoming_connections_program = match Program::create_and_build_from_source(context, NEUROTRANSMITTER_INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        match Kernel::create(&chemical_incoming_connections_program, NEUROTRANSMITTER_INPUTS_KERNEL_NAME) {
            Ok(value) => Ok(value),
            Err(_) => Err(GPUError::KernelCompileFailure),
        }
    }

    fn generate_last_firing_time_kernel(context: &Context) -> Result<Kernel, GPUError> {
        let last_firing_time_program = match Program::create_and_build_from_source(context, LAST_FIRING_TIME_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        match Kernel::create(&last_firing_time_program, LAST_FIRING_TIME_KERNEL_NAME) {
            Ok(value) => Ok(value),
            Err(_) => Err(GPUError::KernelCompileFailure),
        }
    }

    // Generates a GPU lattice given a lattice and a device
    pub fn from_lattice_given_device< 
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, V, W, N>, device: &Device) -> Result<Self, GPUError> {
        let context = match Context::from_device(device) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        let queue =  match CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::GetDeviceFailure),
            };

        let electrical_incoming_connections_kernel = Self::generate_electrical_kernel(&context)?;
        let chemical_incoming_connections_kernel = Self::generate_chemical_kernel(&context)?;
        let last_firing_time_kernel = Self::generate_last_firing_time_kernel(&context)?;

        Ok(
            LatticeGPU { 
                cell_grid: lattice.cell_grid, 
                graph: lattice.graph, 
                electrical_incoming_connections_kernel,
                chemical_incoming_connections_kernel,
                last_firing_time_kernel,
                internal_clock: 0,
                grid_history_kernel: V::get_kernel(&context)?,
                grid_history: lattice.grid_history,
                update_grid_history: lattice.update_grid_history,
                electrical_synapse: lattice.electrical_synapse,
                chemical_synapse: lattice.chemical_synapse,
                context,
                queue,
            }
        )
    }

    /// Generates a GPU lattice from a given lattice
    pub fn from_lattice<
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, V, W, N>) -> Result<Self, GPUError> {
        let platforms = get_platforms().map_err(|_| GPUError::GetDeviceFailure)?;

        for platform in platforms {
            if let Ok(devices) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                if let Some(&device_id) = devices.first() {
                    let device = Device::new(device_id);
                    return LatticeGPU::from_lattice_given_device(lattice, &device);
                }
            }
        }

        Err(GPUError::GetDeviceFailure)
    }

    /// Attempts to clone lattice
    pub fn try_clone(&self) -> Result<Self, GPUError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        let queue =  match CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::GetDeviceFailure),
            };

        let electrical_incoming_connections_kernel = Self::generate_electrical_kernel(&context)?;
        let chemical_incoming_connections_kernel = Self::generate_chemical_kernel(&context)?;
        let last_firing_time_kernel = Self::generate_last_firing_time_kernel(&context)?;
        
        Ok(
            LatticeGPU { 
                cell_grid: self.cell_grid.clone(), 
                graph: self.graph.clone(), 
                electrical_incoming_connections_kernel,
                chemical_incoming_connections_kernel,
                last_firing_time_kernel,
                internal_clock: self.internal_clock,
                grid_history_kernel: V::get_kernel(&context)?,
                grid_history: self.grid_history.clone(),
                update_grid_history: self.update_grid_history,
                electrical_synapse: self.electrical_synapse,
                chemical_synapse: self.chemical_synapse,
                context,
                queue,
            }
        )
    }

    /// Attempts to generate a default configuration
    pub fn try_default() -> Result<Self, GPUError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        let queue =  match CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::GetDeviceFailure),
            };

        let electrical_incoming_connections_kernel = Self::generate_electrical_kernel(&context)?;
        let chemical_incoming_connections_kernel = Self::generate_chemical_kernel(&context)?;
        let last_firing_time_kernel = Self::generate_last_firing_time_kernel(&context)?;

        Ok(
            LatticeGPU { 
                cell_grid: vec![], 
                graph: U::default(), 
                electrical_incoming_connections_kernel,
                chemical_incoming_connections_kernel,
                last_firing_time_kernel,
                internal_clock: 0,
                grid_history_kernel: V::get_kernel(&context)?,
                grid_history: V::default(),
                update_grid_history: false,
                electrical_synapse: true,
                chemical_synapse: true,
                context,
                queue,
            }
        )
    }

    /// Sets timestep variable for the lattice
    pub fn set_dt(&mut self, dt: f32) {
        self.apply(|neuron| neuron.set_dt(dt));
        // self.plasticity.set_dt(dt);
    }

    // Gets reference to graph
    pub fn graph(&self) -> &U {
        &self.graph
    }

    /// Sets the graph of the lattice given a new lattice, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_graph(&mut self, new_graph: U) -> Result<(), GraphError> {
        let id = self.get_id();
        for pos in new_graph.get_every_node_as_ref() {
            match self.cell_grid.get(pos.0) {
                Some(row) => match row.get(pos.1) {
                    Some(_) => { continue },
                    None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
                },
                None => { return Err(GraphError::PositionNotFound(format!("{:#?}", pos))) },
            }
        }
    
        self.graph = new_graph;
        self.set_id(id);
    
        Ok(())
    }

    /// Connects the neurons in a lattice together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) {
        self.graph.get_every_node()
            .iter()
            .for_each(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j) {
                        match weight_logic {
                            Some(logic) => {
                                self.graph.edit_weight(i, j, Some((logic)(*i, *j))).unwrap();
                            },
                            None => {
                                self.graph.edit_weight(i, j, Some(1.)).unwrap();
                            }
                        };
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }
            });
    }

    /// Connects the neurons in a lattice together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect(
        &mut self, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        let output: Result<Vec<_>, LatticeNetworkError> = self.graph.get_every_node()
            .iter()
            .map(|i| {
                for j in self.graph.get_every_node().iter() {
                    if (connecting_conditional)(*i, *j)? {
                        match weight_logic {
                            Some(logic) => {
                                self.graph.edit_weight(i, j, Some((logic)(*i, *j)?)).unwrap();
                            },
                            None => {
                                self.graph.edit_weight(i, j, Some(1.)).unwrap();
                            }
                        };
                    } else {
                        self.graph.edit_weight(i, j, None).unwrap();
                    }
                }

                Ok(())
            })
            .collect();

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e)
        }
    }

    unsafe fn execute_last_firing_time(
        &self, gpu_graph: &GraphGPU, gpu_cell_grid: &HashMap<String, BufferGPU>
    ) -> Result<(), GPUError> {
        let last_firing_time_event = unsafe {
            match ExecuteKernel::new(&self.last_firing_time_kernel)
                .set_arg(&gpu_graph.index_to_position)
                .set_arg(
                    match &gpu_cell_grid.get("is_spiking").expect("Could not retrieve buffer: is_spiking") {
                        BufferGPU::UInt(buffer) => buffer,
                        _ => unreachable!("is_spiking cannot be float or optional unsigned integer"),
                    }
                )
                .set_arg(
                    match &gpu_cell_grid.get("last_firing_time").expect("Could not retrieve buffer: last_firing_time") {
                        BufferGPU::OptionalUInt(buffer) => buffer,
                        _ => unreachable!("last_firing_time cannot be float or mandatory unsigned integer"),
                    }
                )
                .set_arg(&(self.internal_clock as i32))
                .set_global_work_size(gpu_graph.size)
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
        };

        match last_firing_time_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(())
    }

    unsafe fn execute_grid_history(
        &self, 
        gpu_graph: &GraphGPU, 
        gpu_cell_grid: &HashMap<String, BufferGPU>, 
        gpu_grid_history: &HashMap<String, BufferGPU>,
    ) -> Result<(), GPUError> {
        let update_history_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(&self.grid_history_kernel.kernel);

            for i in self.grid_history_kernel.argument_names.iter() {
                if i == "iteration" {
                    kernel_execution.set_arg(&(self.internal_clock as i32));
                } else if i == "size" {
                    kernel_execution.set_arg(&(gpu_graph.size as u32));
                } else if i == "skip_index" {
                    kernel_execution.set_arg(&0);
                } else if i == "index_to_position" {
                    kernel_execution.set_arg(&gpu_graph.index_to_position);
                } else if gpu_cell_grid.contains_key(i) {
                    match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    };
                } else if gpu_grid_history.contains_key(i) {
                    match &gpu_grid_history.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    };
                } else {
                    unreachable!("Unkown argument in history kernel");
                }
            }

            match kernel_execution.set_global_work_size(gpu_graph.size)
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
        };

        match update_history_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(())
    }

    // modify to be falliable
    // modify to account for last firing time (reset firing time macro)
    pub fn run_lattice_electrical_synapses(&mut self, iterations: usize) -> Result<(), GPUError> {
        let gpu_cell_grid = T::convert_to_gpu(&self.cell_grid, &self.context, &self.queue)?;

        let gpu_graph = self.graph.convert_to_gpu(&self.context, &self.queue, &self.cell_grid)?;

        let iterate_kernel = T::iterate_and_spike_electrical_kernel(&self.context)?;

        let sums_buffer = create_and_write_buffer(&self.context, &self.queue, gpu_graph.size, 0.0)?;

        let rows = self.cell_grid.len();
        let cols = self.cell_grid.first().unwrap_or(&vec![]).len();

        let gpu_grid_history = if self.update_grid_history {
            self.grid_history.to_gpu(&self.context, iterations, (rows, cols))?
        } else {
            HashMap::new()
        };

        for _ in 0..iterations {
            let gap_junctions_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&self.electrical_incoming_connections_kernel);

                kernel_execution.set_arg(&gpu_graph.connections)
                    .set_arg(&gpu_graph.weights)
                    .set_arg(&gpu_graph.index_to_position);

                match &gpu_cell_grid.get("gap_conductance").expect("Could not retrieve buffer: gap_conductance") {
                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                    _ => unreachable!("gap_condutance must be float"),
                };

                match &gpu_cell_grid.get("current_voltage").expect("Could not retrieve buffer: current_voltage") {
                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                    _ => unreachable!("current_voltage must be float"),
                };

                match kernel_execution.set_arg(&(gpu_graph.size as u32))
                    .set_arg(&sums_buffer)
                    .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                    .enqueue_nd_range(&self.queue) {
                        Ok(value) => value,
                        Err(_) => return Err(GPUError::QueueFailure),
                    }
            };

            match gap_junctions_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(GPUError::WaitError),
            };

            let iterate_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                for i in iterate_kernel.argument_names.iter() {
                    if i == "inputs" {
                        kernel_execution.set_arg(&sums_buffer);
                    } else if i == "index_to_position" {
                        kernel_execution.set_arg(&gpu_graph.index_to_position);
                    } else {
                        match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                            BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        };
                    }
                }

                match kernel_execution.set_global_work_size(gpu_graph.size)
                    .enqueue_nd_range(&self.queue) {
                        Ok(value) => value,
                        Err(_) => return Err(GPUError::QueueFailure),
                    }
            };

            match iterate_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(GPUError::WaitError),
            };

            unsafe {
                self.execute_last_firing_time(&gpu_graph, &gpu_cell_grid)?
            };

            if self.update_grid_history {
                unsafe {
                    self.execute_grid_history(&gpu_graph, &gpu_cell_grid, &gpu_grid_history)?;
                }
            }

            self.internal_clock += 1;
        }

        if self.update_grid_history {
            self.grid_history.add_from_gpu(&self.queue, gpu_grid_history, iterations, (rows, cols))?;
        }

        T::convert_to_cpu(
            &mut self.cell_grid, 
            &gpu_cell_grid, 
            rows, 
            cols, 
            &self.queue
        )?;

        Ok(())
    }

    // maybe turn on and off gap junctions depending on whether electrical synapses is on
    pub fn run_lattice_chemical_synapses(&mut self, iterations: usize) -> Result<(), GPUError> {
        let gpu_cell_grid = T::convert_electrochemical_to_gpu(&self.cell_grid, &self.context, &self.queue)?;

        let gpu_graph = self.graph.convert_to_gpu(&self.context, &self.queue, &self.cell_grid)?;

        let iterate_kernel = T::iterate_and_spike_electrochemical_kernel(&self.context)?;

        let sums_buffer = create_and_write_buffer(&self.context, &self.queue, gpu_graph.size, 0.0)?;

        let t_sums_buffer = create_and_write_buffer(
            &self.context, &self.queue, gpu_graph.size * N::number_of_types(), 0.0
        )?;

        let counts_buffer = create_and_write_buffer(
            &self.context, &self.queue, gpu_graph.size * N::number_of_types(), 0.0
        )?;

        let rows = self.cell_grid.len();
        let cols = self.cell_grid.first().unwrap_or(&vec![]).len();

        let gpu_grid_history = if self.update_grid_history {
            self.grid_history.to_gpu(&self.context, iterations, (rows, cols))?
        } else {
            HashMap::new()
        };

        for _ in 0..iterations {
            if self.electrical_synapse {
                let gap_junctions_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&self.electrical_incoming_connections_kernel);

                    kernel_execution.set_arg(&gpu_graph.connections)
                        .set_arg(&gpu_graph.weights)
                        .set_arg(&gpu_graph.index_to_position);

                    match &gpu_cell_grid.get("gap_conductance").expect("Could not retrieve buffer") {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("gap_condutance must be float"),
                    };

                    match &gpu_cell_grid.get("current_voltage").expect("Could not retrieve buffer") {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("current_voltage must be float"),
                    };

                    match kernel_execution.set_arg(&(gpu_graph.size as u32))
                        .set_arg(&sums_buffer)
                        .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match gap_junctions_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };
            }

            let chemical_synapses_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&self.chemical_incoming_connections_kernel);

                kernel_execution.set_arg(&gpu_graph.connections)
                    .set_arg(&gpu_graph.weights)
                    .set_arg(&gpu_graph.index_to_position);

                match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                    BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    _ => unreachable!("neurotransmitters$flags"),
                };

                match &gpu_cell_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                    _ => unreachable!("neurotransmitters$t must be float"),
                };

                kernel_execution.set_arg(&(gpu_graph.size as u32))
                    .set_arg(&(N::number_of_types() as u32));

                match kernel_execution
                    .set_arg(&counts_buffer)
                    .set_arg(&t_sums_buffer)
                    .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                    .enqueue_nd_range(&self.queue) {
                        Ok(value) => value,
                        Err(_) => return Err(GPUError::QueueFailure),
                    }
            };

            match chemical_synapses_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(GPUError::WaitError),
            };

            let iterate_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                for i in iterate_kernel.argument_names.iter() {
                    if i == "inputs" {
                        kernel_execution.set_arg(&sums_buffer);
                    } else if i == "t" {
                        kernel_execution.set_arg(&t_sums_buffer);
                    } else if i == "index_to_position" {
                        kernel_execution.set_arg(&gpu_graph.index_to_position);
                    } else if i == "number_of_types" {
                        kernel_execution.set_arg(&(N::number_of_types() as u32));
                    } else if i == "neuro_flags" {
                        match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            _ => unreachable!("Could not retrieve neurotransmitter flags"),
                        };
                    } else if i == "lg_flags" {
                        match &gpu_cell_grid.get("ligand_gates$flags").expect("Could not retrieve receptor flags") {
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            _ => unreachable!("Could not retrieve receptor flags"),
                        };
                    } else {
                        match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                            BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        };
                    }
                }

                match kernel_execution.set_global_work_size(gpu_graph.size)
                    .enqueue_nd_range(&self.queue) {
                        Ok(value) => value,
                        Err(_) => return Err(GPUError::QueueFailure),
                    }
            };

            match iterate_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(GPUError::WaitError),
            };

            unsafe {
                self.execute_last_firing_time(&gpu_graph, &gpu_cell_grid)?
            };

            if self.update_grid_history {
                unsafe {
                    self.execute_grid_history(&gpu_graph, &gpu_cell_grid, &gpu_grid_history)?;
                }
            }

            self.internal_clock += 1;
        }

        if self.update_grid_history {
            self.grid_history.add_from_gpu(&self.queue, gpu_grid_history, iterations, (rows, cols))?;
        }

        T::convert_electrochemical_to_cpu(
            &mut self.cell_grid, 
            &gpu_cell_grid, 
            rows, 
            cols, 
            &self.queue
        )?;

        Ok(())
    }
}

impl<T, U, V, N> CellGrid for LatticeGPU<T, U, V, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
{
    type T = T;
    
    fn cell_grid(&self) -> &[Vec<T>] {
        &self.cell_grid
    }
}

impl<T, U, V, N> RunLattice for LatticeGPU<T, U, V, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
{
    fn run_lattice(&mut self, iterations: usize) -> Result<(), SpikingNeuralNetworksError> {
        if self.cell_grid.is_empty() || self.cell_grid.first().unwrap_or(&vec![]).is_empty() || iterations == 0 {
            return Ok(());
        }

        match (self.electrical_synapse, self.chemical_synapse) {
            (true, false) => self.run_lattice_electrical_synapses(iterations).map_err(Into::into),
            (false, true) => self.run_lattice_chemical_synapses(iterations).map_err(Into::into),
            (true, true) => self.run_lattice_chemical_synapses(iterations).map_err(Into::into),
            (false, false) => Ok(()),
        }
    }
}

impl<T, U, V, N> InternalGraph for LatticeGPU<T, U, V, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
{
    type T = U;
    
    fn internal_graph(&self) -> &U {
        self.graph()
    }

    fn set_internal_graph(&mut self, new_graph: Self::T) -> Result<(), GraphError> {
        self.set_graph(new_graph)
    }
}

pub trait SpikeTrainLatticeHistoryGPU: SpikeTrainLatticeHistory {
    fn get_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
    fn to_gpu(&self, context: &Context, iterations: usize, size: (usize, usize)) -> Result<HashMap<String, BufferGPU>, GPUError>;
    fn add_from_gpu(
        &mut self, queue: &CommandQueue, buffers: HashMap<String, BufferGPU>, iterations: usize, size: (usize, usize)
    ) -> Result<(), GPUError>;  
}

const SPIKE_TRAIN_GRID_VOLTAGE_HISTORY_KERNEL: &str = r#"
__kernel void add_spike_train_grid_voltage_history(
    __global const uint *index_to_position,
    __global const float *current_voltage,
    __global float *history,
    int iteration,
    int size,
    int skip_index
) {
    int gid = get_global_id(0);
    int index = index_to_position[gid + skip_index];

    history[iteration * size + index_to_position[gid]] = current_voltage[index]; 
}
"#;

const SPIKE_TRAIN_GRID_VOLTAGE_HISTORY_KERNEL_NAME: &str = "add_spike_train_grid_voltage_history";

impl SpikeTrainLatticeHistoryGPU for SpikeTrainGridHistory {
    fn get_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let history_program = match Program::create_and_build_from_source(context, SPIKE_TRAIN_GRID_VOLTAGE_HISTORY_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let history_kernel = match Kernel::create(&history_program, SPIKE_TRAIN_GRID_VOLTAGE_HISTORY_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let argument_names = vec![
            String::from("index_to_position"), String::from("current_voltage"), String::from("history"),
            String::from("iteration"), String::from("size"), String::from("skip_index"),
        ];

        Ok(
            KernelFunction { 
                kernel: history_kernel, 
                program_source: String::from(GRID_VOLTAGE_HISTORY_KERNEL), 
                kernel_name: String::from(GRID_VOLTAGE_HISTORY_KERNEL_NAME), 
                argument_names
            }
        )
    }

    fn to_gpu(
        &self, context: &Context, iterations: usize, size: (usize, usize)
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        let history_buffer = unsafe {
            match Buffer::<cl_float>::create(
                context, CL_MEM_READ_WRITE, iterations * size.0 * size.1, ptr::null_mut()
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        let mut buffers = HashMap::new();

        buffers.insert(String::from("history"), BufferGPU::Float(history_buffer));

        Ok(buffers)
    }

    fn add_from_gpu(
        &mut self, 
        queue: &CommandQueue, 
        buffers: HashMap<String, BufferGPU>, 
        iterations: usize, 
        size: (usize, usize)
    ) -> Result<(), GPUError> {
        let mut results = vec![0.0; iterations * size.0 * size.1];
        let results_read_event = unsafe {
            let event = queue.enqueue_read_buffer(
                match buffers.get("history").expect("Could not get history") {
                    BufferGPU::Float(value) => value,
                    BufferGPU::UInt(_) => unreachable!("History is not unsigned integer"),
                    BufferGPU::OptionalUInt(_) => unreachable!("History is not optional unsigned integer"),
                }, 
                CL_NON_BLOCKING, 
                0, 
                &mut results, 
                &[]
            );

            match event {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferCreateError),
            }
        };

        results_read_event.wait().expect("Could not wait");

        let mut nested_vector: Vec<Vec<Vec<f32>>> = Vec::with_capacity(iterations);

        for i in 0..iterations {
            let mut grid: Vec<Vec<f32>> = Vec::with_capacity(size.0);

            for j in 0..size.0 {
                let start_idx = i * size.0 * size.1 + j * size.1;
                let end_idx = start_idx + size.1;
                grid.push(results[start_idx..end_idx].to_vec());
            }

            nested_vector.push(grid);
        }

        self.history.extend(nested_vector);

        Ok(())
    }
}

const NETWORK_ELECTRICAL_INPUTS_KERNEL: &str = r#"
__kernel void calculate_network_electrical_inputs(
    __global const uint *connections, 
    __global const float *weights, 
    __global const int *index_to_position,
    __global const float *gap_conductances,
    __global const float *voltages,
    uint n, 
    __global float *res
) {
    int gid = get_global_id(0);

    float sum = 0.0f;
    float count = 0.0f;
    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i];
            int postsynaptic_index = index_to_position[gid];
            float gap_junction = gap_conductances[postsynaptic_index] * (voltages[presynaptic_index] - voltages[postsynaptic_index]);
            sum += weights[i * n + gid] * gap_junction;
            count++;
        }
    }
    
    if (count != 0.0f) {
        res[gid] = sum / count;
    } else {
        res[gid] = 0.0f;
    }
}
"#;

const NETWORK_ELECTRICAL_INPUTS_KERNEL_NAME: &str = "calculate_network_electrical_inputs";

const NETWORK_CHEMICAL_INPUTS_KERNEL: &str = r#"
__kernel void calculate_network_chemical_inputs(
    __global const uint *connections, 
    __global const float *weights, 
    __global const uint *index_to_position,
    __global const uint *flags,
    __global const float *t,
    uint n, 
    uint number_of_types,
    __global float *counts,
    __global float *res
) {
    int gid = get_global_id(0);

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        int idx = gid * number_of_types + t_index;
        res[idx] = 0.0f;
        counts[idx] = 0.0f;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i] * number_of_types;
            // int postsynaptic_index = index_to_position[gid]; // maybe use this instead of just gid
            for (int t_index = 0; t_index < number_of_types; t_index++) {
                if (flags[presynaptic_index + t_index] == 1) {
                    res[gid * number_of_types + t_index] += weights[i * n + gid] * t[presynaptic_index + t_index];
                    counts[gid * number_of_types + t_index]++;
                }
            }
        }
    }

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        if (counts[gid * number_of_types + t_index] != 0.0f) {
            res[gid * number_of_types + t_index] /= counts[gid * number_of_types + t_index];
        } else {
            res[gid * number_of_types + t_index] = 0.0f;
        }
    }
}
"#;

const NETWORK_CHEMICAL_INPUTS_KERNEL_NAME: &str = "calculate_network_chemical_inputs";

const NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL: &str = r#"
__kernel void calculate_network_with_spike_train_chemical_inputs(
    __global const uint *connections, 
    __global const float *weights, 
    __global const uint *index_to_position,
    __global const uint *is_spike_train,
    __global const uint *flags,
    __global const float *t,
    __global const uint *spike_train_flags,
    __global const float *spike_train_t,
    uint skip_index,
    uint n, 
    uint number_of_types,
    __global float *counts,
    __global float *res
) {
    int gid = get_global_id(0);

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        int idx = gid * number_of_types + t_index;
        res[idx] = 0.0f;
        counts[idx] = 0.0f;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i] * number_of_types;

            if (is_spike_train[index_to_position[i]] == 0) {
                for (int t_index = 0; t_index < number_of_types; t_index++) {
                    if (flags[presynaptic_index + t_index] == 1) {
                        res[gid * number_of_types + t_index] += weights[i * n + gid] * t[presynaptic_index + t_index];
                        counts[gid * number_of_types + t_index]++;
                    }
                }
            } else {
                int spike_train_presynaptic_index = presynaptic_index - (skip_index * number_of_types);
                for (int t_index = 0; t_index < number_of_types; t_index++) {
                    if (spike_train_flags[spike_train_presynaptic_index + t_index] == 1) {
                        res[gid * number_of_types + t_index] += weights[i * n + gid] * spike_train_t[spike_train_presynaptic_index + t_index];
                        counts[gid * number_of_types + t_index]++;
                    }
                }
            }
        }
    }

    for (int t_index = 0; t_index < number_of_types; t_index++) {
        if (counts[gid * number_of_types + t_index] != 0.0f) {
            res[gid * number_of_types + t_index] /= counts[gid * number_of_types + t_index];
        } else {
            res[gid * number_of_types + t_index] = 0.0f;
        }
    }
}
"#;

const NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME: &str = "calculate_network_with_spike_train_chemical_inputs";

fn generate_network_spike_train_electrical_inputs_kernel<U: NeuralRefractorinessGPU>(context: &Context) -> Result<KernelFunction, GPUError> {
    let mut args = vec![
        String::from("connections"), String::from("weights"), String::from("index_to_position"),
        String::from("is_spike_train"), String::from("gap_conductances"), String::from("voltages"),
        String::from("last_firing_time"),
    ];

    let refractoriness_function = U::get_refractoriness_gpu_function()?;

    let mut refractoriness_kernel_args = vec![];
    let mut refractoriness_function_args = vec![];

    let spike_train_prefix = generate_unique_prefix(&args, "spike_train");

    for i in refractoriness_function.0 {
        if i.0 == "last_firing_time" {
            refractoriness_function_args.push(String::from("last_firing_time"));
            continue;
        }

        match i.1 {
            Some(val) => {
                let current_split = i.0.split("$").collect::<Vec<&str>>();
                let current_arg = if current_split.len() == 2 {
                    format!("{}{}", generate_unique_prefix(&args, current_split[0]), current_split[1])
                } else {
                    i.0.clone()
                };

                let current_arg = format!(
                    "{}{}",
                    spike_train_prefix,
                    current_arg
                );

                args.push(i.0.clone());
                refractoriness_function_args.push(current_arg.clone());

                match val {
                    AvailableBufferType::Float => refractoriness_kernel_args.push(format!("__global const float *{}", current_arg)),
                    AvailableBufferType::UInt => refractoriness_kernel_args.push(format!("__global const uint *{}", current_arg)),
                    AvailableBufferType::OptionalUInt => refractoriness_kernel_args.push(format!("__global const int *{}", current_arg)),
                }
            },
            None => {
                refractoriness_kernel_args.push(String::from("int timestep"));
                args.push(String::from("timestep"));
            }
        };
    }

    let program_source = format!(r#"
            {}

            __kernel void calculate_network_electrical_with_spike_train_inputs(
                __global const uint *connections, 
                __global const float *weights, 
                __global const int *index_to_position,
                __global const uint *is_spike_train,
                __global const float *gap_conductances,
                __global const float *voltages,
                __global const int *last_firing_time,
                {},
                uint skip_index,
                uint n, 
                __global float *res
            ) {{
                int gid = get_global_id(0);
                int index = index_to_position[gid];

                float sum = 0.0f;
                float count = 0.0f;
                for (int i = 0; i < n; i++) {{
                    if (connections[i * n + gid] == 1) {{
                        int presynaptic_index = index_to_position[i];
                        int postsynaptic_index = index_to_position[gid];
                        if (is_spike_train[presynaptic_index] == 0) {{
                            float gap_junction = gap_conductances[postsynaptic_index] * (voltages[presynaptic_index] - voltages[postsynaptic_index]);
                            sum += weights[i * n + gid] * gap_junction;
                        }} else {{
                            if (last_firing_time[presynaptic_index - skip_index] < 0) {{
                                sum += weights[i * n + gid] * {}v_resting[presynaptic_index - skip_index];
                            }} else {{
                                sum += weights[i * n + gid] * gap_conductances[postsynaptic_index] * get_effect(timestep, {});
                            }}
                        }}
                        count++;
                    }}
                }}

                if (count != 0.0f) {{
                    res[gid] = sum / count;
                }} else {{
                    res[gid] = 0.0f;
                }}
            }}
        "#,
        refractoriness_function.1,
        refractoriness_kernel_args.join(",\n"),
        spike_train_prefix,
        refractoriness_function_args.iter()
            .map(|i| format!("{}[presynaptic_index - skip_index]", i))
            .collect::<Vec<String>>()
            .join(", "),
    );

    args.extend(vec![String::from("skip_index"), String::from("n"), String::from("res")]);

    let kernel_name = String::from("calculate_network_electrical_with_spike_train_inputs");

    let spike_train_gap_junctions_program = match Program::create_and_build_from_source(context, &program_source, "") {
        Ok(value) => value,
        Err(_) => return Err(GPUError::ProgramCompileFailure),
    };

    let kernel = match Kernel::create(&spike_train_gap_junctions_program, &kernel_name) {
        Ok(value) => value,
        Err(_) => return Err(GPUError::KernelCompileFailure),
    };

    Ok(
        KernelFunction { 
            kernel, 
            program_source, 
            kernel_name, 
            argument_names: args, 
        }
    )
}

/// An implementation of a lattice network that is compatible with the GPU
pub struct LatticeNetworkGPU<
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N, U=R>,
    X: SpikeTrainLatticeHistoryGPU,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
    R: NeuralRefractorinessGPU,
    C: Graph<K=GraphPosition, V=f32> + ConnectingGraphToGPU<ConnectingGraphGPU>, 
> {
    lattices: HashMap<usize, Lattice<T, U, V, Y, N>>,
    spike_train_lattices: HashMap<usize, SpikeTrainLattice<N, W, X>>,
    connecting_graph: C,
    electrical_incoming_connections_kernel: Kernel,
    electrical_and_spike_train_incoming_connections: KernelFunction,
    chemical_incoming_connections_kernel: Kernel,
    chemical_and_spike_train_incoming_connections: Kernel,
    last_firing_time_kernel: Kernel,
    context: Context,
    queue: CommandQueue,
    grid_history_kernel: KernelFunction,
    spike_train_grid_history_kernel: KernelFunction,
    pub electrical_synapse: bool,
    pub chemical_synapse: bool,
    internal_clock: usize,
}

impl<T, U, V, W, X, Y, N, R, C> LatticeNetworkGPU<T, U, V, W, X, Y, N, R, C>
where
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU,
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N, U=R>,
    X: SpikeTrainLatticeHistoryGPU,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
    R: NeuralRefractorinessGPU,
    C: Graph<K=GraphPosition, V=f32> + ConnectingGraphToGPU<ConnectingGraphGPU>,
{
    // Generates a GPU lattice network given a network and a device
    pub fn from_network_given_device(
        lattice_network: LatticeNetwork<T, U, V, W, X, C, Y, N>,
        device: &Device
    ) -> Result<Self, GPUError> {
        let context = match Context::from_device(device) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        let queue = match CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            0,
        ) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        let electrical_incoming_connections_program = match Program::create_and_build_from_source(&context, NETWORK_ELECTRICAL_INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let electrical_incoming_connections_kernel = match Kernel::create(&electrical_incoming_connections_program, NETWORK_ELECTRICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let chemical_incoming_connections_program = match Program::create_and_build_from_source(&context, NETWORK_CHEMICAL_INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let chemical_incoming_connections_kernel = match Kernel::create(&chemical_incoming_connections_program, NETWORK_CHEMICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let chemical_and_spike_train_incoming_connections = match Program::create_and_build_from_source(&context, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let chemical_and_spike_train_incoming_connections = match Kernel::create(&chemical_and_spike_train_incoming_connections, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let last_firing_time_program = match Program::create_and_build_from_source(&context, LAST_FIRING_TIME_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let last_firing_time_kernel = match Kernel::create(&last_firing_time_program, LAST_FIRING_TIME_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        Ok(
            LatticeNetworkGPU { 
                lattices: lattice_network.get_lattices().clone(), 
                spike_train_lattices: lattice_network.get_spike_train_lattices().clone(),
                connecting_graph: lattice_network.get_connecting_graph().clone(), 
                last_firing_time_kernel, 
                electrical_incoming_connections_kernel, 
                electrical_and_spike_train_incoming_connections: generate_network_spike_train_electrical_inputs_kernel::<R>(&context)?,
                chemical_incoming_connections_kernel, 
                chemical_and_spike_train_incoming_connections,
                grid_history_kernel: V::get_kernel(&context)?,
                spike_train_grid_history_kernel: X::get_kernel(&context)?,
                context, 
                queue, 
                electrical_synapse: lattice_network.electrical_synapse, 
                chemical_synapse: lattice_network.chemical_synapse, 
                internal_clock: lattice_network.internal_clock
            }
        )
    }

    // Generates a GPU lattice network from a given lattice network
    pub fn from_network(
        lattice_network: LatticeNetwork<T, U, V, W, X, C, Y, N>,
    ) -> Result<Self, GPUError> {
        let platforms = get_platforms().map_err(|_| GPUError::GetDeviceFailure)?;

        for platform in platforms {
            if let Ok(devices) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                if let Some(&device_id) = devices.first() {
                    let device = Device::new(device_id);
                    return Self::from_network_given_device(lattice_network, &device);
                }
            }
        }

        Err(GPUError::GetDeviceFailure)
    }

    /// Attempts to generate a default configuration of [`LatticeNetworkGPU`]
    pub fn try_default() -> Result<Self, GPUError> {
        LatticeNetworkGPU::from_network(LatticeNetwork::default())
    }

    /// Sets the timestep variable for each neuron, spike train, and plasticity modulator to `dt`
    pub fn set_dt(&mut self, dt: f32) {
        self.lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.set_dt(dt));
    }
    
    /// Resets the clock and last firing times for the entire network
    pub fn reset_timing(&mut self) {
        self.internal_clock = 0;

        self.lattices.values_mut()
            .for_each(|i| i.reset_timing());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.reset_timing());
    }
    
    /// Resets all grid histories in the network
    pub fn reset_grid_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
        self.spike_train_lattices.values_mut()
            .for_each(|i| i.grid_history.reset());
    }

    /// Resets all graph histories in the network (including connecting graph)
    pub fn reset_graph_history(&mut self) {
        self.lattices.values_mut()
            .for_each(|i| i.graph.reset_history());

        self.connecting_graph.reset_history();
    }

    /// Returns the set of [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values(&self) -> Values<usize, Lattice<T, U, V, Y, N>> {
        self.lattices.values()
    }

    /// Returns a mutable set [`Lattice`]s in the hashmap of lattices
    pub fn lattices_values_mut(&mut self) -> ValuesMut<usize, Lattice<T, U, V, Y, N>> {
        self.lattices.values_mut()
    }

    /// Returns an immutable set of [`Lattice`]s
    pub fn get_lattices(&self) -> &HashMap<usize, Lattice<T, U, V, Y, N>> {
        &self.lattices
    }

    /// Returns a reference to [`Lattice`] given the identifier
    pub fn get_lattice(&self, id: &usize) -> Option<&Lattice<T, U, V, Y, N>> {
        self.lattices.get(id)
    }

    /// Returns a mutable reference to a [`Lattice`] given the identifier
    pub fn get_mut_lattice(&mut self, id: &usize) -> Option<&mut Lattice<T, U, V, Y, N>> {
        self.lattices.get_mut(id)
    }

    /// Returns an immutable set of [`SpikeTrainLattice`]s
    pub fn get_spike_train_lattices(&self) -> &HashMap<usize, SpikeTrainLattice<N, W, X>> {
        &self.spike_train_lattices
    }

    /// Returns a reference to [`SpikeTrainLattice`] given the identifier
    pub fn get_spike_train_lattice(&self, id: &usize) -> Option<&SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get(id)
    }

    /// Returns a mutable reference to a [`SpikeTrainLattice`] given the identifier
    pub fn get_mut_spike_train_lattice(&mut self, id: &usize) -> Option<&mut SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.get_mut(id)
    }

    /// Returns the set of [`SpikeTrainLattice`]s in the hashmap of spike train lattices
    pub fn spike_trains_values(&self) -> Values<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values()
    }

    /// Returns a mutable set [`SpikeTrainLattice`]s in the hashmap of spike train lattices    
    pub fn spike_trains_values_mut(&mut self) -> ValuesMut<usize, SpikeTrainLattice<N, W, X>> {
        self.spike_train_lattices.values_mut()
    }

    /// Returns an immutable reference to the connecting graph
    pub fn get_connecting_graph(&self) -> &C {
        &self.connecting_graph
    }

    /// Returns a hashset of each [`Lattice`] id
    pub fn get_all_lattice_ids(&self) -> HashSet<usize> {
        self.lattices.keys().cloned().collect()
    }

    /// Returns a hashset of each [`SpikeTrainLattice`] id
    pub fn get_all_spike_train_lattice_ids(&self) -> HashSet<usize> {
        self.spike_train_lattices.keys().cloned().collect()
    } 

    /// Returns a hashset of all the ids
    pub fn get_all_ids(&self) -> HashSet<usize> {
        let mut ids = HashSet::new();

        self.lattices.keys()
            .for_each(|i| { ids.insert(*i); });
        self.spike_train_lattices.keys()
            .for_each(|i| { ids.insert(*i); });

        ids
    }

    /// Adds a [`Lattice`] to the network if the lattice has an id that is not already in the network
    pub fn add_lattice(
        &mut self, 
        lattice: Lattice<T, U, V, Y, N>
    ) -> Result<(), LatticeNetworkError> {
        let current_id = lattice.get_id();

        if self.get_all_ids().contains(&current_id) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(current_id));
        }
        self.lattices.insert(current_id, lattice);

        let lattice_ref = self.lattices.get_mut(&current_id).unwrap();
        lattice_ref.in_network = true;

        Ok(())
    }

    /// Adds a [`SpikeTrainLattice`] to the network if the lattice has an id that is 
    /// not already in the network
    pub fn add_spike_train_lattice(
        &mut self, 
        spike_train_lattice: SpikeTrainLattice<N, W, X>, 
    ) -> Result<(), LatticeNetworkError> {
        let current_id = spike_train_lattice.id;

        if self.get_all_ids().contains(&current_id) {
            return Err(LatticeNetworkError::GraphIDAlreadyPresent(current_id));
        }

        self.spike_train_lattices.insert(current_id, spike_train_lattice);

        let spike_train_lattice_ref = self.spike_train_lattices.get_mut(&current_id).unwrap();
        spike_train_lattice_ref.in_network = true;

        Ok(())
    }

    /// Clears all lattices and spike train lattices, also resets connecting graph and timing
    pub fn clear(&mut self) {
        self.lattices = HashMap::new();
        self.spike_train_lattices = HashMap::new();
        self.connecting_graph = C::default();
        self.reset_timing();
        self.reset_graph_history();
    }

    /// Sets the connecting graph to a new graph, (id remains the same before and after),
    /// also verifies if graph is valid
    pub fn set_connecting_graph(&mut self, new_graph: C) -> Result<(), SpikingNeuralNetworksError> {
        let id = self.connecting_graph.get_id();

        for graph_pos in new_graph.get_every_node_as_ref() {
            if let Some(lattice) = self.lattices.get(&graph_pos.id) {
                check_position(&lattice.cell_grid, graph_pos)?;
            } else if let Some(spike_train_lattice) = self.spike_train_lattices.get(&graph_pos.id) {
                check_position(&spike_train_lattice.cell_grid, graph_pos)?;
            } else {
                return Err(SpikingNeuralNetworksError::from(
                    LatticeNetworkError::IDNotFoundInLattices(graph_pos.id),
                ));
            }
        }
    
        self.connecting_graph = new_graph;
        self.connecting_graph.set_id(id);
    
        Ok(())
    }

    /// Connects the neurons in lattices together given a function to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            self.connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
        }

        if self.lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id)
                .unwrap()
                .graph;
            self.lattices.get(&presynaptic_id).unwrap()
                .graph
                .get_every_node()
                .iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(*i, *j));
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
                        }
                    }
                });
        } else {
            let presynaptic_positions = self.spike_train_lattices.get(&presynaptic_id)
                .unwrap()
                .cell_grid
                .iter()
                .enumerate()
                .flat_map(|(n1, i)| {
                    i.iter()
                        .enumerate()
                        .map(move |(n2, _)| 
                            GraphPosition {
                                id: presynaptic_id, 
                                pos: (n1, n2),
                            }
                        )
                })
                .collect::<Vec<GraphPosition>>();

            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id).unwrap().graph;

            presynaptic_positions.iter()
                .for_each(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j) {
                            let weight = weight_logic.map_or(1., |logic| (logic)(i.pos, *j));
                            self.connecting_graph.edit_weight(i, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }
                });
        }

        Ok(())
    }

    /// Connects the neurons in lattices together given a function (that can fail) to determine
    /// if the neurons should be connected given their position (usize, usize), and
    /// a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// `presynaptic_id` refers to the lattice that should contain the presynaptic neurons
    /// (which can be a [`Lattice`] or a [`SpikeTrainLattice`]) and `postsynaptic_id` refers
    /// to the lattice that should contain the postsynaptic connections ([`Lattice`] only),
    /// any pre-existing connections in the given direction (presynaptic -> postsynaptic)
    /// will be overwritten based on the rule given in `connecting_conditional`
    pub fn falliable_connect(
        &mut self, 
        presynaptic_id: usize, 
        postsynaptic_id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if self.spike_train_lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticLatticeCannotBeSpikeTrain);
        }

        if !self.get_all_ids().contains(&presynaptic_id) {
            return Err(LatticeNetworkError::PresynapticIDNotFound(presynaptic_id));
        }

        if !self.lattices.contains_key(&postsynaptic_id) {
            return Err(LatticeNetworkError::PostsynapticIDNotFound(postsynaptic_id));
        }

        if presynaptic_id == postsynaptic_id {
            self.falliable_connect_interally(presynaptic_id, connecting_conditional, weight_logic)?;
            return Ok(());
        }

        let output: Result<Vec<_>, LatticeNetworkError> = if self.lattices.contains_key(&presynaptic_id) {
            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id)
                .unwrap()
                .graph;
            self.lattices.get(&presynaptic_id).unwrap()
                .graph
                .get_every_node()
                .iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let i_graph_pos = GraphPosition { id: presynaptic_id, pos: *i};
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(i_graph_pos);
                        self.connecting_graph.add_node(j_graph_pos);

                        if (connecting_conditional)(*i, *j)? {
                            let weight = match weight_logic {
                                Some(logic) => logic(*i, *j)?,
                                None => 1.0,
                            };
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(&i_graph_pos, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
        } else {
            let presynaptic_positions = self.spike_train_lattices.get(&presynaptic_id)
                .unwrap()
                .cell_grid
                .iter()
                .enumerate()
                .flat_map(|(n1, i)| {
                    i.iter()
                        .enumerate()
                        .map(move |(n2, _)| 
                            GraphPosition {
                                id: presynaptic_id, 
                                pos: (n1, n2),
                            }
                        )
                })
                .collect::<Vec<GraphPosition>>();

            let postsynaptic_graph = &self.lattices.get(&postsynaptic_id).unwrap().graph;

            presynaptic_positions.iter()
                .map(|i| {
                    for j in postsynaptic_graph.get_every_node().iter() {
                        let j_graph_pos = GraphPosition { id: postsynaptic_id, pos: *j};
                        self.connecting_graph.add_node(*i);
                        self.connecting_graph.add_node(j_graph_pos);
                        
                        if (connecting_conditional)(i.pos, *j)? {
                            let weight = match weight_logic {
                                Some(logic) => logic(i.pos, *j)?,
                                None => 1.0,
                            };
                            self.connecting_graph.edit_weight(i, &j_graph_pos, Some(weight)).unwrap();
                        } else {
                            self.connecting_graph.edit_weight(i, &j_graph_pos, None).unwrap();
                        }
                    }

                    Ok(())
                })
                .collect()
        };

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetworkGPU`] together given a 
    /// function to determine if the neurons should be connected given their position (usize, usize), 
    /// and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> bool,
        weight_logic: Option<&dyn Fn(Position, Position) -> f32>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().connect(connecting_conditional, weight_logic);

        Ok(())
    }

    /// Connects the neurons in a [`Lattice`] within the [`LatticeNetworkGPU`] together given a 
    /// function (that can fail) to determine if the neurons should be connected given their position (usize, usize), 
    /// and a function to determine what the weight between the neurons should be,
    /// if the `weight_logic` function is `None`, the weights are set as `1.`
    /// if a connect should occur according to `connecting_conditional`,
    /// assumes lattice is already populated using the `populate` method
    pub fn falliable_connect_interally(
        &mut self, 
        id: usize, 
        connecting_conditional: &dyn Fn(Position, Position) -> Result<bool, LatticeNetworkError>,
        weight_logic: Option<&dyn Fn(Position, Position) -> Result<f32, LatticeNetworkError>>,
    ) -> Result<(), LatticeNetworkError> {
        if !self.lattices.contains_key(&id) {
            return Err(LatticeNetworkError::IDNotFoundInLattices(id));
        }

        self.lattices.get_mut(&id).unwrap().falliable_connect(connecting_conditional, weight_logic)?;

        Ok(())
    }

    unsafe fn execute_last_firing_time(
        &self, 
        gpu_graph: &InterleavingGraphGPU, 
        gpu_cell_grid: &HashMap<String, BufferGPU>, 
        size: usize,
    ) -> Result<(), GPUError> {
        let last_firing_time_event = unsafe {
            match ExecuteKernel::new(&self.last_firing_time_kernel)
                .set_arg(&gpu_graph.index_to_position)
                .set_arg(
                    match &gpu_cell_grid.get("is_spiking").expect("Could not retrieve buffer: is_spiking") {
                        BufferGPU::UInt(buffer) => buffer,
                        _ => unreachable!("is_spiking cannot be float or optional unsigned integer"),
                    }
                )
                .set_arg(
                    match &gpu_cell_grid.get("last_firing_time").expect("Could not retrieve buffer: last_firing_time") {
                        BufferGPU::OptionalUInt(buffer) => buffer,
                        _ => unreachable!("last_firing_time cannot be float or mandatory unsigned integer"),
                    }
                )
                .set_arg(&(self.internal_clock as i32))
                .set_global_work_size(size)
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
        };

        match last_firing_time_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(())
    }

    unsafe fn execute_grid_history(
        &self, 
        skip_index: u32,
        size: u32,
        kernel: &KernelFunction,
        gpu_graph: &InterleavingGraphGPU, 
        gpu_cell_grid: &HashMap<String, BufferGPU>, 
        gpu_grid_history: &HashMap<String, BufferGPU>,
    ) -> Result<(), GPUError> {
        let update_history_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(&kernel.kernel);

            for i in kernel.argument_names.iter() {
                if i == "iteration" {
                    kernel_execution.set_arg(&(self.internal_clock as i32));
                } else if i == "size" {
                    kernel_execution.set_arg(&size);
                } else if i == "skip_index" {
                    kernel_execution.set_arg(&skip_index);
                } else if i == "index_to_position" {
                    kernel_execution.set_arg(&gpu_graph.index_to_position);
                } else if gpu_cell_grid.contains_key(i) {
                    match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    };
                } else if gpu_grid_history.contains_key(i) {
                    match &gpu_grid_history.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    };
                } else {
                    unreachable!("Unkown argument in history kernel");
                }
            }

            match kernel_execution.set_global_work_size(size as usize)
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
        };

        match update_history_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(())
    }

    unsafe fn execute_gap_junctions(
        &self,
        sums_buffer: &Buffer<cl_float>,
        gpu_cell_grid: &HashMap<String, BufferGPU>,
        gpu_graph: &InterleavingGraphGPU,
    ) -> Result<(), GPUError> {
        let gap_junctions_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(&self.electrical_incoming_connections_kernel);

            kernel_execution.set_arg(&gpu_graph.connections)
                .set_arg(&gpu_graph.weights)
                .set_arg(&gpu_graph.index_to_position);

            match &gpu_cell_grid.get("gap_conductance").expect("Could not retrieve buffer: gap_conductance") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("gap_condutance must be float"),
            };

            match &gpu_cell_grid.get("current_voltage").expect("Could not retrieve buffer: current_voltage") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("current_voltage must be float"),
            };

            match kernel_execution.set_arg(&(gpu_graph.size as u32))
                .set_arg(sums_buffer)
                .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
        };

        match gap_junctions_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(())
    }

    unsafe fn execute_gap_junctions_with_spike_train(
        &self,
        sums_buffer: &Buffer<cl_float>,
        spike_train_skip_index: u32,
        gpu_cell_grid: &HashMap<String, BufferGPU>,
        gpu_spike_train_grid: &HashMap<String, BufferGPU>,
        gpu_graph: &InterleavingGraphGPU,
    ) -> Result<(), GPUError> {
        let gap_junctions_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(
                &self.electrical_and_spike_train_incoming_connections.kernel
            );

            for i in &self.electrical_and_spike_train_incoming_connections.argument_names {
                if i == "weights" {
                    kernel_execution.set_arg(&gpu_graph.weights);
                } else if i == "connections" {
                    kernel_execution.set_arg(&gpu_graph.connections);
                } else if i == "index_to_position" {
                    kernel_execution.set_arg(&gpu_graph.index_to_position);
                } else if i == "is_spike_train" {
                    kernel_execution.set_arg(&gpu_graph.is_spike_train);
                } else if i == "gap_conductances" {
                    match &gpu_cell_grid.get("gap_conductance").unwrap() {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("gap_conductance must be a float buffer")
                    };
                } else if i == "voltages" {
                    match &gpu_cell_grid.get("current_voltage").unwrap() {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("current_voltage must be a float buffer")
                    };
                } else if i == "timestep" {
                    kernel_execution.set_arg(&(self.internal_clock as i32));
                } else if i == "skip_index" {
                    kernel_execution.set_arg(&spike_train_skip_index);
                } else if i == "n" {
                    kernel_execution.set_arg(&(gpu_graph.size as u32));
                } else if i == "res" {
                    kernel_execution.set_arg(sums_buffer);
                } else {
                    match &gpu_spike_train_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                    };
                }
            }

            match kernel_execution.set_global_work_size(spike_train_skip_index as usize)
                .enqueue_nd_range(&self.queue) {
                    Ok(value) => value,
                    Err(_) => return Err(GPUError::QueueFailure),
                }
            };

            match gap_junctions_event.wait() {
                Ok(_) => {},
                Err(_) => return Err(GPUError::WaitError),
            };
        
        Ok(())
    }

    fn run_lattices_with_electrical_synapses(&mut self, iterations: usize) -> Result<(), GPUError> {
        // concat cell grids into a single 1d vector of cell grids
        // create a new vector that keeps track of which lattices and positions each cell belongs to
        // use those vectors to write the new values back to the cpu and execute the connections kernel
        // connections kernel should take each weight matrix in and use lattice size to index where to go

        let (
            lattice_ids, 
            lattice_sizes_map, 
            mut cell_vector, 
            cell_vector_size
        ) = self.generate_cell_grid_vector();

        let gpu_cell_grid = T::convert_to_gpu(&cell_vector, &self.context, &self.queue)?;

        let (
            spike_train_lattice_ids, 
            spike_train_lattice_sizes_map, 
            mut spike_train_cell_vector, 
            spike_train_vector_size
        ) = self.generate_spike_train_grid_vector();

        let gpu_spike_train_grid = W::convert_to_gpu(&spike_train_cell_vector, &self.context, &self.queue)?;

        let gpu_graph = InterleavingGraphGPU::convert_to_gpu(
            &self.context, &self.queue, &self.lattices, &self.spike_train_lattices, &self.connecting_graph
        )?;

        let iterate_kernel = T::iterate_and_spike_electrical_kernel(&self.context)?;
        let spike_train_iterate_kernel = W::spike_train_electrical_kernel(&self.context)?;

        let sums_buffer = create_and_write_buffer(&self.context, &self.queue, gpu_graph.size, 0.0)?;

        let mut gpu_grid_histories = HashMap::new();
        let mut spike_train_gpu_grid_histories = HashMap::new();

        for (key, value) in &self.lattices {
            if value.update_grid_history {
                let rows = value.cell_grid().len();
                let cols = value.cell_grid().first().unwrap_or(&vec![]).len();

                gpu_grid_histories.insert(
                    key,
                    value.grid_history.clone().to_gpu(&self.context, iterations, (rows, cols))?,
                );
            }
        }

        for (key, value) in &self.spike_train_lattices {
            if value.update_grid_history {
                let rows = value.spike_train_grid().len();
                let cols = value.spike_train_grid().first().unwrap_or(&vec![]).len();

                spike_train_gpu_grid_histories.insert(
                    key,
                    value.grid_history.clone().to_gpu(&self.context, iterations, (rows, cols))?,
                );
            }
        }

        let lattices_exist = !self.lattices.is_empty();
        let spike_train_lattices_exist = !self.spike_train_lattices.is_empty();
        let spike_train_skip_index = gpu_graph.lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>() as u32;
        let spike_train_size: usize = gpu_graph.spike_train_lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum();

        for _ in 0..iterations {
            if lattices_exist && !spike_train_lattices_exist {
                unsafe {
                    self.execute_gap_junctions(
                        &sums_buffer,
                        &gpu_cell_grid,
                        &gpu_graph,
                    )?;
                }
            } else if lattices_exist && spike_train_lattices_exist {
                unsafe {
                    self.execute_gap_junctions_with_spike_train(
                        &sums_buffer,
                        spike_train_skip_index,
                        &gpu_cell_grid,
                        &gpu_spike_train_grid,
                        &gpu_graph,
                    )?;
                }
            }

            if lattices_exist {
                // only execute if there exists lattices, same with spike train lattices
                let iterate_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                    for i in iterate_kernel.argument_names.iter() {
                        if i == "inputs" {
                            kernel_execution.set_arg(&sums_buffer);
                        } else if i == "index_to_position" {
                            kernel_execution.set_arg(&gpu_graph.index_to_position);
                        } else {
                            match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            };
                        }
                    }

                    match kernel_execution.set_global_work_size(spike_train_skip_index as usize)
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match iterate_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };

                unsafe {
                    self.execute_last_firing_time(
                        &gpu_graph, 
                        &gpu_cell_grid, 
                        spike_train_skip_index as usize
                    )?
                };

                for (key, value) in self.lattices.iter() {
                    if value.update_grid_history {
                        let mut skip_index = 0;
                        for i in gpu_graph.ordered_keys.iter() {
                            if *i >= *key {
                                break;
                            }
                            let current_size = gpu_graph.lattice_sizes_map.get(i).unwrap();
                            skip_index += current_size.0 * current_size.1;
                        }

                        let dim = &gpu_graph.lattice_sizes_map.get(key).unwrap();
                        let current_size = dim.0 * dim.1;
                        
                        unsafe {
                            self.execute_grid_history(
                                skip_index as u32,
                                current_size as u32,
                                &self.grid_history_kernel,
                                &gpu_graph, 
                                &gpu_cell_grid, 
                                gpu_grid_histories.get(key).unwrap()
                            )?;
                        }
                    }
                }
            }

            if spike_train_lattices_exist {
                let iterate_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&spike_train_iterate_kernel.kernel);

                    for i in spike_train_iterate_kernel.argument_names.iter() {
                        if i == "number_of_types" {
                            kernel_execution.set_arg(&(N::number_of_types() as u32));
                        } else if i == "index_to_position" {
                            kernel_execution.set_arg(&gpu_graph.index_to_position);
                        } else if i == "skip_index" { 
                            kernel_execution.set_arg(&spike_train_skip_index);
                        } else if i == "neuro_flags" {
                            match &gpu_spike_train_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                                _ => unreachable!("Could not retrieve neurotransmitter flags"),
                            };
                        } else {
                            match &gpu_spike_train_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            };
                        }
                    }

                    match kernel_execution.set_global_work_size(spike_train_size)
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match iterate_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };

                unsafe {
                    self.execute_last_firing_time(
                        &gpu_graph, 
                        &gpu_spike_train_grid, 
                        spike_train_size,
                    )?
                };

                for (key, value) in self.spike_train_lattices.iter() {
                    if value.update_grid_history {
                        let mut skip_index = 0;
                        for i in gpu_graph.spike_train_ordered_keys.iter() {
                            if *i >= *key {
                                break;
                            }
                            let current_size = gpu_graph.spike_train_lattice_sizes_map.get(i).unwrap();
                            skip_index += current_size.0 * current_size.1;
                        }

                        let dim = &gpu_graph.spike_train_lattice_sizes_map.get(key).unwrap();
                        let current_size = dim.0 * dim.1;
                        
                        unsafe {
                            self.execute_grid_history(
                                skip_index as u32,
                                current_size as u32,
                                &self.spike_train_grid_history_kernel,
                                &gpu_graph, 
                                &gpu_spike_train_grid, 
                                spike_train_gpu_grid_histories.get(key).unwrap()
                            )?;
                        }
                    }
                }
            }
            
            self.internal_clock += 1;
        }

        T::convert_to_cpu(
            &mut cell_vector, 
            &gpu_cell_grid, 
            1, 
            cell_vector_size, 
            &self.queue
        )?;

        let reshaped_grids = Self::consolidate_neurons(lattice_ids, lattice_sizes_map, cell_vector);

        let updates = Self::consolidate_neuron_histories(&self.lattices, gpu_grid_histories);

        for (key, (rows, cols), grid_history) in updates {
            let value = self.lattices.get_mut(&key).unwrap();
            value.grid_history.add_from_gpu(
                &self.queue,
                grid_history,
                iterations,
                (rows, cols),
            )?;
        }

        for (key, value) in self.lattices.iter_mut() {
            value.set_cell_grid(reshaped_grids.get(key).unwrap().clone()).expect("Same dimensions");
        }

        W::convert_to_cpu(
            &mut spike_train_cell_vector, 
            &gpu_spike_train_grid, 
            1, 
            spike_train_vector_size, 
            &self.queue
        )?;

        let spike_train_reshaped_grids = Self::consolidate_spike_trains(
            spike_train_lattice_ids, spike_train_lattice_sizes_map, spike_train_cell_vector
        );

        let updates = Self::consolidate_spike_train_histories(
            &self.spike_train_lattices, spike_train_gpu_grid_histories
        );

        for (key, (rows, cols), grid_history) in updates {
            let value = self.spike_train_lattices.get_mut(&key).unwrap();
            value.grid_history.add_from_gpu(
                &self.queue,
                grid_history,
                iterations,
                (rows, cols),
            )?;
        }

        for (key, value) in self.spike_train_lattices.iter_mut() {
            value.set_spike_train_grid(spike_train_reshaped_grids.get(key).unwrap().clone())
                .expect("Same dimensions");
        }

        InterleavingGraphGPU::convert_to_cpu(
            &self.queue, &gpu_graph, &mut self.lattices, &mut self.spike_train_lattices, &mut self.connecting_graph
        )?;

        Ok(())
    }

    fn run_lattices_with_chemical_synapses(&mut self, iterations: usize) -> Result<(), GPUError> {
        let (
            lattice_ids, 
            lattice_sizes_map, 
            mut cell_vector, 
            cell_vector_size
        ) = self.generate_cell_grid_vector();

        let gpu_cell_grid = T::convert_electrochemical_to_gpu(&cell_vector, &self.context, &self.queue)?;

        let (
            spike_train_lattice_ids, 
            spike_train_lattice_sizes_map, 
            mut spike_train_cell_vector, 
            spike_train_vector_size
        ) = self.generate_spike_train_grid_vector();

        let gpu_spike_train_grid = W::convert_electrochemical_to_gpu(&spike_train_cell_vector, &self.context, &self.queue)?;

        let gpu_graph = InterleavingGraphGPU::convert_to_gpu(
            &self.context, &self.queue, &self.lattices, &self.spike_train_lattices, &self.connecting_graph
        )?;

        let iterate_kernel = T::iterate_and_spike_electrochemical_kernel(&self.context)?;
        let spike_train_iterate_kernel = W::spike_train_electrochemical_kernel(&self.context)?;

        let sums_buffer = create_and_write_buffer(&self.context, &self.queue, gpu_graph.size, 0.0)?;

        let t_sums_buffer = create_and_write_buffer(
            &self.context, &self.queue, gpu_graph.size * N::number_of_types(), 0.0
        )?;

        let counts_buffer = create_and_write_buffer(
            &self.context, &self.queue, gpu_graph.size * N::number_of_types(), 0.0
        )?;

        let mut gpu_grid_histories = HashMap::new();
        let mut spike_train_gpu_grid_histories = HashMap::new();

        for (key, value) in &self.lattices {
            if value.update_grid_history {
                let rows = value.cell_grid().len();
                let cols = value.cell_grid().first().unwrap_or(&vec![]).len();

                gpu_grid_histories.insert(
                    key,
                    value.grid_history.clone().to_gpu(&self.context, iterations, (rows, cols))?,
                );
            }
        }

        for (key, value) in &self.spike_train_lattices {
            if value.update_grid_history {
                let rows = value.spike_train_grid().len();
                let cols = value.spike_train_grid().first().unwrap_or(&vec![]).len();

                spike_train_gpu_grid_histories.insert(
                    key,
                    value.grid_history.clone().to_gpu(&self.context, iterations, (rows, cols))?,
                );
            }
        }

        let lattices_exist = !self.lattices.is_empty();
        let spike_train_lattices_exist = !self.spike_train_lattices.is_empty();
        let spike_train_skip_index = gpu_graph.lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>() as u32;
        let spike_train_size: usize = gpu_graph.spike_train_lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum();

        for _ in 0..iterations {
            if lattices_exist && !spike_train_lattices_exist {
                if self.electrical_synapse {
                    unsafe {
                        self.execute_gap_junctions(
                            &sums_buffer,
                            &gpu_cell_grid,
                            &gpu_graph,
                        )?;
                    }
                }

                let chemical_synapses_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&self.chemical_incoming_connections_kernel);

                    kernel_execution.set_arg(&gpu_graph.connections)
                        .set_arg(&gpu_graph.weights)
                        .set_arg(&gpu_graph.index_to_position);

                    match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$flags"),
                    };

                    match &gpu_cell_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$t must be float"),
                    };

                    kernel_execution.set_arg(&(gpu_graph.size as u32))
                        .set_arg(&(N::number_of_types() as u32));

                    match kernel_execution
                        .set_arg(&counts_buffer)
                        .set_arg(&t_sums_buffer)
                        .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match chemical_synapses_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };
            } else if lattices_exist && spike_train_lattices_exist {
                if self.electrical_synapse {
                    unsafe {
                        self.execute_gap_junctions_with_spike_train(
                            &sums_buffer,
                            spike_train_skip_index,
                            &gpu_cell_grid,
                            &gpu_spike_train_grid,
                            &gpu_graph,
                        )?;
                    }
                }

                let chemical_synapses_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&self.chemical_and_spike_train_incoming_connections);

                    kernel_execution.set_arg(&gpu_graph.connections)
                        .set_arg(&gpu_graph.weights)
                        .set_arg(&gpu_graph.index_to_position)
                        .set_arg(&gpu_graph.is_spike_train);

                    match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$flags"),
                    };

                    match &gpu_cell_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$t must be float"),
                    };

                    match &gpu_spike_train_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                        BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$flags"),
                    };

                    match &gpu_spike_train_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                        BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                        _ => unreachable!("neurotransmitters$t must be float"),
                    };

                    kernel_execution.set_arg(&spike_train_skip_index)
                        .set_arg(&(gpu_graph.size as u32))
                        .set_arg(&(N::number_of_types() as u32));

                    match kernel_execution
                        .set_arg(&counts_buffer)
                        .set_arg(&t_sums_buffer)
                        .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match chemical_synapses_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };
            }

            if lattices_exist {
                let iterate_event = unsafe {
                    let mut kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

                    for i in iterate_kernel.argument_names.iter() {
                        if i == "inputs" {
                            kernel_execution.set_arg(&sums_buffer);
                        } else if i == "t" {
                            kernel_execution.set_arg(&t_sums_buffer);
                        } else if i == "index_to_position" {
                            kernel_execution.set_arg(&gpu_graph.index_to_position);
                        } else if i == "number_of_types" {
                            kernel_execution.set_arg(&(N::number_of_types() as u32));
                        } else if i == "neuro_flags" {
                            match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                                _ => unreachable!("Could not retrieve neurotransmitter flags"),
                            };
                        } else if i == "lg_flags" {
                            match &gpu_cell_grid.get("ligand_gates$flags").expect("Could not retrieve receptor flags") {
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                                _ => unreachable!("Could not retrieve receptor flags"),
                            };
                        } else {
                            match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                            };
                        }
                    }

                    match kernel_execution.set_global_work_size(gpu_graph.size)
                        .enqueue_nd_range(&self.queue) {
                            Ok(value) => value,
                            Err(_) => return Err(GPUError::QueueFailure),
                        }
                };

                match iterate_event.wait() {
                    Ok(_) => {},
                    Err(_) => return Err(GPUError::WaitError),
                };

                unsafe {
                    self.execute_last_firing_time(
                        &gpu_graph, 
                        &gpu_cell_grid, 
                        spike_train_skip_index as usize
                    )?
                };

                for (key, value) in self.lattices.iter() {
                    if value.update_grid_history {
                        let mut skip_index = 0;
                        for i in gpu_graph.ordered_keys.iter() {
                            if *i >= *key {
                                break;
                            }
                            let current_size = gpu_graph.lattice_sizes_map.get(i).unwrap();
                            skip_index += current_size.0 * current_size.1;
                        }

                        let dim = &gpu_graph.lattice_sizes_map.get(key).unwrap();
                        let current_size = dim.0 * dim.1;
                        
                        unsafe {
                            self.execute_grid_history(
                                skip_index as u32,
                                current_size as u32,
                                &self.grid_history_kernel,
                                &gpu_graph, 
                                &gpu_cell_grid, 
                                gpu_grid_histories.get(key).unwrap()
                            )?;
                        }
                    }
                }

                if spike_train_lattices_exist {
                    let iterate_event = unsafe {
                        let mut kernel_execution = ExecuteKernel::new(&spike_train_iterate_kernel.kernel);
    
                        for i in spike_train_iterate_kernel.argument_names.iter() {
                            if i == "number_of_types" {
                                kernel_execution.set_arg(&(N::number_of_types() as u32));
                            } else if i == "index_to_position" {
                                kernel_execution.set_arg(&gpu_graph.index_to_position);
                            } else if i == "skip_index" { 
                                kernel_execution.set_arg(&spike_train_skip_index);
                            } else if i == "neuro_flags" {
                                match &gpu_spike_train_grid.get("neurotransmitters$flags").expect("Could not retrieve neurotransmitter flags") {
                                    BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                                    _ => unreachable!("Could not retrieve neurotransmitter flags"),
                                };
                            } else {
                                match &gpu_spike_train_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
                                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                                    BufferGPU::OptionalUInt(buffer) => kernel_execution.set_arg(buffer),
                                    BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                                };
                            }
                        }
    
                        match kernel_execution.set_global_work_size(spike_train_size)
                            .enqueue_nd_range(&self.queue) {
                                Ok(value) => value,
                                Err(_) => return Err(GPUError::QueueFailure),
                            }
                    };
    
                    match iterate_event.wait() {
                        Ok(_) => {},
                        Err(_) => return Err(GPUError::WaitError),
                    };
    
                    unsafe {
                        self.execute_last_firing_time(
                            &gpu_graph, 
                            &gpu_spike_train_grid, 
                            spike_train_size,
                        )?
                    };
    
                    for (key, value) in self.spike_train_lattices.iter() {
                        if value.update_grid_history {
                            let mut skip_index = 0;
                            for i in gpu_graph.spike_train_ordered_keys.iter() {
                                if *i >= *key {
                                    break;
                                }
                                let current_size = gpu_graph.spike_train_lattice_sizes_map.get(i).unwrap();
                                skip_index += current_size.0 * current_size.1;
                            }
    
                            let dim = &gpu_graph.spike_train_lattice_sizes_map.get(key).unwrap();
                            let current_size = dim.0 * dim.1;
                            
                            unsafe {
                                self.execute_grid_history(
                                    skip_index as u32,
                                    current_size as u32,
                                    &self.spike_train_grid_history_kernel,
                                    &gpu_graph, 
                                    &gpu_spike_train_grid, 
                                    spike_train_gpu_grid_histories.get(key).unwrap()
                                )?;
                            }
                        }
                    }
                }
            }
            
            self.internal_clock += 1;
        }

        T::convert_electrochemical_to_cpu(
            &mut cell_vector, 
            &gpu_cell_grid, 
            1, 
            cell_vector_size, 
            &self.queue
        )?;

        let reshaped_grids = Self::consolidate_neurons(lattice_ids, lattice_sizes_map, cell_vector);

        let updates = Self::consolidate_neuron_histories(&self.lattices, gpu_grid_histories);

        for (key, (rows, cols), grid_history) in updates {
            let value = self.lattices.get_mut(&key).unwrap();
            value.grid_history.add_from_gpu(
                &self.queue,
                grid_history,
                iterations,
                (rows, cols),
            )?;
        }

        for (key, value) in self.lattices.iter_mut() {
            value.set_cell_grid(reshaped_grids.get(key).unwrap().clone()).expect("Same dimensions");
        }

        W::convert_electrochemical_to_cpu(
            &mut spike_train_cell_vector, 
            &gpu_spike_train_grid, 
            1, 
            spike_train_vector_size, 
            &self.queue
        )?;

        let spike_train_reshaped_grids = Self::consolidate_spike_trains(
            spike_train_lattice_ids, spike_train_lattice_sizes_map, spike_train_cell_vector
        );

        let updates = Self::consolidate_spike_train_histories(
            &self.spike_train_lattices, spike_train_gpu_grid_histories
        );

        for (key, (rows, cols), grid_history) in updates {
            let value = self.spike_train_lattices.get_mut(&key).unwrap();
            value.grid_history.add_from_gpu(
                &self.queue,
                grid_history,
                iterations,
                (rows, cols),
            )?;
        }

        for (key, value) in self.spike_train_lattices.iter_mut() {
            value.set_spike_train_grid(spike_train_reshaped_grids.get(key).unwrap().clone())
                .expect("Same dimensions");
        }

        InterleavingGraphGPU::convert_to_cpu(
            &self.queue, &gpu_graph, &mut self.lattices, &mut self.spike_train_lattices, &mut self.connecting_graph
        )?;

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn consolidate_spike_train_histories(
        spike_train_lattices: &HashMap<usize, SpikeTrainLattice<N, W, X>>, 
        mut spike_train_gpu_grid_histories: HashMap<&usize, HashMap<String, BufferGPU>>
    ) -> Vec<(usize, (usize, usize), HashMap<String, BufferGPU>)> {
        let updates: Vec<(usize, (usize, usize), _)> = spike_train_lattices.iter()
            .filter_map(|(key, value)| {
                if value.update_grid_history {
                    let rows = value.spike_train_grid().len();
                    let cols = value.spike_train_grid().first().unwrap_or(&vec![]).len();
                    Some((*key, (rows, cols), spike_train_gpu_grid_histories.remove(key).unwrap()))
                } else {
                    None
                }
            })
            .collect();
        updates
    }
    
    #[allow(clippy::type_complexity)]
    fn consolidate_neuron_histories(
        lattices: &HashMap<usize, Lattice<T, U, V, Y, N>>, 
        mut gpu_grid_histories: HashMap<&usize, HashMap<String, BufferGPU>>
    ) -> Vec<(usize, (usize, usize), HashMap<String, BufferGPU>)> {
        let updates: Vec<(usize, (usize, usize), _)> = lattices.iter()
            .filter_map(|(key, value)| {
                if value.update_grid_history {
                    let rows = value.cell_grid().len();
                    let cols = value.cell_grid().first().unwrap_or(&vec![]).len();
                    Some((*key, (rows, cols), gpu_grid_histories.remove(key).unwrap()))
                } else {
                    None
                }
            })
            .collect();

        updates
    }
    
    fn consolidate_neurons(
        lattice_ids: Vec<u32>, 
        lattice_sizes_map: HashMap<usize, (usize, usize)>, 
        cell_vector: Vec<Vec<T>>
    ) -> HashMap<usize, Vec<Vec<T>>> {
        let mut new_grids: HashMap<usize, Vec<T>> = HashMap::new();
    
        if let Some(first_vec) = cell_vector.first() {
            for (id, cell) in lattice_ids.iter().zip(first_vec.iter()) {
                new_grids.entry(*id as usize)
                    .and_modify(|vec| vec.push(cell.clone()))
                    .or_insert_with(|| vec![cell.clone()]);
            }
        }
        // maybe use std::mem::take ?
    
        let mut reshaped_grids = HashMap::new();
    
        for (key, vec) in new_grids {
            if let Some(&(_, cols)) = lattice_sizes_map.get(&key) {
                let reshaped_vec: Vec<Vec<T>> = vec.chunks(cols)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                reshaped_grids.insert(key, reshaped_vec);
            }
        }

        reshaped_grids
    }

    fn consolidate_spike_trains(
        spike_train_lattice_ids: Vec<u32>, 
        spike_train_lattice_sizes_map: HashMap<usize, (usize, usize)>, 
        spike_train_cell_vector: Vec<Vec<W>>
    ) -> HashMap<usize, Vec<Vec<W>>> {
        let mut spike_train_new_grids: HashMap<usize, Vec<W>> = HashMap::new();
    
        if let Some(first_vec) = spike_train_cell_vector.first() {
            for (id, cell) in spike_train_lattice_ids.iter().zip(first_vec.iter()) {
                spike_train_new_grids.entry(*id as usize)
                    .and_modify(|vec| vec.push(cell.clone()))
                    .or_insert_with(|| vec![cell.clone()]);
            }
        }
    
        let mut spike_train_reshaped_grids = HashMap::new();
    
        for (key, vec) in spike_train_new_grids {
            if let Some(&(_, cols)) = spike_train_lattice_sizes_map.get(&key) {
                let reshaped_vec: Vec<Vec<W>> = vec.chunks(cols)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                spike_train_reshaped_grids.insert(key, reshaped_vec);
            }
        }

        spike_train_reshaped_grids
    }

    #[allow(clippy::type_complexity)]
    fn generate_spike_train_grid_vector(&mut self) -> 
    (
        Vec<u32>, 
        HashMap<usize, (usize, usize)>, 
        Vec<Vec<W>>, 
        usize
    ) {
        let mut spike_train_cell_vector: Vec<W> = vec![];
        let mut spike_train_lattice_ids: Vec<u32> = vec![];
        let mut spike_train_lattice_sizes: Vec<u32> = vec![];
        let mut spike_train_lattice_sizes_map: HashMap<usize, (usize, usize)> = HashMap::new();
    
        let mut spike_train_lattice_iterator: Vec<(usize, &SpikeTrainLattice<_, _, _>)> = self.spike_train_lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        spike_train_lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
    
        for (key, value) in &spike_train_lattice_iterator {
            let current_cell_grid = value.spike_train_grid();
            for row in current_cell_grid {
                for i in row {
                    spike_train_cell_vector.push(i.clone());
                    spike_train_lattice_ids.push(*key as u32);
                }
            }
    
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();
            spike_train_lattice_sizes.push((rows * cols) as u32);
            spike_train_lattice_sizes_map.insert(*key, (rows, cols));
        }
    
        let spike_train_cell_vector = vec![spike_train_cell_vector];
        let spike_train_vector_size = spike_train_cell_vector.first().unwrap_or(&vec![]).len();

        (spike_train_lattice_ids, spike_train_lattice_sizes_map, spike_train_cell_vector, spike_train_vector_size)
    }
    
    #[allow(clippy::type_complexity)]
    fn generate_cell_grid_vector(&mut self) -> (
        Vec<u32>, 
        HashMap<usize, (usize, usize)>, 
        Vec<Vec<T>>, 
        usize
    ) {
        let mut cell_vector: Vec<T> = vec![];
        let mut lattice_ids: Vec<u32> = vec![];
        let mut lattice_sizes: Vec<u32> = vec![];
        let mut lattice_sizes_map: HashMap<usize, (usize, usize)> = HashMap::new();
        
        let mut lattice_iterator: Vec<(usize, &Lattice<_, _, _, _, _>)> = self.lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
        
        for (key, value) in &lattice_iterator {
            let current_cell_grid = value.cell_grid();
            for row in current_cell_grid {
                for i in row {
                    cell_vector.push(i.clone());
                    lattice_ids.push(*key as u32);
                }
            }
            
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();
            lattice_sizes.push((rows * cols) as u32);
            lattice_sizes_map.insert(*key, (rows, cols));
        }
        
        let cell_vector = vec![cell_vector];
        let cell_vector_size = cell_vector.first().unwrap_or(&vec![]).len();

        (lattice_ids, lattice_sizes_map, cell_vector, cell_vector_size)
    }
}

impl<T, U, V, W, X, Y, N, R, C> RunNetwork for LatticeNetworkGPU<T, U, V, W, X, Y, N, R, C>
where
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N, U=R>,
    X: SpikeTrainLatticeHistoryGPU,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
    R: NeuralRefractorinessGPU,
    C: Graph<K=GraphPosition, V=f32> + ConnectingGraphToGPU<ConnectingGraphGPU>, 
{
    fn run_lattices(&mut self, iterations: usize) -> Result<(), SpikingNeuralNetworksError> {
        if (self.lattices.is_empty() && self.spike_train_lattices.is_empty()) || iterations == 0 {
            return Ok(());
        }

        if self.lattices.values().all(|i| i.cell_grid.is_empty()) && 
            self.spike_train_lattices.values().all(|i| i.cell_grid.is_empty()) {
                return Ok(());
        }

        match (self.electrical_synapse, self.chemical_synapse) {
            (true, false) => self.run_lattices_with_electrical_synapses(iterations).map_err(Into::into),
            (false, true) => self.run_lattices_with_chemical_synapses(iterations).map_err(Into::into),
            (true, true) => self.run_lattices_with_chemical_synapses(iterations).map_err(Into::into),
            (false, false) => Ok(()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::{cl_float, create_and_write_buffer, SpikingNeuralNetworksError};
    use std::{collections::HashSet, ptr};
    use opencl3::{command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_READ_WRITE}, program::Program, types::{cl_uint, CL_NON_BLOCKING}};
    use rand::Rng;
    use crate::{
        error::GPUError, 
        graph::{AdjacencyMatrix, Graph, GraphPosition, InterleavingGraphGPU}, 
        neuron::{
            gpu_lattices::LatticeNetworkGPU, 
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                aggregate_neurotransmitter_concentrations, weight_neurotransmitter_concentration, 
                AMPAReceptor, ApproximateNeurotransmitter, ApproximateReceptor, BufferGPU, 
                IonotropicNeurotransmitterType, IonotropicType, IterateAndSpike, IterateAndSpikeGPU, 
                NeurotransmitterConcentrations, NeurotransmitterTypeGPU, Receptors, Timestep
            }, 
            plasticity::STDP, 
            spike_train::{DeltaDiracRefractoriness, RateSpikeTrain, SpikeTrain, SpikeTrainGPU},
            GridVoltageHistory, Lattice, LatticeNetwork, SpikeTrainGrid, SpikeTrainGridHistory, SpikeTrainLattice}};
    use crate::neuron::gpu_lattices::{
        NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL, 
        NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME
    };


    type SpikeTrainType = RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;
    type SpikeTrainLatticeType = SpikeTrainLattice<IonotropicNeurotransmitterType, SpikeTrainType, SpikeTrainGridHistory>;
    type NeuronType = QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
    type LatticeType = Lattice<NeuronType, AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, STDP, IonotropicNeurotransmitterType>;
    type NetworkType = LatticeNetwork<
        NeuronType,
        AdjacencyMatrix<(usize, usize), f32>,
        GridVoltageHistory,
        SpikeTrainType,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition, f32>,
        STDP,
        IonotropicNeurotransmitterType,
    >;

    fn fill_buffer_f32(
        context: &Context,
        queue: &CommandQueue,
        size: usize,
        data: &[f32],
    ) -> Result<Buffer<cl_float>, GPUError> {
        let mut buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, data, &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };

        write_event.wait().map_err(|_| GPUError::WaitError)?;

        Ok(buffer)
    }

    fn fill_buffer_u32(
        context: &Context,
        queue: &CommandQueue,
        size: usize,
        data: &[u32],
    ) -> Result<Buffer<cl_uint>, GPUError> {
        let mut buffer = unsafe {
            Buffer::<cl_uint>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, data, &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };

        write_event.wait().map_err(|_| GPUError::WaitError)?;

        Ok(buffer)
    }

    #[test]
    fn test_chemical_kernel_with_spike_trains() -> Result<(), SpikingNeuralNetworksError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };

        let queue = match CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            0,
        ) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };  

        let chemical_inputs = match Program::create_and_build_from_source(&context, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::ProgramCompileFailure)),
        };

        let kernel = match Kernel::create(&chemical_inputs, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        };

        let n: u32 = 2;
        let num_types: u32 = 3;
        let connections: Vec<cl_uint> = vec![0, 0, 1, 0];
        let weights: Vec<cl_float> = vec![0.0, 0.0, 0.7, 0.0];
        let index_to_position: Vec<cl_uint> = vec![0, 1];
        let is_spike_train: Vec<cl_uint> = vec![0, 1];
        let flags: Vec<cl_uint> = vec![1, 1, 0];
        let spike_train_flags: Vec<cl_uint> = vec![1, 1, 0];
        let t: Vec<cl_float> = vec![0.5, 1.0, 0.0];
        let spike_train_t: Vec<cl_float> = vec![0.2, 0.4, 0.0];

        // let mut res = vec![0.0f32; num_types * n];
        // let mut counts = vec![0.0f32; num_types * n];

        let t_sums_buffer = create_and_write_buffer(&context, &queue, (num_types * n) as usize, 0.0)?;
        let counts_buffer = create_and_write_buffer(&context, &queue, (num_types * n) as usize, 0.0)?;

        let connections_buffer = fill_buffer_u32(&context, &queue, (n * n) as usize, &connections)?;
        let weights_buffer = fill_buffer_f32(&context, &queue, (n * n) as usize, &weights)?;     
        let index_to_position_buffer = fill_buffer_u32(&context, &queue, n as usize, &index_to_position)?;
        let is_spike_train_buffer = fill_buffer_u32(&context, &queue, n as usize, &is_spike_train)?;      
        let flags_buffer = fill_buffer_u32(&context, &queue, num_types as usize, &flags)?;      
        let spike_train_flags_buffer = fill_buffer_u32(&context, &queue, num_types as usize, &spike_train_flags)?;     
        let t_buffer = fill_buffer_f32(&context, &queue, num_types as usize, &t)?;      
        let spike_train_t_buffer = fill_buffer_f32(&context, &queue, num_types as usize, &spike_train_t)?; 

        let mut kernel_execution = ExecuteKernel::new(&kernel);

        unsafe {
            kernel_execution.set_arg(&connections_buffer);
            kernel_execution.set_arg(&weights_buffer);
            kernel_execution.set_arg(&index_to_position_buffer);
            kernel_execution.set_arg(&is_spike_train_buffer);
            kernel_execution.set_arg(&flags_buffer);
            kernel_execution.set_arg(&t_buffer);
            kernel_execution.set_arg(&spike_train_flags_buffer);
            kernel_execution.set_arg(&spike_train_t_buffer);
            kernel_execution.set_arg(&1);
            kernel_execution.set_arg(&n);
            kernel_execution.set_arg(&num_types);
            kernel_execution.set_arg(&counts_buffer);
            kernel_execution.set_arg(&t_sums_buffer);
        }

        let event = unsafe {
            match kernel_execution.set_global_work_size(n as usize)
                .enqueue_nd_range(&queue) {
                    Ok(value) => value,
                    Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::QueueFailure)),
                }
        };

        match event.wait() {
            Ok(_) => {},
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError)),
        };

        let mut counts_result = vec![0.0f32; (num_types * n) as usize];
        let mut res_result = vec![0.0f32; (num_types * n) as usize];

        let read_counts_event = unsafe {
            queue.enqueue_read_buffer(&counts_buffer, CL_NON_BLOCKING, 0, &mut counts_result, &[])
                .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::BufferReadError))?
        };
        read_counts_event.wait()
            .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError))?;

        let read_res_event = unsafe {
            queue.enqueue_read_buffer(&t_sums_buffer, CL_NON_BLOCKING, 0, &mut res_result, &[])
                .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::BufferReadError))?
        };
        read_res_event.wait()
            .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError))?;

        assert_eq!(counts_result, vec![1., 1., 0., 0., 0., 0.]);
        assert_eq!(res_result, vec![0.2 * 0.7, 0.4 * 0.7, 0., 0., 0., 0.]);

        Ok(())    
    }

    #[test]
    fn test_counts_chemical_kernel_with_spike_trains() -> Result<(), SpikingNeuralNetworksError> {
        let mut spike_train: SpikeTrainType = RateSpikeTrain { rate: 100., dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::default());
        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::GABA, ApproximateNeurotransmitter::default());

        let mut spike_train_lattice1: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice1.set_id(2);
        spike_train_lattice1.populate(&spike_train, 3, 3)?;
        spike_train_lattice1.apply(|neuron: &mut SpikeTrainType| {
            neuron.step = rand::thread_rng().gen_range(0.0..=100.);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
        });
        spike_train_lattice1.update_grid_history = true;

        let mut spike_train_lattice2: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice2.set_id(3);
        spike_train_lattice2.populate(&spike_train, 3, 3)?;
        spike_train_lattice2.apply(|neuron: &mut SpikeTrainType| {
            neuron.step = rand::thread_rng().gen_range(0.0..=100.);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
        });
        spike_train_lattice2.update_grid_history = true;

        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 5.,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::default());
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::GABA, ApproximateNeurotransmitter::default());
        base_neuron.set_dt(1.);

        let mut lattice1: LatticeType = Lattice::default();
        lattice1.set_id(0);
        lattice1.populate(&base_neuron, 3, 3)?;
        lattice1.apply(|neuron: &mut _| {
            neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
        });
        lattice1.connect(&(|x, y| x != y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 6.0 + rand::thread_rng().gen_range(0.0..=0.5))));
        lattice1.update_grid_history = true;

        let mut lattice2: LatticeType = Lattice::default();
        lattice2.set_id(1);
        lattice2.populate(&base_neuron, 2, 2)?;
        lattice2.apply(|neuron: &mut _| {
            neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
        });
        lattice2.connect(&(|x, y| x != y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 2.0 + rand::thread_rng().gen_range(0.0..=0.5))));
        lattice2.update_grid_history = true;

        let lattices = vec![lattice1, lattice2];
        let spike_trains = vec![spike_train_lattice1, spike_train_lattice2];
        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_trains)?;

        network.set_dt(1.);

        network.connect(0, 1, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 4.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(1, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 3.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(2, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 5.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(3, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 7.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;

        network.electrical_synapse = false;
        network.chemical_synapse = true;

        let gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };

        let queue = match CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            0,
        ) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };  

        let chemical_inputs = match Program::create_and_build_from_source(&context, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::ProgramCompileFailure)),
        };

        let kernel = match Kernel::create(&chemical_inputs, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        };

        let mut cell_vector: Vec<NeuronType> = vec![];

        let mut lattice_iterator: Vec<(usize, &Lattice<_, _, _, _, _>)> = gpu_network.lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
        for (_, value) in &lattice_iterator {
            let current_cell_grid = value.cell_grid();
            for row in current_cell_grid {
                for i in row {
                    cell_vector.push(i.clone());
                }
            }
        }
        
        let cell_vector = vec![cell_vector];

        let mut spike_train_cell_vector: Vec<SpikeTrainType> = vec![];

        let mut spike_train_lattice_iterator: Vec<(usize, &SpikeTrainLattice<_, _, _>)> = gpu_network.spike_train_lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        spike_train_lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));

        for (_, value) in &spike_train_lattice_iterator {
            let current_cell_grid = value.spike_train_grid();
            for row in current_cell_grid {
                for i in row {
                    spike_train_cell_vector.push(i.clone());
                }
            }
        }
    
        let spike_train_cell_vector = vec![spike_train_cell_vector];

        let gpu_cell_grid = NeuronType::convert_electrochemical_to_gpu(&cell_vector, &context, &queue)?;
        let gpu_spike_train_grid = SpikeTrainType::convert_electrochemical_to_gpu(&spike_train_cell_vector, &context, &queue)?;

        let gpu_graph = InterleavingGraphGPU::convert_to_gpu(
            &context, &queue, &gpu_network.lattices, &gpu_network.spike_train_lattices, gpu_network.get_connecting_graph()
        )?;

        let spike_train_skip_index = gpu_graph.lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>() as u32;

        let t_sums_buffer = create_and_write_buffer(
            &context, &queue, gpu_graph.size * IonotropicNeurotransmitterType::number_of_types(), 0.0
        )?;

        let counts_buffer = create_and_write_buffer(
            &context, &queue, gpu_graph.size * IonotropicNeurotransmitterType::number_of_types(), 0.0
        )?;

        let chemical_synapses_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(&kernel);

            kernel_execution.set_arg(&gpu_graph.connections)
                .set_arg(&gpu_graph.weights)
                .set_arg(&gpu_graph.index_to_position)
                .set_arg(&gpu_graph.is_spike_train);

            match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$flags"),
            };

            match &gpu_cell_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$t must be float"),
            };

            match &gpu_spike_train_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$flags"),
            };

            match &gpu_spike_train_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$t must be float"),
            };

            kernel_execution.set_arg(&spike_train_skip_index)
                .set_arg(&(gpu_graph.size as u32))
                .set_arg(&(IonotropicNeurotransmitterType::number_of_types() as u32));

            match kernel_execution
                .set_arg(&counts_buffer)
                .set_arg(&t_sums_buffer)
                .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                .enqueue_nd_range(&queue) {
                    Ok(value) => value,
                    Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::QueueFailure)),
                }
        };

        match chemical_synapses_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError)),
        };

        let mut res = vec![0.0f32; (IonotropicNeurotransmitterType::number_of_types() * gpu_graph.size) as usize];
        let read_event = unsafe {
            queue.enqueue_read_buffer(&t_sums_buffer, CL_NON_BLOCKING, 0, &mut res, &[])
                .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::BufferReadError))?
        };
        read_event.wait()
            .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError))?;

        let mut expected_res = vec![];

        for k in 0..2 {
            for n in 0..3 {
                for m in 0..3 {
                    if k == 1 && (n == 2 || m == 2) {
                        continue;
                    }

                    let postsynaptic_position = GraphPosition { id: k, pos: (n, m) };

                    let mut input_positions: HashSet<_> = network.get_lattice(&k).unwrap().graph().get_incoming_connections(&(n, m)).unwrap()
                        .iter()
                        .map(|i| GraphPosition { id: k, pos: *i })
                        .collect();
                    input_positions.extend(
                        network.get_connecting_graph().get_incoming_connections(&postsynaptic_position).unwrap()
                    );

                    let input_vals: Vec<NeurotransmitterConcentrations<_>> = input_positions
                        .iter()
                        .map(|input_position| {
                            let (pos_x, pos_y) = input_position.pos;

                            let mut neurotransmitter_input = if network.lattices.contains_key(&input_position.id) {
                                let input_cell = &network.lattices.get(&input_position.id)
                                    .unwrap()
                                    .cell_grid[pos_x][pos_y]; 

                                input_cell.get_neurotransmitter_concentrations()
                            } else {
                                let input_cell = &network.spike_train_lattices.get(&input_position.id)
                                    .unwrap()
                                    .cell_grid[pos_x][pos_y];

                                input_cell.get_neurotransmitter_concentrations()
                            };
                            
                            let weight: f32 = if input_position.id != postsynaptic_position.id {
                                network.get_connecting_graph().lookup_weight(input_position, &postsynaptic_position)
                                    .unwrap_or(Some(0.))
                                    .unwrap()
                            } else {
                                network.lattices.get(&input_position.id).unwrap()
                                    .graph()
                                    .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                                    .unwrap_or(Some(0.))
                                    .unwrap()
                            };

                            weight_neurotransmitter_concentration(&mut neurotransmitter_input, weight);

                            neurotransmitter_input
                        })
                        .collect();

                    let reference_neurotransmitter = aggregate_neurotransmitter_concentrations(&input_vals);

                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::AMPA).unwrap_or(&0.));
                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::NMDA).unwrap_or(&0.));
                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::GABA).unwrap_or(&0.));
                }
            }
        }

        for _ in 0..(res.len() - expected_res.len()) {
            expected_res.push(0.)
        }

        for (n, (i, j)) in res.iter().zip(expected_res.iter()).enumerate() {
            assert!((i - j).abs() < 0.001, "{} | {} != {}", n, i, j);
        }

        Ok(())
    }
    
    #[test]
    fn test_many_lattices_chemical_kernel() -> Result<(), SpikingNeuralNetworksError> {
        let mut spike_train: SpikeTrainType = RateSpikeTrain { rate: 100., dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::default());
        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::GABA, ApproximateNeurotransmitter::default());

        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 5.,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::default());
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::GABA, ApproximateNeurotransmitter::default());
        base_neuron.set_dt(1.);

        let (num_lattices, num_spike_train_lattices) = (20, 20);

        let mut lattices = vec![];

        for i in 0..num_lattices {
            let mut lattice: LatticeType = Lattice::default();
            lattice.set_id(i);
            lattice.populate(&base_neuron, 3, 3)?;
            lattice.apply(|neuron: &mut _| {
                neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            });
            lattice.connect(&(|x, y| x != y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 6.0 + rand::thread_rng().gen_range(0.0..=0.5))));
            lattice.update_grid_history = true;

            lattices.push(lattice);
        }

        let mut spike_train_lattices = vec![];

        for i in num_lattices..(num_lattices + num_spike_train_lattices) {
            let mut spike_train_lattice: SpikeTrainLatticeType = SpikeTrainLattice::default();
            spike_train_lattice.set_id(i);
            spike_train_lattice.populate(&spike_train, 3, 3)?;
            spike_train_lattice.apply(|neuron: &mut SpikeTrainType| {
                neuron.step = rand::thread_rng().gen_range(0.0..=100.);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::AMPA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::NMDA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
                neuron.synaptic_neurotransmitters.get_mut(&IonotropicNeurotransmitterType::GABA).unwrap().t = rand::thread_rng().gen_range(0.0..=1.0);
            });
            spike_train_lattice.update_grid_history = true;

            spike_train_lattices.push(spike_train_lattice);
        }
        
        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;

        network.set_dt(1.);

        network.connect(0, 1, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 4.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(1, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 3.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(2, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 5.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;
        network.connect(3, 0, &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), Some(&(|_, _| 7.0 + rand::thread_rng().gen_range(0.0..=0.5))))?;

        for i in num_lattices..(num_lattices - 1) {
            network.connect(
                i, 
                i + 1, 
                &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), 
                Some(&(|_, _| 1.0 + rand::thread_rng().gen_range(0.0..=0.5)))
            )?;

            network.connect(
                i + 1, 
                i, 
                &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), 
                Some(&(|_, _| 1.0 + rand::thread_rng().gen_range(0.0..=0.5)))
            )?;
        }

        for i in num_lattices..(num_lattices + num_spike_train_lattices) {
            network.connect(
                i, 
                i - num_lattices, 
                &(|x, y| x == y && rand::thread_rng().gen_range(0.0..=1.) < 0.5), 
                Some(&(|_, _| 1.0 + rand::thread_rng().gen_range(0.0..=0.5)))
            )?;
        }

        network.electrical_synapse = false;
        network.chemical_synapse = true;

        let gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };

        let queue = match CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            0,
        ) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };  

        let chemical_inputs = match Program::create_and_build_from_source(&context, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL, "") {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::ProgramCompileFailure)),
        };

        let kernel = match Kernel::create(&chemical_inputs, NETWORK_WITH_SPIKE_TRAIN_CHEMICAL_INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::KernelCompileFailure)),
        };

        let mut cell_vector: Vec<NeuronType> = vec![];

        let mut lattice_iterator: Vec<(usize, &Lattice<_, _, _, _, _>)> = gpu_network.lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
        for (_, value) in &lattice_iterator {
            let current_cell_grid = value.cell_grid();
            for row in current_cell_grid {
                for i in row {
                    cell_vector.push(i.clone());
                }
            }
        }
        
        let cell_vector = vec![cell_vector];

        let mut spike_train_cell_vector: Vec<SpikeTrainType> = vec![];

        let mut spike_train_lattice_iterator: Vec<(usize, &SpikeTrainLattice<_, _, _>)> = gpu_network.spike_train_lattices.iter()
            .map(|(i, j)| (*i, j))
            .collect();
        spike_train_lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));

        for (_, value) in &spike_train_lattice_iterator {
            let current_cell_grid = value.spike_train_grid();
            for row in current_cell_grid {
                for i in row {
                    spike_train_cell_vector.push(i.clone());
                }
            }
        }
    
        let spike_train_cell_vector = vec![spike_train_cell_vector];

        let gpu_cell_grid = NeuronType::convert_electrochemical_to_gpu(&cell_vector, &context, &queue)?;
        let gpu_spike_train_grid = SpikeTrainType::convert_electrochemical_to_gpu(&spike_train_cell_vector, &context, &queue)?;

        let gpu_graph = InterleavingGraphGPU::convert_to_gpu(
            &context, &queue, &gpu_network.lattices, &gpu_network.spike_train_lattices, gpu_network.get_connecting_graph()
        )?;

        let spike_train_skip_index = gpu_graph.lattice_sizes_map.values()
            .map(|(x, y)| *x * *y)
            .collect::<Vec<usize>>()
            .iter()
            .sum::<usize>() as u32;

        let t_sums_buffer = create_and_write_buffer(
            &context, &queue, gpu_graph.size * IonotropicNeurotransmitterType::number_of_types(), 0.0
        )?;

        let counts_buffer = create_and_write_buffer(
            &context, &queue, gpu_graph.size * IonotropicNeurotransmitterType::number_of_types(), 0.0
        )?;

        let chemical_synapses_event = unsafe {
            let mut kernel_execution = ExecuteKernel::new(&kernel);

            kernel_execution.set_arg(&gpu_graph.connections)
                .set_arg(&gpu_graph.weights)
                .set_arg(&gpu_graph.index_to_position)
                .set_arg(&gpu_graph.is_spike_train);

            match &gpu_cell_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$flags"),
            };

            match &gpu_cell_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$t must be float"),
            };

            match &gpu_spike_train_grid.get("neurotransmitters$flags").expect("Could not retrieve buffer: neurotransmitters$flags") {
                BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$flags"),
            };

            match &gpu_spike_train_grid.get("neurotransmitters$t").expect("Could not retrieve buffer: neurotransmitters$t") {
                BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                _ => unreachable!("neurotransmitters$t must be float"),
            };

            kernel_execution.set_arg(&spike_train_skip_index)
                .set_arg(&(gpu_graph.size as u32))
                .set_arg(&(IonotropicNeurotransmitterType::number_of_types() as u32));

            match kernel_execution
                .set_arg(&counts_buffer)
                .set_arg(&t_sums_buffer)
                .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                .enqueue_nd_range(&queue) {
                    Ok(value) => value,
                    Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::QueueFailure)),
                }
        };

        match chemical_synapses_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError)),
        };

        let mut res = vec![0.0f32; (IonotropicNeurotransmitterType::number_of_types() * gpu_graph.size) as usize];
        let read_event = unsafe {
            queue.enqueue_read_buffer(&t_sums_buffer, CL_NON_BLOCKING, 0, &mut res, &[])
                .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::BufferReadError))?
        };
        read_event.wait()
            .map_err(|_| SpikingNeuralNetworksError::GPURelatedError(GPUError::WaitError))?;

        let mut expected_res = vec![];

        for k in 0..num_lattices {
            for n in 0..3 {
                for m in 0..3 {
                    let postsynaptic_position = GraphPosition { id: k, pos: (n, m) };

                    let mut input_positions: HashSet<_> = network.get_lattice(&k).unwrap().graph().get_incoming_connections(&(n, m)).unwrap()
                        .iter()
                        .map(|i| GraphPosition { id: k, pos: *i })
                        .collect();
                    input_positions.extend(
                        network.get_connecting_graph().get_incoming_connections(&postsynaptic_position).unwrap()
                    );

                    let input_vals: Vec<NeurotransmitterConcentrations<_>> = input_positions
                        .iter()
                        .map(|input_position| {
                            let (pos_x, pos_y) = input_position.pos;

                            let mut neurotransmitter_input = if network.lattices.contains_key(&input_position.id) {
                                let input_cell = &network.lattices.get(&input_position.id)
                                    .unwrap()
                                    .cell_grid[pos_x][pos_y]; 

                                input_cell.get_neurotransmitter_concentrations()
                            } else {
                                let input_cell = &network.spike_train_lattices.get(&input_position.id)
                                    .unwrap()
                                    .cell_grid[pos_x][pos_y];

                                input_cell.get_neurotransmitter_concentrations()
                            };
                            
                            let weight: f32 = if input_position.id != postsynaptic_position.id {
                                network.get_connecting_graph().lookup_weight(input_position, &postsynaptic_position)
                                    .unwrap_or(Some(0.))
                                    .unwrap()
                            } else {
                                network.lattices.get(&input_position.id).unwrap()
                                    .graph()
                                    .lookup_weight(&input_position.pos, &postsynaptic_position.pos)
                                    .unwrap_or(Some(0.))
                                    .unwrap()
                            };

                            weight_neurotransmitter_concentration(&mut neurotransmitter_input, weight);

                            neurotransmitter_input
                        })
                        .collect();

                    let reference_neurotransmitter = aggregate_neurotransmitter_concentrations(&input_vals);

                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::AMPA).unwrap_or(&0.));
                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::NMDA).unwrap_or(&0.));
                    expected_res.push(*reference_neurotransmitter.get(&IonotropicNeurotransmitterType::GABA).unwrap_or(&0.));
                }
            }
        }

        for _ in 0..(res.len() - expected_res.len()) {
            expected_res.push(0.)
        }

        for (n, (i, j)) in res.iter().zip(expected_res.iter()).enumerate() {
            assert!((i - j).abs() < 0.001, "{} | {} != {}", n, i, j);
        }

        Ok(())
    }
}
