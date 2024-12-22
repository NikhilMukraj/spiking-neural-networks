use std::collections::HashMap;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, 
    context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
    kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_READ_WRITE}, 
    program::Program, types::{cl_float, CL_NON_BLOCKING},
};
use crate::{
    error::{GPUError, SpikingNeuralNetworksError}, 
    graph::{ConnectingGraphGPU, Graph, GraphGPU, GraphToGPU, ConnectingGraphToGPU, GraphPosition}
};
use super::{
    iterate_and_spike::{
        BufferGPU, IterateAndSpike, IterateAndSpikeGPU, 
        KernelFunction, NeurotransmitterTypeGPU,
    }, 
    spike_train::SpikeTrainGPU,
    plasticity::Plasticity,
    GridVoltageHistory, RunLattice, RunNetwork, SpikeTrainLatticeHistory,
    Lattice, LatticeHistory, Position, LatticeNetwork, SpikeTrainLattice, impl_apply
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
    int size
) {
    int gid = get_global_id(0);
    uint index = index_to_position[gid];

    history[iteration * size + index] = current_voltage[index]; 
}
"#;

const GRID_VOLTAGE_HISTORY_KERNEL_NAME: &str = "add_grid_voltage_history";
pub trait LatticeHistoryGPU: LatticeHistory {
    fn get_kernel(&self, context: &Context) -> Result<KernelFunction, GPUError>;
    fn to_gpu(&self, context: &Context, iterations: usize, size: (usize, usize)) -> Result<HashMap<String, BufferGPU>, GPUError>;
    fn add_from_gpu(
        &mut self, queue: &CommandQueue, buffers: HashMap<String, BufferGPU>, iterations: usize, size: (usize, usize)
    ) -> Result<(), GPUError>;  
}

impl LatticeHistoryGPU for GridVoltageHistory {
    fn get_kernel(&self, context: &Context) -> Result<KernelFunction, GPUError> {
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
            String::from("iteration"), String::from("size")
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

    if (is_spiking[index] == 1) {
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
    pub cell_grid: Vec<Vec<T>>,
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
                CL_QUEUE_SIZE,
            ) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::GetDeviceFailure),
            };

        let electrical_incoming_connections_program = match Program::create_and_build_from_source(&context, INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let electrical_incoming_connections_kernel = match Kernel::create(&electrical_incoming_connections_program, INPUTS_KERNEL_NAME) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let chemical_incoming_connections_program = match Program::create_and_build_from_source(&context, NEUROTRANSMITTER_INPUTS_KERNEL, ""){
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let chemical_incoming_connections_kernel = match Kernel::create(&chemical_incoming_connections_program, NEUROTRANSMITTER_INPUTS_KERNEL_NAME) {
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
            LatticeGPU { 
                cell_grid: lattice.cell_grid, 
                graph: lattice.graph, 
                electrical_incoming_connections_kernel,
                chemical_incoming_connections_kernel,
                last_firing_time_kernel,
                internal_clock: 0,
                grid_history_kernel: lattice.grid_history.get_kernel(&context)?,
                grid_history: lattice.grid_history,
                update_grid_history: lattice.update_grid_history,
                electrical_synapse: lattice.electrical_synapse,
                chemical_synapse: lattice.chemical_synapse,
                context,
                queue,
            }
        )
    }

    // Generates a GPU lattice from a given lattice
    pub fn from_lattice<
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, V, W, N>) -> Result<Self, GPUError> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        LatticeGPU::from_lattice_given_device(lattice, &device)
    }

    /// Sets timestep variable for the lattice
    pub fn set_dt(&mut self, dt: f32) {
        self.apply(|neuron| neuron.set_dt(dt));
        // self.plasticity.set_dt(dt);
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
                    kernel_execution.set_arg(&self.internal_clock);
                } else if i == "size" {
                    kernel_execution.set_arg(&gpu_graph.size);
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

                match kernel_execution.set_arg(&gpu_graph.size)
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

                    match kernel_execution.set_arg(&gpu_graph.size)
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

                kernel_execution.set_arg(&gpu_graph.size)
                    .set_arg(&N::number_of_types());

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
                        kernel_execution.set_arg(&N::number_of_types());
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

impl<T, U, V, N> RunLattice for LatticeGPU<T, U, V, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
    V: LatticeHistory + LatticeHistoryGPU,
    N: NeurotransmitterTypeGPU,
{
    fn run_lattice(&mut self, iterations: usize) -> Result<(), SpikingNeuralNetworksError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, false) => self.run_lattice_electrical_synapses(iterations).map_err(Into::into),
            (false, true) => self.run_lattice_chemical_synapses(iterations).map_err(Into::into),
            (true, true) => self.run_lattice_chemical_synapses(iterations).map_err(Into::into),
            (false, false) => Ok(()),
        }
    }
}

/// An implementation of a lattice network that is compatible with the GPU
#[allow(dead_code)]
pub struct LatticeNetworkGPU<
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
    C: Graph<K=GraphPosition, V=f32> + ConnectingGraphToGPU<ConnectingGraphGPU>, 
> {
    lattices: HashMap<usize, Lattice<T, U, V, Y, N>>,
    spike_train_lattices: HashMap<usize, SpikeTrainLattice<N, W, X>>,
    connecting_graph: C,
    electrical_incoming_connections_kernel: Kernel,
    chemical_incoming_connections_kernel: Kernel,
    last_firing_time_kernel: Kernel,
    context: Context,
    queue: CommandQueue,
    grid_history_kernels: HashMap<usize, KernelFunction>,
    pub electrical_synapse: bool,
    pub chemical_synapse: bool,
    internal_clock: usize,
}

impl<T, U, V, W, X, Y, N, C> LatticeNetworkGPU<T, U, V, W, X, Y, N, C>
where
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
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

        let queue =  match CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE,
        ) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::GetDeviceFailure),
        };

        // let electrical_incoming_connections_program = match Program::create_and_build_from_source(&context, INPUTS_KERNEL, ""){
        //     Ok(value) => value,
        //     Err(_) => return Err(GPUError::ProgramCompileFailure),
        // };
        // let electrical_incoming_connections_kernel = match Kernel::create(&electrical_incoming_connections_program, INPUTS_KERNEL_NAME) {
        //     Ok(value) => value,
        //     Err(_) => return Err(GPUError::KernelCompileFailure),
        // };

        // let chemical_incoming_connections_program = match Program::create_and_build_from_source(&context, NEUROTRANSMITTER_INPUTS_KERNEL, ""){
        //     Ok(value) => value,
        //     Err(_) => return Err(GPUError::ProgramCompileFailure),
        // };
        // let chemical_incoming_connections_kernel = match Kernel::create(&chemical_incoming_connections_program, NEUROTRANSMITTER_INPUTS_KERNEL_NAME) {
        //     Ok(value) => value,
        //     Err(_) => return Err(GPUError::KernelCompileFailure),
        // };

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
                context, 
                queue, 
                last_firing_time_kernel, 
                electrical_incoming_connections_kernel: todo!(), 
                #[allow(unreachable_code)]
                chemical_incoming_connections_kernel: todo!(), 
                #[allow(unreachable_code)]
                grid_history_kernels: todo!(),
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
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        Self::from_network_given_device(lattice_network, &device)
    }

    fn run_lattices_with_electrical_synapses(&mut self, _iterations: usize) -> Result<(), GPUError> {
        // concat cell grids into a single 1d vector of cell grids
        // create a new vector that keeps track of which lattices and positions each cell belongs to
        // use those vectors to write the new values back to the cpu and execute the connections kernel
        // connections kernel should take each weight matrix in and use lattice size to index where to go

        let mut cell_vector: Vec<T> = vec![];
        let mut lattice_ids: Vec<u32> = vec![];
        let mut lattice_sizes: Vec<u32> = vec![];
        let mut lattice_sizes_map: HashMap<usize, (usize, usize)> = HashMap::new();

        for (key, value) in &self.lattices {
            let current_cell_grid = value.get_cell_grid();
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

        let mut cell_vector = vec![cell_vector];
        let cell_vector_size = cell_vector.first().unwrap_or(&vec![]).len();

        let gpu_cell_grid = T::convert_to_gpu(&cell_vector, &self.context, &self.queue)?;

        T::convert_electrochemical_to_cpu(
            &mut cell_vector, 
            &gpu_cell_grid, 
            1, 
            cell_vector_size, 
            &self.queue
        )?;

        let mut new_grids: HashMap<usize, Vec<T>> = HashMap::new();

        if let Some(first_vec) = cell_vector.first() {
            for (id, cell) in lattice_ids.iter().zip(first_vec.iter()) {
                new_grids.entry(*id as usize)
                    .and_modify(|vec| vec.push(cell.clone()))
                    .or_insert_with(|| vec![cell.clone()]);
            }
        } // maybe use std::mem::take ?

        let mut reshaped_grids = HashMap::new();

        for (key, vec) in new_grids {
            if let Some(&(_, cols)) = lattice_sizes_map.get(&key) {
                let reshaped_vec: Vec<Vec<T>> = vec.chunks(cols)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                reshaped_grids.insert(key, reshaped_vec);
            }
        }

        for (key, value) in self.lattices.iter_mut() {
            value.set_cell_grid(reshaped_grids.get(key).unwrap().clone()).expect("Same dimensions");
        }

        Ok(())
    }
}

impl<T, U, V, W, X, Y, N, C> RunNetwork for LatticeNetworkGPU<T, U, V, W, X, Y, N, C>
where
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
    V: LatticeHistory + LatticeHistoryGPU,
    W: SpikeTrainGPU<N=N>,
    X: SpikeTrainLatticeHistory,
    Y: Plasticity<W, T, f32> + Plasticity<T, T, f32>,
    N: NeurotransmitterTypeGPU,
    C: Graph<K=GraphPosition, V=f32> + ConnectingGraphToGPU<ConnectingGraphGPU>, 
{
    fn run_lattices(&mut self, iterations: usize) -> Result<(), SpikingNeuralNetworksError> {
        match (self.electrical_synapse, self.chemical_synapse) {
            (true, false) => self.run_lattices_with_electrical_synapses(iterations).map_err(Into::into),
            (false, true) => todo!(),
            (true, true) => todo!(),
            (false, false) => Ok(()),
        }
    }
}

// track history over time 
// last firing time kernel
// int buffer
// should probably make method or macro to extract a buffer
