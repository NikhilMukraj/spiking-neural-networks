use std::collections::HashMap;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, 
    context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
    kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_READ_WRITE}, 
    program::Program, types::{cl_float, CL_NON_BLOCKING},
};
use crate::graph::{Graph, GraphToGPU};
use super::{iterate_and_spike::{BufferGPU, IterateAndSpike, IterateAndSpikeGPU, KernelFunction, NeurotransmitterType}, GridVoltageHistory};
use super::plasticity::Plasticity;
use super::{Lattice, LatticeHistory, Position, impl_apply};
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

const GRID_VOLTAGE_HISTORY_KERNEL: &str = r#"
__kernel void add_grid_voltage_history(
    __global const uint *index_to_position,
    __global const float *current_voltage,
    __global float *history,
    __global const int iteration,
    __global const int size
) {
    int gid = get_global_id(0);
    int index = index_to_position[i];

    history[iteration * size + index] = current_voltage[index]; 
}
"#;

const GRID_VOLTAGE_HISTORY_KERNEL_NAME: &str = "add_grid_voltage_history";
pub trait LatticeHistoryGPU {
    fn get_kernel(&self, context: &Context) -> KernelFunction;
    fn to_gpu(&self, context: &Context, iterations: usize, size: (usize, usize)) -> HashMap<String, BufferGPU>;
    fn add_from_gpu(&mut self, queue: &CommandQueue, buffers: HashMap<String, BufferGPU>, iterations: usize, size: (usize, usize));  
}

impl LatticeHistoryGPU for GridVoltageHistory {
    fn get_kernel(&self, context: &Context) -> KernelFunction {
        let history_program = Program::create_and_build_from_source(context, GRID_VOLTAGE_HISTORY_KERNEL, "")
            .expect("Program::create_and_build_from_source failed");
        let history_kernel = Kernel::create(&history_program, GRID_VOLTAGE_HISTORY_KERNEL_NAME)
            .expect("Kernel::create failed");

        let argument_names = vec![
            String::from("index_to_position"), String::from("current_voltage"), String::from("history"),
            String::from("iteration"), String::from("size")
        ];

        KernelFunction { 
            kernel: history_kernel, 
            program_source: String::from(GRID_VOLTAGE_HISTORY_KERNEL), 
            kernel_name: String::from(GRID_VOLTAGE_HISTORY_KERNEL_NAME), 
            argument_names
        }
    }

    fn to_gpu(&self, context: &Context, iterations: usize, size: (usize, usize)) -> HashMap<String, BufferGPU> {
        let history_buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, iterations * size.0 * size.1, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let mut buffers = HashMap::new();

        buffers.insert(String::from("history"), BufferGPU::Float(history_buffer));

        buffers
    }

    fn add_from_gpu(&mut self, queue: &CommandQueue, buffers: HashMap<String, BufferGPU>, iterations: usize, size: (usize, usize)) {
        let mut results = vec![0.0; iterations * size.0 * size.1];
        let results_read_event = unsafe {
            queue.enqueue_read_buffer(
                match buffers.get("history").expect("Could not get history") {
                    BufferGPU::Float(value) => value,
                    BufferGPU::UInt(_) => unreachable!("History is not unsigned integer")
                }, 
                CL_NON_BLOCKING, 
                0, 
                &mut results, 
                &[]
            ).expect("Could not read buffer")
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

        self.history.extend(nested_vector)
    }
}

// const LAST_FIRING_TIME_KERNEL: &str = r#"
// __kernel__ void set_last_firing_time(
//     __global const uint *index_to_position,
//     __global const uint *is_spiking,
//     __global const int iteration,
//     __global int last_firing_time
// ) {
//     int gid = get_global_id(0);
//     int index = index_to_position[gid];

//     if (is_spiking[index] == 1) {
//         last_firing_time[index] = iteration;
//     }
// } 
// "#;

// const LAST_FIRING_TIME_KERNEL_NAME: &str = "set_last_firing_time";

pub struct LatticeGPU<
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU, 
    N: NeurotransmitterType,
> {
    pub cell_grid: Vec<Vec<T>>,
    graph: U,
    incoming_connections_kernel: Kernel,
    // last_firing_time_kernel: Kernel,
    context: Context,
    queue: CommandQueue,
}

impl<T, U, N> LatticeGPU<T, U, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU,
    N: NeurotransmitterType,
{
    impl_apply!();

    // Generates a GPU lattice given a lattice and a device
    pub fn from_lattice_given_device< 
        LatticeHistoryCPU: LatticeHistory, 
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, LatticeHistoryCPU, W, N>, device: &Device) -> Self {
        let context = Context::from_device(device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let incoming_connections_program = Program::create_and_build_from_source(&context, INPUTS_KERNEL, "")
            .expect("Program::create_and_build_from_source failed");
        let incoming_connections_kernel = Kernel::create(&incoming_connections_program, INPUTS_KERNEL_NAME)
            .expect("Kernel::create failed");

        // let last_firing_time_program = Program::create_and_build_from_source(&context, LAST_FIRING_TIME_KERNEL, "")
        //     .expect("Program::create_and_build_from_source failed");
        // let last_firing_time_kernel = Kernel::create(&last_firing_time_program, LAST_FIRING_TIME_KERNEL_NAME)
        //     .expect("Kernel::create failed");

        LatticeGPU { 
            cell_grid: lattice.cell_grid, 
            graph: lattice.graph, 
            incoming_connections_kernel,
            // last_firing_time_kernel,
            context,
            queue,
        }
    }

    // Generates a GPU lattice from a given lattice
    pub fn from_lattice<
        LatticeHistoryCPU: LatticeHistory, 
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, LatticeHistoryCPU, W, N>) -> Self {
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

    // modify to be falliable
    // modify to account for last firing time (reset firing time macro)
    pub fn run_lattice(&mut self, iterations: usize) {
        let gpu_cell_grid = T::convert_to_gpu(&self.cell_grid, &self.context, &self.queue);

        let gpu_graph = self.graph.convert_to_gpu(&self.context, &self.queue, &self.cell_grid);

        let iterate_kernel = T::iterate_and_spike_electrical_kernel(&self.context);

        let mut sums_buffer = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, gpu_graph.size, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let sums_write_event = unsafe { 
            self.queue.enqueue_write_buffer(
                &mut sums_buffer, 
                CL_NON_BLOCKING, 
                0, 
                &(0..gpu_graph.size).map(|_| 0.).collect::<Vec<f32>>(), 
                &[]
            ).expect("Could not write to sums")
        };
    
        sums_write_event.wait().expect("Could not wait");

        for _n in 0..iterations {
            let gap_junctions_event = unsafe {
                let mut kernel_execution = ExecuteKernel::new(&self.incoming_connections_kernel);

                kernel_execution.set_arg(&gpu_graph.connections)
                    .set_arg(&gpu_graph.weights)
                    .set_arg(&gpu_graph.index_to_position);

                match &gpu_cell_grid.get("gap_conductance").expect("Could not retrieve buffer") {
                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                    BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                };

                match &gpu_cell_grid.get("current_voltage").expect("Could not retrieve buffer") {
                    BufferGPU::Float(buffer) => kernel_execution.set_arg(buffer),
                    BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                };

                kernel_execution.set_arg(&gpu_graph.size)
                    .set_arg(&sums_buffer)
                    .set_global_work_size(gpu_graph.size) // number of threads executing in parallel
                    // .set_wait_event(&sums_write_event)
                    .enqueue_nd_range(&self.queue)
                    .expect("Could not queue kernel")
            };

            gap_junctions_event.wait().expect("Could not wait");

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
                            BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
                        };
                    }
                }

                kernel_execution.set_global_work_size(gpu_graph.size)
                    .enqueue_nd_range(&self.queue)
                    .expect("Could not queue kernel")
            };

            iterate_event.wait().expect("Could not wait");

            // let last_firing_time_event = unsafe {
            //     let mut kernel_execution = ExecuteKernel::new(&self.last_firing_time_kernel)
            //         .set_arg(&gpu_graph.index_to_position)
            //         .set_arg(
            //             match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
            //                 BufferGPU::Float(buffer) => unreachable!("Is spiking cannot be float"),
            //                 BufferGPU::UInt(buffer) => kernel_execution.set_arg(buffer),
            //             }
            //         )
            //         .set_arg(iteration)
            //         .set_arg(
            //             match &gpu_cell_grid.get(i).unwrap_or_else(|| panic!("Could not retrieve buffer: {}", i)) {
            //                 _ => unreachable!("Last firing cannot be float or unsigned integer"),
            //                 BufferGPU::Int(buffer) => kernel_execution.set_arg(buffer),
            //             }
            //         )
            //         .set_global_work_size(gpu_graph.size)
            //         .enqueue_nd_range(&self.queue)
            //         .expect("Could not queue kernel")
            // };

            // last_firing_time_event.wait().expect("Could not wait")

            // if history add history
        }

        let rows = self.cell_grid.len();
        let cols = self.cell_grid.first().unwrap_or(&vec![]).len();

        T::convert_to_cpu(
            &mut self.cell_grid, 
            &gpu_cell_grid, 
            rows, 
            cols, 
            &self.queue
        );
    }
}

// track history over time 
// last firing time kernel
// int buffer
// should probably make method or macro to extract a buffer
