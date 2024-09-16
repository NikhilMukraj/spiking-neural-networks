use std::ptr;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, 
    context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
    kernel::{ExecuteKernel, Kernel}, memory::{Buffer, CL_MEM_WRITE_ONLY}, 
    program::Program, types::{cl_float, CL_NON_BLOCKING},
};
use crate::graph::{Graph, GraphToGPU};
use super::iterate_and_spike::{IterateAndSpike, IterateAndSpikeGPU, NeurotransmitterType};
use super::plasticity::Plasticity;
use super::{Lattice, LatticeHistory};


// pub trait GraphGPU: Default {

// }

// eventually will need neurotransmitter gpu type
// eventually add lattice history

// convert graph to gpu
// may need to use gpu graph outside of trait first

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
    uint count = 0;
    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] == 1) {
            int presynaptic_index = index_to_position[i];
            int postsynaptic_index = index_to_position[gid];
            float gap_junction = gap_conductances[postsynaptic_index] * (voltages[presynaptic_index] - voltages[postsynaptic_index]);
            sum += weights[i * n + gid] * gap_junction;
            count++;
        }
    }
    
    if (count != 0) {
        res[gid] = sum / count;
    } else {
        res[gid] = 0;
    }
}
"#;

const INPUTS_KERNEL_NAME: &str = "calculate_internal_electrical_inputs";

pub struct LatticeGPU<
    T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
    U: Graph<K=(usize, usize), V=f32> + GraphToGPU, 
    N: NeurotransmitterType,
> {
    pub cell_grid: Vec<Vec<T>>,
    graph: U,
    incoming_connections_kernel: Kernel,
    context: Context,
    queue: CommandQueue,
}

impl<T, U, N> LatticeGPU<T, U, N>
where
    T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
    U: Graph<K = (usize, usize), V = f32> + GraphToGPU,
    N: NeurotransmitterType,
{
    pub fn from_lattice<
        LatticeHistoryCPU: LatticeHistory, 
        W: Plasticity<T, T, f32>,
    >(lattice: Lattice<T, U, LatticeHistoryCPU, W, N>) -> Self {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

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

        LatticeGPU { 
            cell_grid: lattice.cell_grid, 
            graph: lattice.graph, 
            incoming_connections_kernel,
            context,
            queue,
        }
    }

    // modify to be falliable
    // modify to account for last firing time
    pub fn run_lattice(&mut self, iterations: usize) {
        let gpu_cell_grid = T::convert_to_gpu(&self.cell_grid, &self.context);

        let gpu_graph = self.graph.convert_to_gpu(&self.context, &self.queue);

        let iterate_kernel = T::iterate_and_spike_electrical_kernel(&self.context);

        let mut sums_buffer = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, gpu_graph.size, ptr::null_mut())
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
                ExecuteKernel::new(&self.incoming_connections_kernel)
                    .set_arg(&gpu_graph.connections)
                    .set_arg(&gpu_graph.weights)
                    .set_arg(&gpu_graph.index_to_position)
                    .set_arg(&gpu_cell_grid.get("gap_conductance"))
                    .set_arg(&gpu_cell_grid.get("current_voltage"))
                    .set_arg(&gpu_graph.size)
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
                    kernel_execution.set_arg(&gpu_cell_grid.get(i));
                }

                kernel_execution.set_global_work_size(gpu_graph.size)
                    .enqueue_nd_range(&self.queue)
                    .expect("Could not queue kernel")
            };

            iterate_event.wait().expect("Could not wait");
        }
    }
}
