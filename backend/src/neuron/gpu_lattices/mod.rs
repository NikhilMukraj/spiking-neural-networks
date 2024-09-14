// use opencl3::{memory::Buffer, types::{cl_float, cl_uint}};
// use crate::graph::{Graph, GraphToGPU, GraphGPU};
// use super::iterate_and_spike::{IterateAndSpike, IterateAndSpikeGPU, NeurotransmitterType};
// use super::plasticity::Plasticity;
// use super::{Lattice, LatticeHistory};


// // pub trait GraphGPU: Default {

// // }

// // eventually will need neurotransmitter gpu type
// // eventually add lattice history

// // convert graph to gpu
// // may need to use gpu graph outside of trait first

// pub struct LatticeGPU<
//     T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
//     U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
//     N: NeurotransmitterType,
// > {
//     cell_grid: Vec<Vec<T>>,
//     graph: U,
//     graph_gpu: Option<GraphGPU>,
// }

// impl<T, U, N> LatticeGPU<T, U, N>
// where
//     T: IterateAndSpike<N = N> + IterateAndSpikeGPU,
//     U: Graph<K = (usize, usize), V = f32> + GraphToGPU<GraphGPU>,
//     N: NeurotransmitterType,
// {
//     pub fn from_lattice<
//         LatticeHistoryCPU: LatticeHistory, 
//         W: Plasticity<T, T, f32>,
//     >(lattice: Lattice<T, U, LatticeHistoryCPU, W, N>) -> Self {
//         LatticeGPU { 
//             cell_grid: lattice.cell_grid, 
//             graph: lattice.graph, 
//             graph_gpu: None,
//         }
//     }

    // // modify to be falliable
    // // modify to account for last firing time
    // fn run_lattices(iterations: usize) {
    //     let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
    //         .first()
    //         .expect("No GPU found");
    //     let device = Device::new(device_id);

    //     let context = Context::from_device(&device).expect("Context::from_device failed");

    //     let queue = CommandQueue::create_default_with_properties(
    //             &context, 
    //             CL_QUEUE_PROFILING_ENABLE,
    //             CL_QUEUE_SIZE,
    //         )
    //         .expect("CommandQueue::create_default failed");

    //     let gpu_cell_grid = T::convert(&self.cell_grid, &context, &queue);

    //     self.gpu_graph = self.graph.convert_to_gpu();

    //     let iterate_kernel = T::iterate_and_spike_electrical_kernel(&context);

    //     for n in 0..iterations {
    //         let gap_junctions_event = unsafe {
    //             ExecuteKernel::new(&incoming_connections_kernel)
    //                 .set_arg(&connections_buffer)
    //                 .set_arg(&weights_buffer)
    //                 .set_arg(&index_to_position_buffer)
    //                 .set_arg(&gap_conductances_buffer)
    //                 .set_arg(&voltages_buffer)
    //                 .set_arg(&n_cl)
    //                 .set_arg(&sums_buffer)
    //                 .set_global_work_size(gpu_graph.length) // number of threads executing in parallel
    //                 // .set_wait_event(&sums_write_event)
    //                 .enqueue_nd_range(&queue)
    //                 .expect("Could not queue kernel")
    //         };

    //         gap_junctions_event.wait().expect("Could not wait");

    //         let iterate_event = unsafe {
    //             let kernel_execution = ExecuteKernel::new(&iterate_kernel.kernel);

    //             for i in iterate_kernel.argument_names {
    //                 kernel_execution.set_arg(gpu_cell_grid.get(i));
    //             }

    //             kernel_execution.set_global_work_size(gpu_graph.length)
    //                 .enqueue_nd_range(&queue)
    //                 .expect("Could not queue kernel");
    //         };

    //         iterate_event.wait()?;
    //     }
    // }
// }
