use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel; // ExecuteKernel
// use opencl3::memory::{Buffer, CL_MEM_READ_ONLY}; // CL_MEM_WRITE_ONLY
use opencl3::program::Program;
// use opencl3::types::{cl_float, cl_uint, cl_event, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
// use std::ptr;
use rand::Rng;


const PROGRAM_SOURCE: &str = r#"
__kernel void incoming_connections_sum(
    __global const uint *connections, 
    __global_const float *weights, 
    uint n, 
    __global float *res,
) {
    int gid = get_global_id(0);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (adjacency_matrix[i * n + gid] != 0) {
            sum += weights[i * n + gid];
        }
    }
    
    res[gid] = sum;
}
"#;

const KERNEL_NAME: &str = "incoming_connections_sum";

fn create_random_flattened_adj_matrix(size: u32, lower_bound: f32, upper_bound: f32) -> (Vec<u32>, Vec<f32>) {
    let mut connections = Vec::new();
    let mut weights = Vec::new();

    let mut rng = rand::thread_rng();

    for i in 0..(size * size) {
        connections[i as usize] = rng.gen_range(0..=1);
        if connections[i as usize] != 0 {
            weights[i as usize] = rng.gen_range(lower_bound..=upper_bound);
        } else {
            weights[i as usize] = 0.;
        }
    }

    (connections, weights)
}

fn main() -> Result<()> {
    // create two matrices
    // boolean matrix (is connecting or not)
    // weight matrix (value of weight if connecting)
    // calculate gap junction (insert gap junction kernel at runtime if possible)

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("No GPU found");
    let device = Device::new(device_id);

    let context = Context::from_device(&device).expect("Context::from_device failed");

    let _queue = CommandQueue::create_default_with_properties(
            &context, 
            CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE,
        )
        .expect("CommandQueue::create_default failed");

    // build kernel
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let _kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // number of nodes
    const N: u32 = 10;

    let (_connections, _weights) = create_random_flattened_adj_matrix(N, 0., 2.);

    // execute kernel
    // let kernel_event = unsafe {
    //     ExecuteKernel::new(&kernel)
    //         .set_arg(&matrix_buffer)
    //         .set_arg(&sums_buffer)
    //         .set_arg(&m_cl)
    //         .set_arg(&n_cl)
    //         .set_global_work_size(FULL_SIZE)
    //         .set_wait_event(&sums_write_event)
    //         .enqueue_nd_range(&queue)?
    // };

    Ok(())
}
