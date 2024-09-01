use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY}; // CL_MEM_WRITE_ONLY
use opencl3::program::Program;
use opencl3::types::{cl_bool, cl_float, cl_uint, cl_event, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;
use rand::Rng;


const PROGRAM_SOURCE: &str = r#"
__kernel void incoming_connections_sum(
    __global const bool *connections, 
    __global const float *weights, 
    uint n, 
    __global float *res
) {
    int gid = get_global_id(0);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (connections[i * n + gid] != 0) {
            sum += weights[i * n + gid];
        }
    }
    
    res[gid] = sum;
}
"#;

const KERNEL_NAME: &str = "incoming_connections_sum";

fn create_random_flattened_adj_matrix(size: usize, lower_bound: f32, upper_bound: f32) -> (Vec<bool>, Vec<f32>) {
    let full_size = size * size;
    
    let mut connections = vec![false; full_size];
    let mut weights = vec![0.; full_size];

    let mut rng = rand::thread_rng();

    for i in 0..(full_size) {
        connections[i] = rng.gen_bool(0.5);
        if connections[i] {
            weights[i] = rng.gen_range(lower_bound..=upper_bound);
        }
    }

    (connections, weights)
}

fn main() -> Result<()> {
    // create two matrices
    // boolean matrix (is connecting or not)
    // weight matrix (value of weight if connecting)
    // calculate gap junction (insert gap junction kernel at runtime if possible)

    // array where index corresponds to a position in cell grid
    // index in graph can that be associated with cell grid positions
    // assume non-ragged for now

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
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

    // build kernel
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // number of nodes
    const N: usize = 100;

    let (connections, weights) = create_random_flattened_adj_matrix(N, 0., 2.);
    let mut connections_array: [cl_bool; N * N] = [0; N * N];
    let mut weights_array: [cl_float; N * N] = [0.; N * N];
    let sums_array: [cl_float; N] = [0.; N];

    for i in 0..(N * N) {
        connections_array[i] = connections[i] as u32;
        weights_array[i] = weights[i];
    }

    let mut connections_buffer = unsafe {
        Buffer::<cl_bool>::create(&context, CL_MEM_READ_ONLY, N * N, ptr::null_mut())?
    };
    let mut weights_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, N * N, ptr::null_mut())?
    };
    let mut sums_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, N, ptr::null_mut())?
    };

    let _connections_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut connections_buffer, CL_BLOCKING, 0, &connections_array, &[])? 
    };
    let _weights_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut weights_buffer, CL_BLOCKING, 0, &weights_array, &[])? 
    };
    let sums_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut sums_buffer, CL_NON_BLOCKING, 0, &sums_array, &[])? 
    };

    let n_cl: cl_uint = N as u32;

    // execute kernel
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&connections_buffer)
            .set_arg(&weights_buffer)
            .set_arg(&n_cl)
            .set_arg(&sums_buffer)
            .set_global_work_size(N) // number of threads executing in parallel
            .set_wait_event(&sums_write_event)
            .enqueue_nd_range(&queue)?
    };

    let events: Vec<cl_event> = vec![kernel_event.get()];

    let mut results: [cl_float; N] = [0.0; N];
    let results_read_event = unsafe {
        queue.enqueue_read_buffer(&sums_buffer, CL_NON_BLOCKING, 0, &mut results, &events)?
    };

    results_read_event.wait()?;
    
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;

    println!("Graph execution time (ns): {}", duration);

    Ok(())
}
