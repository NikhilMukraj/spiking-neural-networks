use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{Kernel, ExecuteKernel}; 
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY}; // CL_MEM_WRITE_ONLY
use opencl3::program::Program;
use opencl3::types::{cl_float, cl_uint, cl_event, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;
use std::time::Instant;


const PROGRAM_SOURCE: &str = r#"
__kernel void colwise_sum(
    __global const float *mat, __global float *res, uint m, uint n
) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += mat[i * m + gid];
    }
    res[gid] = sum;
}"#;

const KERNEL_NAME: &str = "colwise_sum";

fn colwise_sum(matrix: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.; m];

    for i in 0..m {
        for j in 0..n {
            output[i] += matrix[j * m + i];
        }
    }

    output
}

fn main()  -> Result<()> {
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

    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    const N: usize = 12;
    const M: usize = 64;
    const FULL_SIZE: usize = M * N;

    let mut matrix: [cl_float; FULL_SIZE] = [1.0; FULL_SIZE];
    for (n, i) in matrix.iter_mut().enumerate().take(FULL_SIZE) {
        *i = n as cl_float;
    }

    let sums: [cl_float; M] = [0.0; M];

    let gpu_load_and_execute_start = Instant::now();

    let mut matrix_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, FULL_SIZE, ptr::null_mut())?
    };
    let mut sums_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, M, ptr::null_mut())?
    };

    let _matrix_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut matrix_buffer, CL_BLOCKING, 0, &matrix, &[])? 
    };
    let sums_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut sums_buffer, CL_NON_BLOCKING, 0, &sums, &[])? 
    };

    let m_cl: cl_uint = M as u32;
    let n_cl: cl_uint = N as u32;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&matrix_buffer)
            .set_arg(&sums_buffer)
            .set_arg(&m_cl)
            .set_arg(&n_cl)
            .set_global_work_size(FULL_SIZE)
            .set_wait_event(&sums_write_event)
            .enqueue_nd_range(&queue)?
    };

    let events: Vec<cl_event> = vec![kernel_event.get()];

    let mut results: [cl_float; M] = [0.0; M];
    let results_read_event = unsafe {
        queue.enqueue_read_buffer(&sums_buffer, CL_NON_BLOCKING, 0, &mut results, &events)?
    };

    results_read_event.wait()?;

    let gpu_load_duration = gpu_load_and_execute_start.elapsed();

    println!("results front: {}", results[0]);
    println!("results back: {}", results[M - 1]);

    let start = Instant::now();
    let cpu_results = colwise_sum(&matrix, M, N);
    let cpu_duration = start.elapsed();

    println!("expected: {}", cpu_results[0]);
    println!("expected: {}", cpu_results[M - 1]);

    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);
    println!("cpu execution duration (ns): {}", cpu_duration.as_nanos());

    println!("gpu load and execution (ns): {}", gpu_load_duration.as_nanos());
    
    Ok(())
}
