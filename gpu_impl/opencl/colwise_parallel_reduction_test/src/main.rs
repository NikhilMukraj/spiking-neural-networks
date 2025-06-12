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
__kernel void colwise_sum_parallel(__global const float *mat, __global float *res, uint m, uint n) {
    int col = get_group_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    __local float scratch[256];

    float sum = 0.0f;

    for (int i = lid; i < n; i += lsize) {
        sum += mat[i * m + col];
    }

    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        res[col] = scratch[0];
    }
}"#;

const KERNEL_NAME: &str = "colwise_sum_parallel";

const SERIAL_PROGRAM_SOURCE: &str = r#"
__kernel void colwise_sum_serial(
    __global const float *mat, __global float *res, uint m, uint n
) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += mat[i * m + gid];
    }
    res[gid] = sum;
}"#;

const SERIAL_KERNEL_NAME: &str = "colwise_sum_serial";

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

    let serial_program = Program::create_and_build_from_source(&context, SERIAL_PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let serial_kernel = Kernel::create(&serial_program, SERIAL_KERNEL_NAME).expect("Kernel::create failed");

    const N: usize = 256;
    const M: usize = 512;
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
            .set_local_work_size(64)
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

    let serial_gpu_load_and_execute_start = Instant::now();

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

    let serial_kernel_event = unsafe {
        ExecuteKernel::new(&serial_kernel)
            .set_arg(&matrix_buffer)
            .set_arg(&sums_buffer)
            .set_arg(&m_cl)
            .set_arg(&n_cl)
            .set_global_work_size(FULL_SIZE)
            .set_local_work_size(64)
            .set_wait_event(&sums_write_event)
            .enqueue_nd_range(&queue)?
    };

    let events: Vec<cl_event> = vec![kernel_event.get()];

    let mut serial_results: [cl_float; M] = [0.0; M];
    let serial_results_read_event = unsafe {
        queue.enqueue_read_buffer(&sums_buffer, CL_NON_BLOCKING, 0, &mut serial_results, &events)?
    };

    serial_results_read_event.wait()?;

    let serial_gpu_load_duration = serial_gpu_load_and_execute_start.elapsed();

    println!("parallel results front: {}", results[0]);
    println!("parallel results back: {}", results[M - 1]);

    println!("serial results front: {}", serial_results[0]);
    println!("serial results back: {}", serial_results[M - 1]);

    let start = Instant::now();
    let cpu_results = colwise_sum(&matrix, M, N);
    let cpu_duration = start.elapsed();

    println!("expected: {}", cpu_results[0]);
    println!("expected: {}", cpu_results[M - 1]);

    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    let serial_start_time = serial_kernel_event.profiling_command_start()?;
    let serial_end_time = serial_kernel_event.profiling_command_end()?;
    let serial_duration = serial_end_time - serial_start_time;
    println!("parallel kernel execution duration (ns): {}", duration);
    println!("serial kernel execution duration (ns): {}", serial_duration);
    println!("cpu execution duration (ns): {}", cpu_duration.as_nanos());

    println!("parallel gpu load and execution (ns): {}", gpu_load_duration.as_nanos());
    println!("serial gpu load and execution (ns): {}", serial_gpu_load_duration.as_nanos());
    
    Ok(())
}
