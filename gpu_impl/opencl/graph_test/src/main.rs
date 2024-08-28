use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel; // ExecuteKernel
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY}; // CL_MEM_WRITE_ONLY
use opencl3::program::Program;
use opencl3::types::{cl_float, CL_BLOCKING, CL_NON_BLOCKING}; // cl_event
use opencl3::Result;
use std::ptr;


const PROGRAM_SOURCE: &str = r#"
__kernel void colwise_sum(
    __global const float *mat, __global float *res, int m, int n
) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += mat[i * m + gid];
    }
    res[gid] = sum;
}"#;

const KERNEL_NAME: &str = "colwise_sum";

// fn colwise_sum(matrix: &[f32], n: usize, m: usize) -> Vec<f32> {
//     let mut output: Vec<f32> = vec![0.; m];

//     for i in 0..m {
//         for j in 0..n {
//             output[i] += matrix[j * m + i];
//         }
//     }

//     output
// }

fn main()  -> Result<()> {
    // try colwise sum on gpu
    // then move to getting incoming connections on a basic graph
    // then move to calculating input values given basic graph, voltages, and gap conductance kernel
    // then move to a more advanced graph with a seperate key set
    // benchmark calculation of inputs

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
    let _kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    const N: usize = 12;
    const M: usize = 64;
    const FULL_SIZE: usize = M * N;

    let mut matrix: [cl_float; FULL_SIZE] = [1.0; FULL_SIZE];
    for (n, i) in matrix.iter_mut().enumerate().take(FULL_SIZE) {
        *i = n as cl_float;
    }

    let sums: [cl_float; M] = [0.0; M];

    let mut matrix_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, FULL_SIZE, ptr::null_mut())?
    };
    let mut sums_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, M, ptr::null_mut())?
    };

    let _matrix_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut matrix_buffer, CL_BLOCKING, 0, &matrix, &[])? 
    };

    let _sums_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut sums_buffer, CL_NON_BLOCKING, 0, &sums, &[])? 
    };
    
    Ok(())
}
