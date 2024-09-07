use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_float, cl_uint, CL_BLOCKING, CL_NON_BLOCKING}; // cl_event
use opencl3::Result;
use std::ptr;
use std::time::Instant;
// use std::time::Instant;
use rand::Rng;


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

const ITERATE_AND_SPIKE_KERNEL: &str = r#"
__kernel void iterate_and_spike(
    __global const float *inputs,
    __global const uint *index_to_position,
    __global float *v,
    __global float *g,
    __global float *e,
    __global float *v_th,
    __global float *v_reset,
    __global uint *is_spiking,
    __global float *dt
) {
    int gid = get_global_id(0);
    int index = index_to_position[gid];

    v[index] += (g[index] * (v[index] - e[index]) + inputs[index]) * dt[index];
    if (v[index] >= v_th[index]) {
        v[index] = v_reset[index];
        is_spiking[index] = 1;
    } else {
        is_spiking[index] = 0;
    }
}
"#;

const ITERATE_AND_SPIKE_KERNEL_NAME: &str = "iterate_and_spike";

#[allow(clippy::too_many_arguments)]
fn cpu_iterate_and_spike(
    inputs: &[f32], 
    index_to_position: &[u32],
    v: &mut [f32], 
    g: &mut [f32],
    e: &mut [f32],
    v_th: &mut [f32],
    v_reset: &mut [f32],
    is_spiking: &mut [u32],
    dt: &mut [f32],
) {
    for (n, i) in inputs.iter().enumerate() {
        let index = index_to_position[n] as usize;

        v[index] += (g[index] * (v[index] - e[index]) + i) * dt[index];
        if v[index] >= v_th[index] {
            v[index] = v_reset[index];
            is_spiking[index] = 1;
        } else {
            is_spiking[index] = 0;
        }
    }
}

fn cpu_electrical_inputs(
    connections: &[u32], 
    weights: &[f32], 
    index_to_position: &[u32],
    n: usize, 
    gap_conductances: &[f32], 
    voltages: &[f32]
) -> Vec<f32> {
    let connections: Vec<Vec<u32>> = connections.chunks(n)
        .map(|chunk| chunk.to_vec())
        .collect();

    let weights: Vec<Vec<f32>> = weights.chunks(n)
        .map(|chunk| chunk.to_vec())
        .collect();

    let mut result = Vec::new();

    for i in 0..n {
        let mut sum: f32 = 0.;
        let mut counter: usize = 0;
    
        for (j, row) in connections.iter().enumerate() {
            if row[i] == 1 { 
                let presynaptic_index = index_to_position[j] as usize;
                let postsynaptic_index = index_to_position[i] as usize;
                sum += weights[j][i] * gap_conductances[postsynaptic_index] * 
                (voltages[presynaptic_index] - voltages[postsynaptic_index]);
                counter += 1;
            }
        }

        if counter != 0 {
            sum /= counter as f32;
        } else {
            sum = 0.;
        }

        result.push(sum);
    }

    result
}

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

fn assert_vec_eq_with_tolerance(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() <= tolerance,
            "Assertion failed at index {}: (left == right) (left: `{}`, right: `{}`, tolerance: `{}`)",
            i, ai, bi, tolerance
        );
    }
}

macro_rules! create_cl_float_buffer {
    ($name:ident, $context:expr, $num:ident) => {
        let mut $name = unsafe {
            Buffer::<cl_float>::create($context, CL_MEM_READ_ONLY, $num, ptr::null_mut())?
        };
    };
}

fn main() -> Result<()> {
    // move relevant data to gpu
    // for n in iterations
        // execute inputs kernel
        // execute iterate kernel

    // check against cpu calculation
        // potential issues: 
            // conversion between rust and opencl types
            // kernel waits not being correct
            
    // benchmark on cpu

    // seperate key set associating indexes to positions

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

    let incoming_connections_program = Program::create_and_build_from_source(&context, INPUTS_KERNEL, "")
        .expect("Program::create_and_build_from_source failed");
    let incoming_connections_kernel = Kernel::create(&incoming_connections_program, INPUTS_KERNEL_NAME)
        .expect("Kernel::create failed");

    let iterate_and_spike_program = Program::create_and_build_from_source(&context, ITERATE_AND_SPIKE_KERNEL, "")
        .expect("Program::create_and_build_from_source failed");
    let iterate_and_spike_kernel = Kernel::create(&iterate_and_spike_program, ITERATE_AND_SPIKE_KERNEL_NAME)
        .expect("Kernel::create failed");

    const N: usize = 1000;
    const NUM_ITERATIONS: usize = 10;

    let (connections, weights) = create_random_flattened_adj_matrix(N, 0., 2.);
    let connections: Vec<u32> = connections.iter().map(|i| if *i { 1 } else { 0 } ).collect();

    let index_to_position: Vec<u32> = (0..N).map(|i| i as u32).collect();

    let mut voltages: Vec<f32> = (0..N).map(|_| rand::thread_rng().gen_range(-75.0..-50.)).collect();
    let mut gs: Vec<f32> = (0..N).map(|_| -0.1).collect();
    let mut es: Vec<f32> = (0..N).map(|_| 0.).collect();
    let mut v_ths: Vec<f32> = (0..N).map(|_| -55.).collect();
    let mut v_resets: Vec<f32> = (0..N).map(|_| -75.).collect();
    let mut is_spikings: Vec<u32> = (0..N).map(|_| 0).collect();
    let mut dts: Vec<f32> = (0..N).map(|_| 0.1).collect();
    let gap_conductances: Vec<f32> = (0..N).map(|_| 10.).collect();
    let sums: Vec<f32> = (0..N).map(|_| 0.).collect();

    // let mut connections_array: Vec<cl_uint> = vec![0; N * N];
    // let mut weights_array: Vec<cl_float> = vec![0.; N * N];
    // let sums_array: Vec<cl_float> = vec![0.; N];
    // let mut gap_conductances_array: Vec<cl_float> = vec![0., N];
    // let mut voltages_array: Vec<cl_float> = vec![0.; N];
    // let mut gs_array: Vec<cl_float> = vec![0.; N];
    // let mut es_array: Vec<cl_float> = vec![0.; N];
    // let mut v_ths_array: Vec<cl_float> = vec![0.; N];
    // let mut v_resets_array: Vec<cl_float> = vec![0.; N];
    // let mut is_spikings_array: Vec<cl_uint> = vec![0; N];
    // let mut dts_array: Vec<cl_float> = vec![0.; N];

    // for i in 0..N {
    //     voltages_array[i] = voltages[i];
    //     gap_conductances_array[i] = gap_conductances[i];
    //     gs_array[i] = gs[i];
    //     es_array[i] = es[i];
    //     v_ths_array[i] = v_ths[i];
    //     v_resets_array[i] = v_resets[i];
    //     is_spikings_array[i] = is_spikings[i];
    //     dts_array[i] = dts[i];
    // }

    let mut connections_buffer = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, N * N, ptr::null_mut())?
    };
    let mut weights_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, N * N, ptr::null_mut())?
    };
    let mut index_to_position_buffer = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, N, ptr::null_mut())?
    };
    let mut sums_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, N, ptr::null_mut())?
    };

    let mut is_spikings_buffer = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_WRITE_ONLY, N, ptr::null_mut())?
    };

    create_cl_float_buffer!(voltages_buffer, &context, N);
    create_cl_float_buffer!(gap_conductances_buffer, &context, N);
    create_cl_float_buffer!(gs_buffer, &context, N);
    create_cl_float_buffer!(es_buffer, &context, N);
    create_cl_float_buffer!(v_ths_buffer, &context, N);
    create_cl_float_buffer!(v_resets_buffer, &context, N);
    create_cl_float_buffer!(dts_buffer, &context, N);

    let mut cl_float_buffers: Vec<(&mut Buffer<cl_float>, &Vec<f32>)> = vec![
        (&mut voltages_buffer, &voltages),
        (&mut gap_conductances_buffer, &gap_conductances),
        (&mut gs_buffer, &gs),
        (&mut es_buffer, &es),
        (&mut v_ths_buffer, &v_ths),
        (&mut v_resets_buffer, &v_resets),
        (&mut dts_buffer, &dts),
    ];

    for (buffer, array) in cl_float_buffers.iter_mut() {
        let _ = unsafe { 
            queue.enqueue_write_buffer(buffer, CL_BLOCKING, 0, array, &[])? 
        };
    }

    let _is_spiking_event = unsafe {
        queue.enqueue_write_buffer(&mut is_spikings_buffer, CL_BLOCKING, 0, &is_spikings, &[])?
    };

    let _connections_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut connections_buffer, CL_BLOCKING, 0, &connections, &[])? 
    };
    let _weights_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut weights_buffer, CL_BLOCKING, 0, &weights, &[])? 
    };
    let _index_to_position_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut index_to_position_buffer, CL_BLOCKING, 0, &index_to_position, &[])? 
    };

    let sums_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut sums_buffer, CL_NON_BLOCKING, 0, &sums, &[])? 
    };

    sums_write_event.wait()?;

    let n_cl: cl_uint = N as u32;

    let start = Instant::now();

    for _ in 0..NUM_ITERATIONS {
        let gap_junctions_event = unsafe {
            ExecuteKernel::new(&incoming_connections_kernel)
                .set_arg(&connections_buffer)
                .set_arg(&weights_buffer)
                .set_arg(&index_to_position_buffer)
                .set_arg(&gap_conductances_buffer)
                .set_arg(&voltages_buffer)
                .set_arg(&n_cl)
                .set_arg(&sums_buffer)
                .set_global_work_size(N) // number of threads executing in parallel
                // .set_wait_event(&sums_write_event)
                .enqueue_nd_range(&queue)?
        };

        // gap_junctions_event.wait()?;

        let iterate_and_spike_event = unsafe {
            ExecuteKernel::new(&iterate_and_spike_kernel)
                .set_arg(&sums_buffer)
                .set_arg(&index_to_position_buffer)
                .set_arg(&voltages_buffer)
                .set_arg(&gs_buffer)
                .set_arg(&es_buffer)
                .set_arg(&v_ths_buffer)
                .set_arg(&v_resets_buffer)
                .set_arg(&is_spikings_buffer)
                .set_arg(&dts_buffer)
                .set_global_work_size(N) // number of threads executing in parallel
                .set_wait_event(&gap_junctions_event)
                .enqueue_nd_range(&queue)?
        };

        iterate_and_spike_event.wait()?;
    }

    let gpu_duration = start.elapsed().as_nanos();

    let mut results: [cl_float; N] = [0.0; N];
    let results_read_event = unsafe {
        queue.enqueue_read_buffer(&voltages_buffer, CL_NON_BLOCKING, 0, &mut results, &[])?
    };

    results_read_event.wait()?;

    let start = Instant::now();

    for _ in 0..NUM_ITERATIONS {
        let inputs = cpu_electrical_inputs(
            &connections, 
            &weights,
            &index_to_position,
            N, 
            &gap_conductances, 
            &voltages
        );

        cpu_iterate_and_spike(
            &inputs, 
            &index_to_position,
            &mut voltages, 
            &mut gs, 
            &mut es, 
            &mut v_ths, 
            &mut v_resets, 
            &mut is_spikings, 
            &mut dts,
        );
    }

    let cpu_duration = start.elapsed().as_nanos();

    assert_vec_eq_with_tolerance(&results, &voltages, 1.);

    println!("CPU execution (ns): {}", cpu_duration);
    println!("GPU execution (ns): {}", gpu_duration);
    
    Ok(())
}
