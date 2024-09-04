// use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE};
// use opencl3::context::Context;
// use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
// use opencl3::kernel::{Kernel, ExecuteKernel};
// use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
// use opencl3::program::Program;
// use opencl3::types::{cl_float, cl_uint, cl_event, CL_BLOCKING, CL_NON_BLOCKING};
// use opencl3::Result;
// use std::ptr;
// use std::time::Instant;
// use rand::Rng;


// const INPUTS_KERNEL: &str = r#"
// __kernel void calculate_internal_electrical_inputs(
//     __global const uint *connections, 
//     __global const float *weights, 
//     __global const float *gap_conductances,
//     __global const float *voltages,
//     uint n, 
//     __global float *res
// ) {
//     int gid = get_global_id(0);

//     float sum = 0.0f;
//     uint count = 0;
//     for (int i = 0; i < n; i++) {
//         if (connections[i * n + gid] == 1) {
//             float gap_junction = gap_conductances[gid] * (voltages[i] - voltages[gid]);
//             sum += weights[i * n + gid] * gap_junction;
//             count++;
//         }
//     }
    
//     if (count != 0) {
//         res[gid] = sum / count;
//     } else {
//         res[gid] = 0;
//     }
// }
// "#;

// const INPUTS_KERNEL_NAME: &str = "calculate_internal_electrical_inputs";

// const ITERATE_AND_SPIKE_KERNEL: &str = r#"
// __kernel void iterate_and_spike(
//     __global const float *inputs,
//     __global float *v,
//     __global float *g,
//     __global float *e,
//     __global float *v_th,
//     __global float *v_reset,
//     __global uint *is_spiking,
//     __global float *dt,
// ) {
//     int gid = get_global_id(0);

//     v[gid] += (g[gid] * (v[gid] - e[gid]) + inputs[gid]) * dt[gid];
//     if (v[gid] >= v_th[gid]) {
//         v[gid] = v_reset[gid];
//         is_spiking[gid] = 1;
//     } else {
//         is_spiking[gid] = 0;
//     }
// }
// "#;

// const ITERATE_AND_SPIKE_KERNEL_NAME: &str = "iterate_and_spike";

// #[allow(clippy::too_many_arguments)]
// fn cpu_iterate_and_spike(
//     inputs: &[f32], 
//     v: &mut [f32], 
//     g: &mut [f32],
//     e: &mut [f32],
//     v_th: &mut [f32],
//     v_reset: &mut [f32],
//     is_spiking: &mut [bool],
//     dt: &mut [f32],
// ) {
//     for (n, i) in inputs.iter().enumerate() {
//         v[n] += (g[n] * (v[n] - e[n]) + i) * dt[n];
//         if v[n] >= v_th[n] {
//             v[n] = v_reset[n];
//             is_spiking[n] = true;
//         } else {
//             is_spiking[n] = false;
//         }
//     }
// }

// fn create_random_flattened_adj_matrix(size: usize, lower_bound: f32, upper_bound: f32) -> (Vec<bool>, Vec<f32>) {
//     let full_size = size * size;
    
//     let mut connections = vec![false; full_size];
//     let mut weights = vec![0.; full_size];

//     let mut rng = rand::thread_rng();

//     for i in 0..(full_size) {
//         connections[i] = rng.gen_bool(0.5);
//         if connections[i] {
//             weights[i] = rng.gen_range(lower_bound..=upper_bound);
//         }
//     }

//     (connections, weights)
// }

// fn cpu_electrical_inputs(
//     connections: &[bool], 
//     weights: &[f32], 
//     n: usize, 
//     gap_conductances: &[f32], 
//     voltages: &[f32]
// ) -> Vec<f32> {
//     let connections: Vec<Vec<bool>> = connections.chunks(n)
//         .map(|chunk| chunk.to_vec())
//         .collect();

//     let weights: Vec<Vec<f32>> = weights.chunks(n)
//         .map(|chunk| chunk.to_vec())
//         .collect();

//     let mut result = Vec::new();

//     for i in 0..n {
//         let mut sum: f32 = 0.;
//         let mut counter: usize = 0;
    
//         for (j, row) in connections.iter().enumerate() {
//             if row[i] { 
//                 sum += weights[j][i] * gap_conductances[i] * (voltages[j] - voltages[i]);
//                 counter += 1;
//             }
//         }

//         if counter != 0 {
//             sum /= counter as f32;
//         } else {
//             sum = 0.;
//         }

//         result.push(sum);
//     }

//     result
// }

// fn assert_vec_almost_eq(a: &[f32], b: &[f32], tolerance: f32) {
//     assert_eq!(a.len(), b.len(), "Vectors must have the same length");
//     for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
//         assert!(
//             (ai - bi).abs() <= tolerance,
//             "Assertion failed at index {}: (left == right) (left: `{}`, right: `{}`, tolerance: `{}`)",
//             i, ai, bi, tolerance
//         );
//     }
// }

fn main() {
    // move relevant data to gpu
    // for n in iterations
        // execute inputs kernel
        // execute iterate kernel

    // benchmark on cpu
}
