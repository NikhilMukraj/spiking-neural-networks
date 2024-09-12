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
// }
