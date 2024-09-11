// use crate::graph::{Graph, GraphToGPU};
// use super::iterate_and_spike::{IterateAndSpike, IterateAndSpikeGPU, NeurotransmitterType};
// use super::plasticity::Plasticity;
// use super::{Lattice, LatticeHistory};


// pub trait GraphGPU: Default {

// }

// // eventually will need neurotransmitter gpu type
// // eventually add lattice history

// pub struct LatticeGPU<
//     T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
//     U: Graph<K=(usize, usize), V=f32> + GraphToGPU<V>, 
//     V: GraphGPU, 
//     N: NeurotransmitterType,
// > {
//     cell_grid: Vec<Vec<T>>,
//     graph: U,
//     graph_gpu: V,
// }

// impl<T: IterateAndSpike<N=N> + IterateAndSpikeGPU, U: Graph<K=(usize, usize), V=f32> + GraphToGPU<V>, V: GraphGPU, N: NeurotransmitterType> LatticeGPU<T, U, V, N> {
//     pub fn from_lattice<
//         LatticeHistoryCPU: LatticeHistory, 
//         W: Plasticity<T, T, f32>,
//     >(lattice: Lattice<T, U, LatticeHistoryCPU, W, N>) -> Self {
//         LatticeGPU { 
//             cell_grid: lattice.cell_grid, 
//             graph: lattice.graph, 
//             graph_gpu: V::default()
//         }
//     }
// }
