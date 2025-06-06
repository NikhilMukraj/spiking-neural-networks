//! A few different graph implementations to connect various neuron models together.

use std::{
    collections::{HashMap, HashSet}, 
    result::Result,
    fmt::Debug,
    hash::Hash,
    cmp::Eq,
};
use crate::error::GraphError;
#[cfg(feature = "gpu")]
use super::neuron::{
    iterate_and_spike::{IterateAndSpike, IterateAndSpikeGPU, NeurotransmitterTypeGPU},
    spike_train::{SpikeTrainGPU, NeuralRefractorinessGPU},
    CellGrid, InternalGraph, SpikeTrainGrid,
};
#[cfg(feature = "gpu")]
use crate::error::GPUError;
#[cfg(feature = "gpu")]
use opencl3::event::Event;
#[cfg(feature = "gpu")]
use opencl3::{
    memory::{Buffer, CL_MEM_READ_WRITE}, 
    types::{cl_float, cl_uint, cl_int, CL_BLOCKING, CL_NON_BLOCKING}, 
    command_queue::CommandQueue, context::Context,
};
#[cfg(feature = "gpu")]
use std::ptr;


/// Cartesian coordinate represented as unsigned integers for x and y
pub type Position = (usize, usize);

/// Cartesian coordinates as well as and id to specify which graph the coordinates belong to
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct GraphPosition {
    pub id: usize,
    pub pos: Position,
}

/// Implementation of a basic graph
pub trait Graph: Default + Clone + Send + Sync {
    /// Key type
    type K: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy;
    /// Weight type
    type V: Send + Sync + Debug + Clone + Copy;
    /// Sets the identifier of the graph
    fn set_id(&mut self, id: usize);
    /// Gets the identifier of the graph
    fn get_id(&self) -> usize;
    /// Adds a new node to the graph, unconnected to other nodes, no change if node
    /// is already in graph
    fn add_node(&mut self, position: Self::K);
    /// Returns every node or vertex on the graph
    fn get_every_node(&self) -> HashSet<Self::K>;
    /// Returns every node as a reference without cloning
    fn get_every_node_as_ref(&self) -> HashSet<&Self::K>;
    /// Gets the weight between two neurons, errors if the positions are not in the graph 
    /// and returns `None` if there is no connection between the given neurons
    fn lookup_weight(&self, presynaptic: &Self::K, postsynaptic: &Self::K) -> Result<Option<Self::V>, GraphError>; 
    /// Edits the weight between two neurons, errors if the positions are not in the graph,
    /// `None` represents no connection while `Some(U)` represents some weight
    fn edit_weight(&mut self, presynaptic: &Self::K, postsynaptic: &Self::K, weight: Option<Self::V>) -> Result<(), GraphError>;
    /// Returns all presynaptic connections if the position is in the graph
    fn get_incoming_connections(&self, pos: &Self::K) -> Result<HashSet<Self::K>, GraphError>; 
    /// Returns all postsynaptic connections if the position is in the graph
    fn get_outgoing_connections(&self, pos: &Self::K) -> Result<HashSet<Self::K>, GraphError>;
    /// Updates the history of the graph with the current state
    fn update_history(&mut self);
    /// Resets graph history
    fn reset_history(&mut self);
}

pub trait ToGraphPosition: Send + Sync {
    type GraphPos: Graph<K = GraphPosition>;
}

impl<U: Send + Sync + Debug + Clone + Copy> ToGraphPosition for AdjacencyMatrix<Position, U> {
    type GraphPos = AdjacencyMatrix<GraphPosition, U>;
}

impl<U: Send + Sync + Debug + Clone + Copy> ToGraphPosition for AdjacencyList<Position, U> {
    type GraphPos = AdjacencyMatrix<GraphPosition, U>;
}

#[cfg(feature = "gpu")]
/// An implementation of a graph that works on a GPU where the weights are floats
pub struct GraphGPU {
    pub connections: Buffer<cl_uint>,
    pub weights: Buffer<cl_float>,
    pub index_to_position: Buffer<cl_uint>,
    pub size: usize,
}

#[cfg(feature = "gpu")]
/// Handles conversion of graph of CPU to graph on GPU
pub trait GraphToGPU<G> {
    /// Converts graph to graph on GPU
    fn convert_to_gpu<T: IterateAndSpikeGPU>(
        &self, 
        context: &Context, 
        queue: &CommandQueue, 
        cell_grid: &[Vec<T>],
    ) -> Result<G, GPUError>;
    /// Converts from graph on GPU to graph on CPU
    fn convert_from_gpu(&mut self, gpu_graph: G, queue: &CommandQueue) -> Result<(), GPUError>;
}

/// A graph implemented as an adjacency matrix where the positions of each node
/// are converted to `usize` to be index in a 2-dimensional matrix
/// 
/// Example functionality:
/// ```rust
/// # use std::collections::HashSet;
/// use spiking_neural_networks::graph::{Graph, AdjacencyMatrix};
/// 
/// 
/// let mut adjacency_matrix = AdjacencyMatrix::<(usize, usize), f32>::default();
/// adjacency_matrix.add_node((0, 0));
/// adjacency_matrix.add_node((0, 1));
/// adjacency_matrix.add_node((1, 2));
/// 
/// adjacency_matrix.edit_weight(&(0, 0), &(0, 1), Some(0.5));
/// adjacency_matrix.edit_weight(&(1, 2), &(0, 1), Some(1.));
/// assert!(adjacency_matrix.edit_weight(&(0, 1), &(4, 4), Some(1.)).is_err());
/// 
/// assert!(adjacency_matrix.lookup_weight(&(0, 0), &(0, 1)) == Ok(Some(0.5)));
/// assert!(adjacency_matrix.lookup_weight(&(0, 1), &(0, 0)) == Ok(None));
/// assert!(adjacency_matrix.lookup_weight(&(3, 3), &(0, 0)).is_err());
/// 
/// assert!(adjacency_matrix.get_incoming_connections(&(0, 1)) == Ok(HashSet::from([(0, 0), (1, 2)])));
/// assert!(adjacency_matrix.get_outgoing_connections(&(1, 2)) == Ok(HashSet::from([(0, 1)])));
/// 
/// adjacency_matrix.edit_weight(&(0, 0), &(0, 1), None);
/// assert!(adjacency_matrix.lookup_weight(&(0, 0), &(0, 1)) == Ok(None));
/// assert!(adjacency_matrix.get_incoming_connections(&(0, 1)) == Ok(HashSet::from([(1, 2)])));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AdjacencyMatrix<T: Send + Sync + Hash + Eq + PartialEq + Clone + Copy, U: Send + Sync + Debug + Clone + Copy> {
    /// Converts position to a index for the matrix
    pub position_to_index: HashMap<T, usize>,
    /// Converts the index back to a position
    pub index_to_position: HashMap<usize, T>,
    /// Matrix of weights
    pub matrix: Vec<Vec<Option<U>>>,
    /// History of matrix weights
    pub history: Vec<Vec<Vec<Option<U>>>>,
    /// Identifier
    pub id: usize,
}

impl<
    T: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy,
    U: Send + Sync + Debug + Clone + Copy
> AdjacencyMatrix<T, U> {
    pub fn nodes_len(&self) -> usize {
        self.position_to_index.len()
    }
}

impl<
    T: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy, 
    U: Send + Sync + Debug + Clone + Copy
> Graph for AdjacencyMatrix<T, U> {
    type K = T;
    type V = U;

    fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn add_node(&mut self, position: T) {
        if self.position_to_index.contains_key(&position) {
            return;
        }

        let index = self.nodes_len();
    
        self.position_to_index.insert(position, index);
        self.index_to_position.insert(index, position);

        if index != 0 {
            self.matrix.push(vec![None; index]);
            for row in self.matrix.iter_mut() {
                row.push(None);
            }
        } else {
            self.matrix = vec![vec![None]];
        }
    }

    fn get_every_node(&self) -> HashSet<T> {
        self.position_to_index.keys().cloned().collect()
    }

    fn get_every_node_as_ref(&self) -> HashSet<&T> {
        self.position_to_index.keys().collect()
    }

    fn lookup_weight(&self, presynaptic: &T, postsynaptic: &T) -> Result<Option<U>, GraphError> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(GraphError::PostsynapticNotFound(format!("{:#?}", postsynaptic)));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(GraphError::PresynapticNotFound(format!("{:#?}", presynaptic)));
        }

        Ok(self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]])
    }

    fn edit_weight(&mut self, presynaptic: &T, postsynaptic: &T, weight: Option<U>) -> Result<(), GraphError> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(GraphError::PostsynapticNotFound(format!("{:#?}", postsynaptic)));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(GraphError::PresynapticNotFound(format!("{:#?}", presynaptic)));
        }
        
        self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]] = weight;

        Ok(())
    }

    fn get_incoming_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.position_to_index.contains_key(pos) {
            return Err(GraphError::PositionNotFound(format!("{:#?}", pos)));
        }

        // let mut connections: HashSet<T> = HashSet::new();
        // for i in self.position_to_index.keys() {
        //     match self.lookup_weight(i, &pos).unwrap() {
        //         Some(_) => { connections.insert(*i); },
        //         None => {}
        //     };
        // }

        // Ok(connections)

        Ok(
            self.matrix.iter()
                .enumerate()
                .filter_map(|(i, row)| {
                    if row[self.position_to_index[pos]].is_some() { 
                        Some(self.index_to_position[&i]) 
                    } else { 
                        None 
                    }
                })
                .collect()
        )
    }

    fn get_outgoing_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.position_to_index.contains_key(pos) {
            return Err(GraphError::PositionNotFound(format!("{:#?}", pos)));
        }

        let node = self.position_to_index[pos];
        let out_going_connections = self.matrix[node]
            .iter()
            .enumerate()
            .filter_map(|(n, &val)| {
                if val.is_some() {
                    Some(self.index_to_position[&n])
                } else {
                    None
                }
            })
            .collect::<HashSet<T>>();
            
        Ok(out_going_connections)
    }

    fn update_history(&mut self) {
        self.history.push(self.matrix.clone());
    }

    fn reset_history(&mut self) {
        self.history = vec![];
    }
}

impl<T: Send + Sync + Hash + Eq + PartialEq + Clone + Copy, U: Send + Sync + Debug + Clone + Copy> Default for AdjacencyMatrix<T, U> {
    fn default() -> Self {
        AdjacencyMatrix { 
            position_to_index: HashMap::new(), 
            index_to_position: HashMap::new(), 
            matrix: vec![vec![]],
            history: vec![],
            id: 0,
        }
    }
}

#[cfg(feature = "gpu")]
impl GraphToGPU<GraphGPU> for AdjacencyMatrix<(usize, usize), f32> {
    fn convert_to_gpu<T: IterateAndSpikeGPU>(
        &self, 
        context: &Context, 
        queue: &CommandQueue, 
        cell_grid: &[Vec<T>],
    ) -> Result<GraphGPU, GPUError> {
        let size = self.index_to_position.len();
        let grid_row_length = cell_grid.first().unwrap_or(&vec![]).len();

        let weights: Vec<f32> = self.matrix.clone()
            .into_iter()
            .flatten()
            .map(|i| i.unwrap_or(0.))
            .collect();
        let connections: Vec<u32> = self.matrix.clone()
            .into_iter()
            .flatten()
            .map(|i| match i {
                Some(_) => 1,
                None => 0,
            })
            .collect();

        let mut cpu_index_to_position: Vec<_> = self.index_to_position.iter().collect();
        cpu_index_to_position.sort_by_key(|&(key, _)| key);
        let cpu_index_to_position: Vec<_> = cpu_index_to_position.iter()
            .map(|&(_, value)| value)
            .collect();

        let index_to_position: Vec<u32> = cpu_index_to_position
            .iter()
            .map(|pos| (pos.0 * grid_row_length + pos.1) as u32)
            .collect();

        let mut connections_buffer = 
            unsafe { create_buffer::<cl_uint>(context, size * size)? };
        let mut weights_buffer = 
            unsafe { create_buffer::<cl_float>(context, size * size)? };
        let mut index_to_position_buffer = 
            unsafe { create_buffer::<cl_uint>(context, size)? };
        let _connections_write_event = 
            unsafe { write_to_buffer(queue, &mut connections_buffer, &connections)? };
        let _weights_write_event = 
            unsafe { write_to_buffer(queue, &mut weights_buffer, &weights)? };
        let index_to_position_write_event =
            unsafe { write_to_buffer(queue, &mut index_to_position_buffer, &index_to_position)? };
    
        match index_to_position_write_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(
            GraphGPU { 
                connections: connections_buffer, 
                weights: weights_buffer, 
                index_to_position: index_to_position_buffer, 
                size,
            }
        )
    }
    
    #[allow(clippy::needless_range_loop)]
    fn convert_from_gpu(&mut self, gpu_graph: GraphGPU, queue: &CommandQueue) -> Result<(), GPUError> {
        let length = gpu_graph.size;

        let mut connections: Vec<cl_uint> = vec![0; length * length];
        let mut weights: Vec<cl_float> = vec![0.0; length * length];

        let _connections_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.connections, CL_NON_BLOCKING, 0, &mut connections, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
        let weights_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.weights, CL_NON_BLOCKING, 0, &mut weights, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
    
        match weights_read_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        let mut matrix: Vec<Vec<Option<f32>>> = vec![vec![None; length]; length];
        for i in 0..length {
            for j in 0..length {
                let idx = i * length + j;
                matrix[i][j] = if connections[idx] == 1 {
                    Some(weights[idx])
                } else {
                    None
                };
            }
        }

        self.matrix = matrix;  

        Ok(()) 
    }
}

#[cfg(feature = "gpu")]
/// An implementation of a connecting graph that works on a GPU where the weights are floats
pub struct ConnectingGraphGPU {
    pub connections: Buffer<cl_uint>,
    pub weights: Buffer<cl_float>,
    pub index_to_position: Buffer<cl_uint>,
    pub associated_lattices: Buffer<cl_uint>,
    pub associated_lattice_sizes: Buffer<cl_uint>,
    pub size: usize,
}

#[cfg(feature = "gpu")]
/// Handles conversion of a connecting graph of CPU to graph on GPU
pub trait ConnectingGraphToGPU<G> {
    /// Converts graph to graph on GPU
    fn convert_to_gpu<T: IterateAndSpikeGPU>(
        &self, 
        context: &Context, 
        queue: &CommandQueue, 
        cell_grids: &HashMap<usize, &[Vec<T>]>,
    ) -> Result<G, GPUError>;
    /// Converts from graph on GPU to graph on CPU
    fn convert_from_gpu(&mut self, gpu_graph: G, queue: &CommandQueue) -> Result<(), GPUError>;
}

#[cfg(feature = "gpu")]
unsafe fn create_buffer<T>(
    context: &Context,
    size: usize,
) -> Result<Buffer<T>, GPUError> {
    unsafe {
        Buffer::<T>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
            .map_err(|_| GPUError::BufferCreateError)
    }
}

#[cfg(feature = "gpu")]
unsafe fn write_to_buffer<T>(
    queue: &CommandQueue,
    buffer: &mut Buffer<T>,
    data: &[T],
) -> Result<Event, GPUError> {
    unsafe {
        queue.enqueue_write_buffer(buffer, CL_BLOCKING, 0, data, &[])
            .map_err(|_| GPUError::BufferWriteError)
    }
}

#[cfg(feature = "gpu")]
impl ConnectingGraphToGPU<ConnectingGraphGPU> for AdjacencyMatrix<GraphPosition, f32> {
    fn convert_to_gpu<T: IterateAndSpikeGPU>(
        &self, 
        context: &Context, 
        queue: &CommandQueue, 
        cell_grids: &HashMap<usize, &[Vec<T>]>,
    ) -> Result<ConnectingGraphGPU, GPUError> {
        let size = self.index_to_position.len();

        let weights: Vec<f32> = self.matrix.clone()
            .into_iter()
            .flatten()
            .map(|i| i.unwrap_or(0.))
            .collect();
        let connections: Vec<u32> = self.matrix.clone()
            .into_iter()
            .flatten()
            .map(|i| match i {
                Some(_) => 1,
                None => 0,
            })
            .collect();

        let mut cpu_index_to_position: Vec<_> = self.index_to_position.iter().collect();
        cpu_index_to_position.sort_by_key(|&(key, _)| key);
        let cpu_index_to_position: Vec<_> = cpu_index_to_position.iter()
            .map(|&(_, value)| value)
            .collect();

        let mut index_to_position: Vec<u32> = vec![];
        let mut associated_lattices: Vec<u32> = vec![];
        let mut associated_lattice_sizes: Vec<u32> = vec![];

        for graph_pos in cpu_index_to_position {
            let current_cell_grid = cell_grids.get(&graph_pos.id)
                .expect("Valid cell grid");
            let current_row_length = current_cell_grid.first()
                .unwrap_or(&vec![]).len();
            let current_col_length = current_cell_grid.len();
            let index_to_pos_value = (
                graph_pos.pos.0 * current_row_length + graph_pos.pos.1
                ) as u32;

            index_to_position.push(index_to_pos_value);
            associated_lattices.push(graph_pos.id as u32);
            associated_lattice_sizes.push((current_row_length * current_col_length) as u32);
        }

        let mut connections_buffer = unsafe { create_buffer::<cl_uint>(context, size * size)? };
        let mut weights_buffer = unsafe { create_buffer::<cl_float>(context, size * size)? };
        let mut associated_lattices_buffer = unsafe { create_buffer::<cl_uint>(context, size)? };
        let mut associated_lattice_sizes_buffer = unsafe { create_buffer::<cl_uint>(context, size)? };
        let mut index_to_position_buffer = unsafe { create_buffer::<cl_uint>(context, size)? };

        let _connections_write_event = unsafe { write_to_buffer(queue, &mut connections_buffer, &connections)? };
        let _weights_write_event = unsafe { write_to_buffer(queue, &mut weights_buffer, &weights)? };
        let _associated_lattices_write_event =
            unsafe { write_to_buffer(queue, &mut associated_lattices_buffer, &associated_lattices)? };
        let _associated_lattice_sizes_write_event =
            unsafe { write_to_buffer(queue, &mut associated_lattice_sizes_buffer, &associated_lattice_sizes)? };
        let index_to_position_write_event =
            unsafe { write_to_buffer(queue, &mut index_to_position_buffer, &index_to_position)? };

        match index_to_position_write_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(
            ConnectingGraphGPU { 
                connections: connections_buffer, 
                weights: weights_buffer, 
                index_to_position: index_to_position_buffer, 
                associated_lattices: associated_lattices_buffer,
                associated_lattice_sizes: associated_lattice_sizes_buffer,
                size,
            }
        )
    }

    #[allow(clippy::needless_range_loop)]
    fn convert_from_gpu(&mut self, gpu_graph: ConnectingGraphGPU, queue: &CommandQueue) -> Result<(), GPUError> {
        let length = gpu_graph.size;

        let mut connections: Vec<cl_uint> = vec![0; length * length];
        let mut weights: Vec<cl_float> = vec![0.0; length * length];

        let _connections_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.connections, CL_NON_BLOCKING, 0, &mut connections, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
        let weights_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.weights, CL_NON_BLOCKING, 0, &mut weights, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
    
        match weights_read_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        let mut matrix: Vec<Vec<Option<f32>>> = vec![vec![None; length]; length];
        for i in 0..length {
            for j in 0..length {
                let idx = i * length + j;
                matrix[i][j] = if connections[idx] == 1 {
                    Some(weights[idx])
                } else {
                    None
                };
            }
        }

        self.matrix = matrix;  

        Ok(()) 
    }
}

#[cfg(feature = "gpu")]
pub struct InterleavingGraphGPU {
    pub connections: Buffer<cl_uint>,
    pub weights: Buffer<cl_float>,
    pub index_to_position: Buffer<cl_int>,
    pub is_spike_train: Buffer<cl_uint>,
    pub lattice_sizes_map: HashMap<usize, (usize, usize)>,
    pub ordered_keys: Vec<usize>,
    pub spike_train_ordered_keys: Vec<usize>,
    pub spike_train_lattice_sizes_map: HashMap<usize, (usize, usize)>,
    pub size: usize,
}

#[cfg(feature = "gpu")]
impl InterleavingGraphGPU {
    fn calculate_index(
        id: usize,
        row: usize, 
        col: usize,
        lattice_sizes_map: &HashMap<usize, (usize, usize)>, 
        ordered_keys: &Vec<usize>,
    ) -> usize {
        let mut skip_index = 0;
        for i in ordered_keys {
            if *i >= id {
                break;
            }
            let current_size = lattice_sizes_map.get(i).unwrap();
            skip_index += current_size.0 * current_size.1;
        }

        let row_len = lattice_sizes_map.get(&id).unwrap().1;
        
        skip_index + row * row_len + col
    }

    fn calculate_index_spike_train(
        id: usize,
        row: usize, 
        col: usize,
        lattice_sizes_map: &HashMap<usize, (usize, usize)>, 
        ordered_keys: &Vec<usize>,
        spike_train_lattice_sizes_map: &HashMap<usize, (usize, usize)>, 
        spike_train_ordered_keys: &Vec<usize>,
    ) -> usize {                    
        let mut skip_index = 0;
        for i in ordered_keys {
            let current_size = lattice_sizes_map.get(i).unwrap();
            skip_index += current_size.0 * current_size.1;
        }
        for i in spike_train_ordered_keys {
            if *i >= id {
                break;
            }

            let current_size = spike_train_lattice_sizes_map.get(i).unwrap();
            skip_index += current_size.0 * current_size.1;
        }

        let row_len = spike_train_lattice_sizes_map.get(&id).unwrap().1;
        
        skip_index + row * row_len + col
    }

    // modify to take in hashmap of spike train lattices
    // spike train lattices are denoted in index to position as negative indexes
    pub fn convert_to_gpu<
        T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
        U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
        Z: Graph<K=GraphPosition, V=f32>,
        N: NeurotransmitterTypeGPU,
        C: CellGrid<T=T> + InternalGraph<T=U>,
        S: SpikeTrainGPU<N=N, U=R>,
        R: NeuralRefractorinessGPU,
        G: SpikeTrainGrid<T=S>,
    >(
        context: &Context, 
        queue: &CommandQueue,
        lattices: &HashMap<usize, C>,
        spike_train_lattices: &HashMap<usize, G>, 
        connecting_graph: &Z
    ) -> Result<Self, GPUError> {
        let mut associated_lattice_sizes: Vec<u32> = vec![];
        let mut associated_spike_train_lattice_sizes: Vec<u32> = vec![];
        let mut lattice_sizes_map: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut spike_train_lattice_sizes_map: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut index_to_position: Vec<i32> = vec![];
        let mut is_spike_train: Vec<u32> = vec![];
        let mut cell_tracker: Vec<(usize, usize, usize, bool)> = vec![];

        let mut lattice_iterator: Vec<(usize, &C)> = lattices.iter()
            .map(|(&key, value)| (key, value))
            .collect();
        lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
        let ordered_keys: Vec<_> = lattice_iterator.iter().map(|i| i.0).collect();

        let mut spike_train_lattice_iterator: Vec<(usize, &G)> = spike_train_lattices.iter()
            .map(|(&key, value)| (key, value))
            .collect();
        spike_train_lattice_iterator.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));
        let spike_train_ordered_keys: Vec<_> = spike_train_lattice_iterator.iter().map(|i| i.0).collect();

        for (key, value) in &lattice_iterator {
            let mut skip_index = 0;

            for i in associated_lattice_sizes.iter() {
                skip_index += i;
            }

            let current_cell_grid = value.cell_grid();    
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();

            for i in 0..rows {
                for j in 0..cols {
                    index_to_position.push(skip_index as i32 + (i * rows + j) as i32);
                    is_spike_train.push(0);
                    cell_tracker.push((*key, i, j, false));
                }
            }

            associated_lattice_sizes.push((rows * cols) as u32);
            lattice_sizes_map.insert(*key, (rows, cols));
        }

        for (key, value) in &spike_train_lattice_iterator {
            let mut skip_index = 0;

            for i in associated_lattice_sizes.iter() {
                skip_index += i;
            }
            for i in associated_spike_train_lattice_sizes.iter() {
                skip_index += i;
            }

            let current_cell_grid = value.spike_train_grid();    
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();

            for i in 0..rows {
                for j in 0..cols {
                    index_to_position.push(skip_index as i32 + (i * rows + j) as i32);
                    is_spike_train.push(1);
                    cell_tracker.push((*key, i, j, true));
                }
            }

            associated_spike_train_lattice_sizes.push((rows * cols) as u32);
            spike_train_lattice_sizes_map.insert(*key, (rows, cols));
        }

        let size = index_to_position.len();

        let mut weights: Vec<Vec<f32>> = (0..size)
            .map(|_| (0..size).map(|_| 0.).collect())
            .collect();
        let mut connections: Vec<Vec<u32>> = (0..size)
            .map(|_| (0..size).map(|_| 0).collect())
            .collect();

        for (id, row, col, current_is_spike_train) in cell_tracker.iter() {
            for (id_post, row_post, col_post, current_is_spike_train_post) in cell_tracker.iter() {
                if *current_is_spike_train_post {
                    continue;
                }

                let index = if !current_is_spike_train {
                    Self::calculate_index(
                        *id, *row, *col, &lattice_sizes_map, &ordered_keys
                    )
                } else {
                    Self::calculate_index_spike_train(
                        *id, *row, *col, 
                        &lattice_sizes_map, &ordered_keys,
                        &spike_train_lattice_sizes_map, &spike_train_ordered_keys
                    )
                };
                let index_post = Self::calculate_index(
                    *id_post, *row_post, *col_post, &lattice_sizes_map, &ordered_keys
                );

                if *id == *id_post {
                    if let Ok(Some(val)) = lattices.get(id).unwrap().internal_graph().lookup_weight(&(*row, *col), &(*row_post, *col_post)) { 
                        connections[index][index_post] = 1;
                        weights[index][index_post] = val; 
                    }
                } else if let Ok(Some(val)) = connecting_graph.lookup_weight(
                    &GraphPosition { id: *id, pos: (*row, *col) },
                    &GraphPosition { id: *id_post, pos: (*row_post, *col_post) }
                ) {
                    connections[index][index_post] = 1;
                    weights[index][index_post] = val; 
                }
            }
        }

        let weights: Vec<f32> = weights.into_iter().flat_map(|inner| inner.into_iter()).collect();
        let connections: Vec<u32> = connections.into_iter().flat_map(|inner| inner.into_iter()).collect();

        let mut connections_buffer = unsafe { create_buffer::<cl_uint>(context, size * size)? };
        let mut weights_buffer = unsafe { create_buffer::<cl_float>(context, size * size)? };
        let mut index_to_position_buffer = unsafe { create_buffer::<cl_int>(context, size)? };
        let mut is_spike_train_buffer = unsafe { create_buffer::<cl_uint>(context, size)? };

        let _connections_write_event = unsafe { write_to_buffer(queue, &mut connections_buffer, &connections)? };
        let _weights_write_event = unsafe { write_to_buffer(queue, &mut weights_buffer, &weights)? };
        let _is_spike_train_write_event =
            unsafe { write_to_buffer(queue, &mut is_spike_train_buffer, &is_spike_train)? };
        let index_to_position_write_event =
            unsafe { write_to_buffer(queue, &mut index_to_position_buffer, &index_to_position)? };

        match index_to_position_write_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        Ok(
            InterleavingGraphGPU {
                connections: connections_buffer,
                weights: weights_buffer,
                index_to_position: index_to_position_buffer,
                is_spike_train: is_spike_train_buffer,
                lattice_sizes_map,
                spike_train_lattice_sizes_map,
                ordered_keys,
                spike_train_ordered_keys,
                size,
            }
        )
    }

    pub fn convert_to_cpu<
        T: IterateAndSpike<N=N> + IterateAndSpikeGPU, 
        U: Graph<K=(usize, usize), V=f32> + GraphToGPU<GraphGPU>, 
        Z: Graph<K=GraphPosition, V=f32>,
        N: NeurotransmitterTypeGPU,
        C: CellGrid<T=T> + InternalGraph<T=U>,
        S: SpikeTrainGPU<N=N, U=R>,
        R: NeuralRefractorinessGPU,
        G: SpikeTrainGrid<T=S>,
    >(
        queue: &CommandQueue,
        gpu_graph: &InterleavingGraphGPU,
        lattices: &mut HashMap<usize, C>, 
        spike_train_lattices: &mut HashMap<usize, G>, 
        connecting_graph: &mut Z
    ) -> Result<(), GPUError> {
        let length = gpu_graph.size;

        let mut connections: Vec<cl_uint> = vec![0; length * length];
        let mut weights: Vec<cl_float> = vec![0.0; length * length];
        let mut index_to_position: Vec<i32> = vec![0; length];

        let _connections_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.connections, CL_NON_BLOCKING, 0, &mut connections, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
        let _weights_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.weights, CL_NON_BLOCKING, 0, &mut weights, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
        let index_to_position_read_event = unsafe {
            match queue.enqueue_read_buffer(&gpu_graph.index_to_position, CL_NON_BLOCKING, 0, &mut index_to_position, &[]) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::BufferReadError),
            }
        };
    
        match index_to_position_read_event.wait() {
            Ok(_) => {},
            Err(_) => return Err(GPUError::WaitError),
        };

        let mut processed_index_to_position: Vec<GraphPosition> = vec![];

        for key in &gpu_graph.ordered_keys {
            let current_cell_grid = lattices.get(key).unwrap().cell_grid();    
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();

            for i in 0..rows {
                for j in 0..cols {
                    processed_index_to_position.push(
                        GraphPosition { id: *key, pos: (i, j) }
                    );
                }
            }
        }

        for key in &gpu_graph.spike_train_ordered_keys {
            let current_cell_grid = spike_train_lattices.get(key).unwrap().spike_train_grid();    
            let rows = current_cell_grid.len();
            let cols = current_cell_grid.first().unwrap_or(&vec![]).len();

            for i in 0..rows {
                for j in 0..cols {
                    processed_index_to_position.push(
                        GraphPosition { id: *key, pos: (i, j) }
                    );
                }
            }
        }

        let processed_weights: Vec<Vec<f32>> = weights.chunks(gpu_graph.size)
            .map(|chunk| chunk.to_vec()).collect();
        let processed_connections: Vec<Vec<u32>> = connections.chunks(gpu_graph.size)
            .map(|chunk| chunk.to_vec()).collect();

        let mut graphs: HashMap<usize, U> = HashMap::new();

        for i in processed_index_to_position.iter() {
            for j in processed_index_to_position.iter() {
                if gpu_graph.spike_train_ordered_keys.contains(&j.id) {
                    continue;
                }

                let index = if gpu_graph.ordered_keys.contains(&i.id) {
                    Self::calculate_index(
                        i.id, i.pos.0, i.pos.1, &gpu_graph.lattice_sizes_map, &gpu_graph.ordered_keys
                    )
                } else {
                    Self::calculate_index_spike_train(
                        i.id, i.pos.0, i.pos.1, 
                        &gpu_graph.lattice_sizes_map, &gpu_graph.ordered_keys,
                        &gpu_graph.spike_train_lattice_sizes_map, &gpu_graph.spike_train_ordered_keys
                    )
                };
                let index_post = Self::calculate_index(
                    j.id, j.pos.0, j.pos.1, &gpu_graph.lattice_sizes_map, &gpu_graph.ordered_keys
                );

                if i.id == j.id {
                    let current_graph = graphs.entry(i.id).or_default();
                    current_graph.add_node(i.pos);
                    current_graph.add_node(j.pos);

                    if processed_connections[index][index_post] != 1 {
                        current_graph.edit_weight(&i.pos, &j.pos, None).unwrap();
                    } else {
                        current_graph.edit_weight(&i.pos, &j.pos, Some(processed_weights[index][index_post])).unwrap();
                    }
                } else if processed_connections[index][index_post] != 1 {
                    connecting_graph.add_node(*i);
                    connecting_graph.add_node(*j);
                    connecting_graph.edit_weight(i, j, None).unwrap();
                } else {
                    connecting_graph.add_node(*i);
                    connecting_graph.add_node(*j);
                    connecting_graph.edit_weight(i, j, Some(processed_weights[index][index_post])).unwrap();
                }
            }
        }

        for (key, value) in graphs.into_iter() {
            if lattices.contains_key(&key) {
                lattices.get_mut(&key).unwrap().set_internal_graph(value).expect("Valid graph mutation");
            }
        }

        Ok(())
    }
}

/// A graph implemented as an adjacency list
/// 
/// Example functionality:
/// ```rust
/// # use std::collections::HashSet;
/// use spiking_neural_networks::graph::{Graph, AdjacencyList};
/// 
/// 
/// let mut adjacency_list = AdjacencyList::<(usize, usize), f32>::default();
/// adjacency_list.add_node((0, 0));
/// adjacency_list.add_node((0, 1));
/// adjacency_list.add_node((1, 2));
/// 
/// adjacency_list.edit_weight(&(0, 0), &(0, 1), Some(0.5));
/// adjacency_list.edit_weight(&(1, 2), &(0, 1), Some(1.));
/// assert!(adjacency_list.edit_weight(&(0, 1), &(4, 4), Some(1.)).is_err());
/// 
/// assert!(adjacency_list.lookup_weight(&(0, 0), &(0, 1)) == Ok(Some(0.5)));
/// assert!(adjacency_list.lookup_weight(&(0, 1), &(0, 0)) == Ok(None));
/// assert!(adjacency_list.lookup_weight(&(3, 3), &(0, 0)).is_err());
/// 
/// assert!(adjacency_list.get_incoming_connections(&(0, 1)) == Ok(HashSet::from([(0, 0), (1, 2)])));
/// assert!(adjacency_list.get_outgoing_connections(&(1, 2)) == Ok(HashSet::from([(0, 1)])));
/// 
/// adjacency_list.edit_weight(&(0, 0), &(0, 1), None);
/// assert!(adjacency_list.lookup_weight(&(0, 0), &(0, 1)) == Ok(None));
/// assert!(adjacency_list.get_incoming_connections(&(0, 1)) == Ok(HashSet::from([(1, 2)])));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AdjacencyList<
    T: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy, 
    U: Send + Sync + Debug + Clone + Copy
> {
    /// All presynaptic connections
    pub incoming_connections: HashMap<T, HashMap<T, U>>,
    /// All postsynaptic connections
    pub outgoing_connections: HashMap<T, HashSet<T>>,
    /// History of presynaptic connection weights
    pub history: Vec<HashMap<T, HashMap<T, U>>>,
    /// Identifier
    pub id: usize,
}

impl<
    T: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy, 
    U: Send + Sync + Debug + Clone + Copy
> Graph for AdjacencyList<T, U> {
    type K = T;
    type V = U;

    fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn add_node(&mut self, position: T) {
        if self.incoming_connections.contains_key(&position) {
            return;
        }

        self.incoming_connections.entry(position)
            .or_default();
    }

    fn get_every_node(&self) -> HashSet<T> {
        self.incoming_connections.keys().cloned().collect()
    }

    fn get_every_node_as_ref(&self) -> HashSet<&T> {
        self.incoming_connections.keys().collect()
    }

    fn lookup_weight(&self, presynaptic: &T, postsynaptic: &T) -> Result<Option<U>, GraphError> {
        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(GraphError::PostsynapticNotFound(format!("{:#?}", postsynaptic)));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(GraphError::PresynapticNotFound(format!("{:#?}", presynaptic)));
        }

        Ok(self.incoming_connections[postsynaptic].get(presynaptic).copied())
    }

    fn edit_weight(&mut self, presynaptic: &T, postsynaptic: &T, weight: Option<U>) -> Result<(), GraphError> {
        // self.incoming_connections[presynaptic][postsynaptic] = weight;

        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(GraphError::PostsynapticNotFound(format!("{:#?}", postsynaptic)));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(GraphError::PresynapticNotFound(format!("{:#?}", presynaptic)));
        }
        
        match weight {
            Some(value) => {
                self.incoming_connections.entry(*postsynaptic)
                    .or_default()
                    .insert(*presynaptic, value);
    
                if let Some(vector) = self.outgoing_connections.get_mut(presynaptic) {
                    vector.insert(*postsynaptic);
                } else {
                    self.outgoing_connections.insert(*presynaptic, HashSet::from([*postsynaptic]));
                }
            },
            None => {
                if let Some(inner_map) = self.incoming_connections.get_mut(postsynaptic) {
                    inner_map.remove(presynaptic);
                }
                if let Some(connections) = self.outgoing_connections.get_mut(presynaptic) {
                    if connections.contains(postsynaptic) {
                        connections.remove(postsynaptic);
                    }
                }
            },
        }

        // if let Some(positions_and_weights) = self.incoming_connections.get_mut(postsynaptic) {
        //     positions_and_weights.insert(*presynaptic, weight);
        // }

        Ok(())
    }

    // to be cached
    // or point to reference
    fn get_incoming_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.incoming_connections.contains_key(pos) {
            return Err(GraphError::PositionNotFound(format!("{:#?}", pos)));
        }

        Ok(self.incoming_connections[pos].keys().cloned().collect::<HashSet<T>>())
    }

    // to be cached
    // or point to reference
    fn get_outgoing_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.incoming_connections.contains_key(pos) {
            return Err(GraphError::PositionNotFound(format!("{:#?}", pos)));
        }

        // self.outgoing_connections[pos].clone()
        Ok(
            self.outgoing_connections.get(pos)
                .unwrap_or(&HashSet::from([]))
                .clone()
        )
    }

    fn update_history(&mut self) {
        self.history.push(self.incoming_connections.clone());
    }

    fn reset_history(&mut self) {
        self.history = vec![];
    }
}

impl<
    T: Send + Sync + Debug + Hash + Eq + PartialEq + Clone + Copy, 
    U: Send + Sync + Debug + Clone + Copy
> Default for AdjacencyList<T, U> {
    fn default() -> Self {
        AdjacencyList { 
            incoming_connections: HashMap::new(), 
            outgoing_connections: HashMap::new(), 
            history: vec![], 
            id: 0,
        }
    }
}
