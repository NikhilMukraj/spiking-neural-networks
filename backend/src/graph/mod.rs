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
use opencl3::{
    memory::{Buffer, CL_MEM_READ_WRITE}, types::{cl_float, cl_uint, CL_BLOCKING}, 
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
pub trait Graph: Default + Send + Sync {
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
    pub size: u32,
}

#[cfg(feature = "gpu")]
/// Handles conversion of graph of CPU to graph on GPU
pub trait GraphToGPU {
    /// Converts graph to graph on GPU
    fn convert_to_gpu(&self, context: &Context, queue: &CommandQueue) -> GraphGPU;
    /// Converts from graph on GPU to graph on CPU
    fn convert_from_gpu(&mut self, gpu_graph: GraphGPU);
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
#[derive(Clone, Debug)]
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

impl GraphToGPU for AdjacencyMatrix<(usize, usize), f32> {
    fn convert_to_gpu(&self, context: &Context, queue: &CommandQueue) -> GraphGPU {
        let length = self.position_to_index.len();

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
        let index_to_position: Vec<u32> = self.index_to_position.values()
            .map(|pos| (pos.0 * length + pos.1) as u32)
            .collect();

        let mut connections_buffer = unsafe {
            Buffer::<cl_uint>::create(context, CL_MEM_READ_WRITE, length * length, ptr::null_mut())
                .expect("Could not create buffer")
        };
        let mut weights_buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, length * length, ptr::null_mut())
                .expect("Could not create buffer")
        };
        let mut index_to_position_buffer = unsafe {
            Buffer::<cl_uint>::create(context, CL_MEM_READ_WRITE, length, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let _connections_write_event = unsafe { 
            queue.enqueue_write_buffer(&mut connections_buffer, CL_BLOCKING, 0, &connections, &[])
                .expect("Could not write to buffer") 
        };
        let _weights_write_event = unsafe { 
            queue.enqueue_write_buffer(&mut weights_buffer, CL_BLOCKING, 0, &weights, &[])
                .expect("Could not write to buffer") 
        };
        let index_to_position_write_event = unsafe { 
            queue.enqueue_write_buffer(&mut index_to_position_buffer, CL_BLOCKING, 0, &index_to_position, &[])
                .expect("Could not write to buffer") 
        };

        index_to_position_write_event.wait().expect("Could not wait");

        GraphGPU { 
            connections: connections_buffer, 
            weights: weights_buffer, 
            index_to_position: index_to_position_buffer, 
            size: length as u32,
        }
    }
    
    fn convert_from_gpu(&mut self, _gpu_graph: GraphGPU) {
        todo!()
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
#[derive(Clone, Debug)]
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
        // println!("{:#?} {:#?}", presynaptic, postsynaptic);

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
