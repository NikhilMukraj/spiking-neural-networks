//! A few different graph implementations to connect various neuron models together.

use std::{
    collections::{HashMap, HashSet}, 
    result::Result,
    fmt::Debug,
    hash::Hash,
    cmp::Eq,
};
use crate::error::{GraphError, GraphErrorKind};
use crate::neuron::iterate_and_spike::GaussianParameters;


/// Cartesian coordinate represented as unsigned integers for x and y
pub type Position = (usize, usize);

/// Cartesian coordinates as well as and id to specify which graph the coordinates belong to
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct GraphPosition {
    pub id: usize,
    pub pos: Position,
}

/// Implementation of a basic graph
pub trait Graph: Default {
    type T: Debug + Hash + Eq + PartialEq + Clone + Copy;
    /// Sets the identifier of the graph
    fn set_id(&mut self, id: usize);
    /// Gets the identifier of the graph
    fn get_id(&self) -> usize;
    /// Adds a new node to the graph, unconnected to other nodes, no change if node
    /// is already in graph
    fn add_node(&mut self, position: Self::T);
    /// Initializes connections between a set of presynaptic neurons and one postsynaptic neuron, 
    /// if `weight_params` is `None`, then each connection is initialized as `1.`, otherwise it
    /// is initialized as a normally distributed random value based on the inputted weight parameters
    fn initialize_connections(
        &mut self, 
        postsynaptic: Self::T, 
        presynaptic_connections: Vec<Self::T>, 
        weight_params: &Option<GaussianParameters>,
    );
    /// Returns every node or vertex on the graph
    fn get_every_node(&self) -> HashSet<Self::T>;
    /// Returns every node as a reference without cloning
    fn get_every_node_as_ref(&self) -> HashSet<&Self::T>;
    /// Gets the weight between two neurons, errors if the positions are not in the graph 
    /// and returns `None` if there is no connection between the given neurons
    fn lookup_weight(&self, presynaptic: &Self::T, postsynaptic: &Self::T) -> Result<Option<f64>, GraphError>; 
    /// Edits the weight between two neurons, errors if the positions are not in the graph,
    /// `None` represents no connection while `Some(f64)` represents some weight
    fn edit_weight(&mut self, presynaptic: &Self::T, postsynaptic: &Self::T, weight: Option<f64>) -> Result<(), GraphError>;
    /// Returns all presynaptic connections if the position is in the graph
    fn get_incoming_connections(&self, pos: &Self::T) -> Result<HashSet<Self::T>, GraphError>; 
    /// Returns all postsynaptic connections if the position is in the graph
    fn get_outgoing_connections(&self, pos: &Self::T) -> Result<HashSet<Self::T>, GraphError>;
    /// Updates the history of the graph with the current state
    fn update_history(&mut self);
}

pub trait ToGraphPosition {
    type GraphPos: Graph<T = GraphPosition>;
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> ToGraphPosition for AdjacencyMatrix<T> {
    type GraphPos = AdjacencyMatrix<GraphPosition>;
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> ToGraphPosition for AdjacencyList<T> {
    type GraphPos = AdjacencyMatrix<GraphPosition>;
}

/// A graph implemented as an adjacency matrix where the positions of each node
/// are converted to `usize` to be index in a 2-dimensional matrix
#[derive(Clone, Debug)]
pub struct AdjacencyMatrix<T: Hash + Eq + PartialEq + Clone + Copy> {
    /// Converts position to a index for the matrix
    pub position_to_index: HashMap<T, usize>,
    /// Converts the index back to a position
    pub index_to_position: HashMap<usize, T>,
    /// Matrix of weights
    pub matrix: Vec<Vec<Option<f64>>>,
    /// History of matrix weights
    pub history: Vec<Vec<Vec<Option<f64>>>>,
    /// Identifier
    pub id: usize,
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> AdjacencyMatrix<T> {
    pub fn nodes_len(&self) -> usize {
        self.position_to_index.len()
    }
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> Graph for AdjacencyMatrix<T> {
    type T = T;

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

    fn initialize_connections(
        &mut self, 
        postsynaptic: T, 
        connections: Vec<T>, 
        weight_params: &Option<GaussianParameters>,
    ) {
        if !self.position_to_index.contains_key(&postsynaptic) {
            self.add_node(postsynaptic)
        }
        for i in connections.iter() {
            if !self.position_to_index.contains_key(i) {
                self.add_node(*i);
            }

            let weight = match weight_params {
                Some(value) => {
                    Some(
                        value.get_random_number()
                    )
                },
                None => Some(1.0),
            };

            self.edit_weight(i, &postsynaptic, weight).unwrap();
        }
    }

    fn get_every_node(&self) -> HashSet<T> {
        self.position_to_index.keys().cloned().collect()
    }

    fn get_every_node_as_ref(&self) -> HashSet<&T> {
        self.position_to_index.keys().collect()
    }

    fn lookup_weight(&self, presynaptic: &T, postsynaptic: &T) -> Result<Option<f64>, GraphError> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(GraphError::new(GraphErrorKind::PostsynapticNotFound, file!(), line!()));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(GraphError::new(GraphErrorKind::PresynapticNotFound, file!(), line!()));
        }

        Ok(self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]])
    }

    fn edit_weight(&mut self, presynaptic: &T, postsynaptic: &T, weight: Option<f64>) -> Result<(), GraphError> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(GraphError::new(GraphErrorKind::PostsynapticNotFound, file!(), line!()));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(GraphError::new(GraphErrorKind::PresynapticNotFound, file!(), line!()));
        }
        
        self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]] = weight;

        Ok(())
    }

    fn get_incoming_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.position_to_index.contains_key(pos) {
            return Err(GraphError::new(GraphErrorKind::PositionNotFound, file!(), line!()));
        }

        let mut connections: HashSet<T> = HashSet::new();
        for i in self.position_to_index.keys() {
            match self.lookup_weight(i, &pos).unwrap() {
                Some(_) => { connections.insert(*i); },
                None => {}
            };
        }

        Ok(connections)
    }

    fn get_outgoing_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.position_to_index.contains_key(pos) {
            return Err(GraphError::new(GraphErrorKind::PositionNotFound, file!(), line!()));
        }

        let node = self.position_to_index[pos];
        let out_going_connections = self.matrix[node]
            .iter()
            .enumerate()
            .filter(|(_, &val)| val.is_some())
            .map(|(n, _)| self.index_to_position[&n])
            .collect::<HashSet<T>>();
            
        Ok(out_going_connections)
    }

    fn update_history(&mut self) {
        self.history.push(self.matrix.clone());
    }
}

impl<T: Hash + Eq + PartialEq + Clone + Copy> Default for AdjacencyMatrix<T> {
    fn default() -> Self {
        AdjacencyMatrix { 
            position_to_index: HashMap::new(), 
            index_to_position: HashMap::new(), 
            matrix: vec![vec![]],
            history: vec![vec![vec![]]],
            id: 0,
        }
    }
}

/// A graph implemented as an adjacency list
#[derive(Clone, Debug)]
pub struct AdjacencyList<T: Debug + Hash + Eq + PartialEq + Clone + Copy> {
    /// All presynaptic connections
    pub incoming_connections: HashMap<T, HashMap<T, f64>>,
    /// All postsynaptic connections
    pub outgoing_connections: HashMap<T, HashSet<T>>,
    /// History of presynaptic connection weights
    pub history: Vec<HashMap<T, HashMap<T, f64>>>,
    /// Identifier
    pub id: usize,
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> Graph for AdjacencyList<T> {
    type T = T;

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
            .or_insert_with(HashMap::new);
    }

    fn initialize_connections(
        &mut self, 
        postsynaptic: T, 
        connections: Vec<T>, 
        weight_params: &Option<GaussianParameters>,
    ) {
        for i in connections.iter() {
            let weight = match weight_params {
                Some(value) => {
                    value.get_random_number()
                },
                None => 1.0,
            };

            if !self.incoming_connections.contains_key(&postsynaptic) {
                self.incoming_connections.entry(postsynaptic)
                    .or_insert_with(HashMap::new)
                    .insert(*i, weight);
            } else {
                if let Some(positions_and_weights) = self.incoming_connections.get_mut(&postsynaptic) {
                    positions_and_weights.insert(*i, weight);
                }
            }

            if let Some(vector) = self.outgoing_connections.get_mut(&i) {
                vector.insert(postsynaptic);
            } else {
                self.outgoing_connections.insert(*i, HashSet::from([postsynaptic]));
            }
        }
    }

    fn get_every_node(&self) -> HashSet<T> {
        self.incoming_connections.keys().cloned().collect()
    }

    fn get_every_node_as_ref(&self) -> HashSet<&T> {
        self.incoming_connections.keys().collect()
    }

    fn lookup_weight(&self, presynaptic: &T, postsynaptic: &T) -> Result<Option<f64>, GraphError> {
        // println!("{:#?} {:#?}", presynaptic, postsynaptic);

        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(GraphError::new(GraphErrorKind::PostsynapticNotFound, file!(), line!()));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(GraphError::new(GraphErrorKind::PresynapticNotFound, file!(), line!()));
        }

        Ok(self.incoming_connections[postsynaptic].get(presynaptic).copied())
    }

    fn edit_weight(&mut self, presynaptic: &T, postsynaptic: &T, weight: Option<f64>) -> Result<(), GraphError> {
        // self.incoming_connections[presynaptic][postsynaptic] = weight;

        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(GraphError::new(GraphErrorKind::PostsynapticNotFound, file!(), line!()));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(GraphError::new(GraphErrorKind::PresynapticNotFound, file!(), line!()));
        }
        
        match weight {
            Some(value) => {
                self.incoming_connections.entry(*postsynaptic)
                    .or_insert_with(HashMap::new)
                    .insert(*presynaptic, value);
    
                if let Some(vector) = self.outgoing_connections.get_mut(&presynaptic) {
                    vector.insert(*postsynaptic);
                } else {
                    self.outgoing_connections.insert(*presynaptic, HashSet::from([*postsynaptic]));
                }
            },
            None => {
                if let Some(inner_map) = self.incoming_connections.get_mut(&postsynaptic) {
                    inner_map.remove(&presynaptic);
                }
                if let Some(connections) = self.outgoing_connections.get_mut(&presynaptic) {
                    if connections.contains(&postsynaptic) {
                        connections.remove(&postsynaptic);
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
            return Err(GraphError::new(GraphErrorKind::PositionNotFound, file!(), line!()));
        }

        Ok(self.incoming_connections[pos].keys().cloned().collect::<HashSet<T>>())
    }

    // to be cached
    // or point to reference
    fn get_outgoing_connections(&self, pos: &T) -> Result<HashSet<T>, GraphError> {
        if !self.incoming_connections.contains_key(pos) {
            return Err(GraphError::new(GraphErrorKind::PositionNotFound, file!(), line!()));
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
}

impl<T: Debug + Hash + Eq + PartialEq + Clone + Copy> Default for AdjacencyList<T> {
    fn default() -> Self {
        AdjacencyList { 
            incoming_connections: HashMap::new(), 
            outgoing_connections: HashMap::new(), 
            history: vec![], 
            id: 0,
        }
    }
}
