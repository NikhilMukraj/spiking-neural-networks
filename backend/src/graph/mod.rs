//! A few different graph implementations to connect various neuron models together.

use std::{
    collections::{HashMap, HashSet}, 
    fs::File, 
    io::{Write, BufWriter, Result, Error, ErrorKind}, 
    fmt::Display,
};
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
pub trait GraphFunctionality: Default {
    /// Sets the identifier of the graph
    fn set_id(&mut self, id: usize);
    /// Gets the identifier of the graph
    fn get_id(&self) -> usize;
    /// Adds a new vertex to the graph, unconnected to other graphs
    fn add_vertex(&mut self, position: GraphPosition);
    /// Initializes connections between a set of presynaptic neurons and one postsynaptic neuron, 
    /// if `weight_params` is `None`, then each connection is initialized as `1.`, otherwise it
    /// is initialized as a normally distributed random value based on the inputted weight parameters
    fn initialize_connections(
        &mut self, 
        postsynaptic: GraphPosition, 
        presynaptic_connections: Vec<GraphPosition>, 
        weight_params: &Option<GaussianParameters>,
    );
    /// Returns every node or vertex on the graph
    fn get_every_node(&self) -> Vec<GraphPosition>;
    /// Gets the weight between two neurons, errors if the positions are not in the graph 
    /// and returns `None` if there is no connection between the given neurons
    fn lookup_weight(&self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition) -> Result<Option<f64>>; 
    /// Edits the weight between two neurons, errors if the positions are not in the graph,
    /// `None` represents no connection while `Some(f64)` represents some weight
    fn edit_weight(&mut self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition, weight: Option<f64>) -> Result<()>;
    /// Returns all presynaptic connections if the position is in the graph
    fn get_incoming_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>>; 
    /// Returns all postsynaptic connections if the position is in the graph
    fn get_outgoing_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>>;
    /// Updates the history of the graph with the current state
    fn update_history(&mut self);
    /// Writes current weights to files prefixed by the `tag` value
    fn write_current_weights(&self, tag: &str);
    /// Writes history of weights to files prefixed by the `tag` value
    fn write_history(&self, tag: &str);
}

fn csv_write<T: Display>(csv_file: &mut BufWriter<File>, grid: &Vec<Vec<Option<T>>>) {
    for row in grid {
        for (n, i) in row.iter().enumerate() {
            let item_to_write = match i {
                Some(value) => format!("{}", value),
                None => String::from("None"),
            };

            if n < row.len() - 1 {
                write!(csv_file, "{},", item_to_write).expect("Could not write to file");
            } else {
                write!(csv_file, "{}", item_to_write).expect("Could not write to file");
            }
        }
        writeln!(csv_file).expect("Could not write to file");
    }
} 

/// A graph implemented as an adjacency matrix where the positions of each node
/// are converted to `usize` to be index in a 2-dimensional matrix
#[derive(Clone, Debug)]
pub struct AdjacencyMatrix {
    /// Converts position to a index for the matrix
    pub position_to_index: HashMap<GraphPosition, usize>,
    /// Converts the index back to a position
    pub index_to_position: HashMap<usize, GraphPosition>,
    /// Matrix of weights
    pub matrix: Vec<Vec<Option<f64>>>,
    /// History of matrix weights
    pub history: Vec<Vec<Vec<Option<f64>>>>,
    /// Identifier
    pub id: usize,
}

fn transform_index_to_position(
    original_map: &HashMap<usize, GraphPosition>
) -> HashMap<usize, (usize, usize, usize)> {
    let mut new_map = HashMap::new();
    
    for (key, value) in original_map {
        let tuple_value = (value.id, value.pos.0, value.pos.1);
        new_map.insert(*key, tuple_value);
    }
    
    new_map
}

impl AdjacencyMatrix {
    pub fn nodes_len(&self) -> usize {
        self.position_to_index.len()
    }
}

impl GraphFunctionality for AdjacencyMatrix {
    fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn add_vertex(&mut self, position: GraphPosition) {
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
        postsynaptic: GraphPosition, 
        connections: Vec<GraphPosition>, 
        weight_params: &Option<GaussianParameters>,
    ) {
        if !self.position_to_index.contains_key(&postsynaptic) {
            self.add_vertex(postsynaptic)
        }
        for i in connections.iter() {
            if !self.position_to_index.contains_key(i) {
                self.add_vertex(*i);
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

    fn get_every_node(&self) -> Vec<GraphPosition> {
        self.position_to_index.keys().cloned().collect()
    }

    fn lookup_weight(&self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition) -> Result<Option<f64>> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Postsynaptic value not in graph"));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Presynaptic value not in graph"));
        }

        Ok(self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]])
    }

    fn edit_weight(&mut self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition, weight: Option<f64>) -> Result<()> {
        if !self.position_to_index.contains_key(postsynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Postsynaptic value not in graph"));
        }
        if !self.position_to_index.contains_key(presynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Presynaptic value not in graph"));
        }
        
        self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]] = weight;

        Ok(())
    }

    fn get_incoming_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>> {
        if !self.position_to_index.contains_key(pos) {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot find position in graph"));
        }

        let mut connections: HashSet<GraphPosition> = HashSet::new();
        for i in self.position_to_index.keys() {
            match self.lookup_weight(i, &pos).unwrap() {
                Some(_) => { connections.insert(*i); },
                None => {}
            };
        }

        Ok(connections)
    }

    fn get_outgoing_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>> {
        if !self.position_to_index.contains_key(pos) {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot find position in graph"));
        }

        let node = self.position_to_index[pos];
        let out_going_connections = self.matrix[node]
            .iter()
            .enumerate()
            .filter(|(_, &val)| val.is_some())
            .map(|(n, _)| self.index_to_position[&n])
            .collect::<HashSet<GraphPosition>>();
            
        Ok(out_going_connections)
    }

    fn update_history(&mut self) {
        self.history.push(self.matrix.clone());
    }

    fn write_current_weights(&self, tag: &str) {
        let serializable_map = transform_index_to_position(&self.index_to_position);

        let json_string = serde_json::to_string(&serializable_map)
                .expect("Failed to convert to JSON");
        let mut json_file = BufWriter::new(File::create(format!("{}_positions.json", tag))
            .expect("Could not create file"));

        write!(json_file, "{}", json_string).expect("Could not create to file");

        let mut csv_file = BufWriter::new(File::create(format!("{}_connections.csv", tag))
            .expect("Could not create file"));

        csv_write(&mut csv_file, &self.matrix);
    }

    fn write_history(&self, tag: &str) {
        let serializable_map = transform_index_to_position(&self.index_to_position);

        let json_string = serde_json::to_string(&serializable_map)
                .expect("Failed to convert to JSON");
        let mut json_file = BufWriter::new(File::create(format!("{}_positions.json", tag))
            .expect("Could not create file"));

        write!(json_file, "{}", json_string).expect("Could not create to file");

        let mut csv_file = BufWriter::new(File::create(format!("{}_connections.csv", tag))
            .expect("Could not create file"));

        for grid in &self.history {
            csv_write(&mut csv_file, &grid);
            writeln!(csv_file, "-----").expect("Could not write to file");
        }
    }
}

impl Default for AdjacencyMatrix {
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
pub struct AdjacencyList {
    /// All presynaptic connections
    pub incoming_connections: HashMap<GraphPosition, HashMap<GraphPosition, f64>>,
    /// All postsynaptic connections
    pub outgoing_connections: HashMap<GraphPosition, HashSet<GraphPosition>>,
    /// History of presynaptic connection weights
    pub history: Vec<HashMap<GraphPosition, HashMap<GraphPosition, f64>>>,
    /// Identifier
    pub id: usize,
}

fn transform_incoming_connections(
    original_map: &HashMap<GraphPosition, HashMap<GraphPosition, f64>>
) -> HashMap<(usize, usize, usize), HashMap<(usize, usize, usize), f64>> {
    let mut new_map: HashMap<(usize, usize, usize), HashMap<(usize, usize, usize), f64>> = HashMap::new();
    
    for (outer_key, inner_map) in original_map {
        let outer_tuple_key = (outer_key.id, outer_key.pos.0, outer_key.pos.1);
        let mut new_inner_map: HashMap<(usize, usize, usize), f64> = HashMap::new();
        
        for (inner_key, &value) in inner_map {
            let inner_tuple_key = (inner_key.id, inner_key.pos.0, inner_key.pos.1);
            new_inner_map.insert(inner_tuple_key, value);
        }
        
        new_map.insert(outer_tuple_key, new_inner_map);
    }
    
    new_map
}

type KeyWeightPair = HashMap<String, f64>;

impl GraphFunctionality for AdjacencyList {
    fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn add_vertex(&mut self, position: GraphPosition) {
        if self.incoming_connections.contains_key(&position) {
            return;
        }

        self.incoming_connections.entry(position)
            .or_insert_with(HashMap::new);
    }

    fn initialize_connections(
        &mut self, 
        postsynaptic: GraphPosition, 
        connections: Vec<GraphPosition>, 
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

    fn get_every_node(&self) -> Vec<GraphPosition> {
        self.incoming_connections.keys().cloned().collect()
    }

    fn lookup_weight(&self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition) -> Result<Option<f64>> {
        // println!("{:#?} {:#?}", presynaptic, postsynaptic);

        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Postsynaptic value not in graph"));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Presynaptic value not in graph"));
        }

        Ok(self.incoming_connections[postsynaptic].get(presynaptic).copied())
    }

    fn edit_weight(&mut self, presynaptic: &GraphPosition, postsynaptic: &GraphPosition, weight: Option<f64>) -> Result<()> {
        // self.incoming_connections[presynaptic][postsynaptic] = weight;

        if !self.incoming_connections.contains_key(postsynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Postsynaptic value not in graph"));
        }
        if !self.incoming_connections.contains_key(presynaptic) {
            return Err(Error::new(ErrorKind::InvalidInput, "Presynaptic value not in graph"));
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
    fn get_incoming_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>> {
        if !self.incoming_connections.contains_key(pos) {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot find position in graph"));
        }

        Ok(self.incoming_connections[pos].keys().cloned().collect::<HashSet<GraphPosition>>())
    }

    // to be cached
    // or point to reference
    fn get_outgoing_connections(&self, pos: &GraphPosition) -> Result<HashSet<GraphPosition>> {
        if !self.incoming_connections.contains_key(pos) {
            return Err(Error::new(ErrorKind::InvalidInput, "Cannot find position in graph"));
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

    fn write_current_weights(&self, tag: &str) {
        let serialiable_map = transform_incoming_connections(&self.incoming_connections);

        let json_string = serde_json::to_string(&serialiable_map)
                .expect("Failed to convert to JSON");
        let mut json_file = BufWriter::new(File::create(format!("{}_incoming_connections.json", tag))
            .expect("Could not create file"));

        write!(json_file, "{}", json_string).expect("Could not create to file");
    }

    fn write_history(&self, tag: &str) {
        let mut history_json: HashMap<String, HashMap<String, KeyWeightPair>> = HashMap::new();

        for (key, value) in self.history.iter().enumerate() {
            // let wrapped_key = PositionWrapper(key);
            let mut inner_map: HashMap<String, KeyWeightPair> = HashMap::new();

            for (inner_key, inner_value) in value {
                let wrapped_inner_key = format!(
                    "{}_{}_{}", inner_key.id, inner_key.pos.0, inner_key.pos.1
                );
                let mut innermost_map: KeyWeightPair = HashMap::new();

                for (innermost_key, innermost_value) in inner_value {
                    let wrapped_innermost_key = format!(
                        "{}_{}_{}", innermost_key.id, innermost_key.pos.0, innermost_key.pos.1
                    );
                    innermost_map.insert(wrapped_innermost_key, *innermost_value);
                }

                inner_map.insert(wrapped_inner_key, innermost_map);
            }

            history_json.insert(key.to_string(), inner_map);
        }

        let json_string = serde_json::to_string_pretty(&history_json)
            .expect("Failed to convert to JSON");
        let mut json_file = BufWriter::new(File::create(format!("{}_history.json", tag))
            .expect("Could not create file"));

        write!(json_file, "{}", json_string).expect("Could not create to file");
    }
}

impl Default for AdjacencyList {
    fn default() -> Self {
        AdjacencyList { 
            incoming_connections: HashMap::new(), 
            outgoing_connections: HashMap::new(), 
            history: vec![], 
            id: 0,
        }
    }
}
