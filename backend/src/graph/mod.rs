use std::{
    collections::HashMap, 
    fs::File, 
    io::{Write, BufWriter, Result, Error, ErrorKind}, 
    fmt::Display,
};
use serde_json;
#[path = "../distribution/mod.rs"]
mod distribution;
use distribution::limited_distr;
use crate::neuron::BayesianParameters;


pub type Position = (usize, usize);

pub trait GraphFunctionality {
    fn initialize_connections(
        &mut self, 
        postsynaptic: Position, 
        connections: Vec<Position>, 
        weight_params: &Option<BayesianParameters>,
    );
    fn get_every_node(&self) -> Vec<Position>;
    fn lookup_weight(&self, presynaptic: &Position, postsynaptic: &Position) -> Option<f64>; 
    fn edit_weight(&mut self, presynaptic: &Position, postsynaptic: &Position, weight: Option<f64>);
    fn get_incoming_connections(&self, pos: &Position) -> Vec<Position>; 
    fn get_outgoing_connections(&self, pos: &Position) -> Vec<Position>;
    fn update_history(&mut self);
    fn write_current_weights(&self, tag: &str);
    fn write_history(&self, tag: &str);
}

// pub trait GraphFunctionalitySynced: Sync {
//     fn get_every_node(&self) -> Vec<Position>;
//     fn lookup_weight(&self, presynaptic: &Position, postsynaptic: &Position) -> Option<f64>; 
//     fn get_incoming_connections(&self, pos: &Position) -> Vec<Position>; 
//     fn get_outgoing_connections(&self, pos: &Position) -> Vec<Position>;
// }

#[derive(Clone, Debug)]
pub struct AdjacencyMatrix {
    pub position_to_index: HashMap<Position, usize>,
    pub index_to_position: HashMap<usize, Position>,
    pub matrix: Vec<Vec<Option<f64>>>,
    pub history: Vec<Vec<Vec<Option<f64>>>>,
}

impl AdjacencyMatrix {
    fn nodes_len(&self) -> usize {
        self.position_to_index.len()
    }

    fn add_vertex(&mut self, position: Position) {
        let index = self.nodes_len();
    
        self.position_to_index.insert(position, index);
        self.index_to_position.insert(index, position);

        self.matrix.push(vec![None; index]);
        for row in self.matrix.iter_mut() {
            row.push(None);
        }
    }
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

impl GraphFunctionality for AdjacencyMatrix {
    fn initialize_connections(
        &mut self, 
        postsynaptic: Position, 
        connections: Vec<Position>, 
        weight_params: &Option<BayesianParameters>,
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
                        limited_distr(
                            value.mean, 
                            value.std, 
                            value.min, 
                            value.max,
                        )
                    )
                },
                None => Some(1.0),
            };

            self.edit_weight(i, &postsynaptic, weight);
        }
    }

    fn get_every_node(&self) -> Vec<Position> {
        self.position_to_index.keys().cloned().collect()
    }

    fn lookup_weight(&self, presynaptic: &Position, postsynaptic: &Position) -> Option<f64> {
        self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]]
    }

    fn edit_weight(&mut self, presynaptic: &Position, postsynaptic: &Position, weight: Option<f64>) {
        self.matrix[self.position_to_index[presynaptic]][self.position_to_index[postsynaptic]] = weight;
    }

    // to be cached
    fn get_incoming_connections(&self, pos: &Position) -> Vec<Position> {
        let mut connections: Vec<Position> = Vec::new();
        for i in self.position_to_index.keys() {
            match self.lookup_weight(i, &pos) {
                Some(_) => { connections.push(*i); },
                None => {}
            };
        }

        return connections;
    }

    // #[cache]
    // fn cached_get_incoming_connections(&self, pos: &Position) -> Vec<Position> {
    //     let mut connections: Vec<Position> = Vec::new();
    //     for i in self.position_to_index.keys() {
    //         match self.lookup_weight(i, &pos) {
    //             Some(_) => { connections.push(*i); },
    //             None => {}
    //         };
    //     }

    //     return connections;
    // }

    // to be cached
    fn get_outgoing_connections(&self, pos: &Position) -> Vec<Position> {
        let node = self.position_to_index[pos];
        let out_going_connections = self.matrix[node]
            .iter()
            .enumerate()
            .filter(|(_, &val)| val.is_some())
            .map(|(n, _)| self.index_to_position[&n])
            .collect::<Vec<Position>>();
            
        return out_going_connections;
    }

    fn update_history(&mut self) {
        self.history.push(self.matrix.clone());
    }

    fn write_current_weights(&self, tag: &str) {
        let json_string = serde_json::to_string(&self.index_to_position)
                .expect("Failed to convert to JSON");
        let mut json_file = BufWriter::new(File::create(format!("{}_positions.json", tag))
            .expect("Could not create file"));

        write!(json_file, "{}", json_string).expect("Could not create to file");

        let mut csv_file = BufWriter::new(File::create(format!("{}_connections.csv", tag))
            .expect("Could not create file"));

        csv_write(&mut csv_file, &self.matrix);
    }

    fn write_history(&self, tag: &str) {
        let json_string = serde_json::to_string(&self.index_to_position)
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdjacencyList {
    pub incoming_connections: HashMap<Position, HashMap<Position, Option<f64>>>,
    pub outgoing_connections: HashMap<Position, Vec<Position>>,
    pub history: Vec<HashMap<Position, HashMap<Position, Option<f64>>>>,
}

type KeyWeightPair = HashMap<String, Option<f64>>;

impl GraphFunctionality for AdjacencyList {
    fn initialize_connections(
        &mut self, 
        postsynaptic: Position, 
        connections: Vec<Position>, 
        weight_params: &Option<BayesianParameters>,
    ) {
        for i in connections.iter() {
            let weight = match weight_params {
                Some(value) => {
                    Some(
                        limited_distr(
                            value.mean, 
                            value.std, 
                            value.min, 
                            value.max,
                        )
                    )
                },
                None => Some(1.0),
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
                vector.push(postsynaptic);
            } else {
                self.outgoing_connections.insert(*i, vec![postsynaptic]);
            }
        }
    }

    fn get_every_node(&self) -> Vec<Position> {
        self.incoming_connections.keys().cloned().collect()
    }

    fn lookup_weight(&self, presynaptic: &Position, postsynaptic: &Position) -> Option<f64> {
        // println!("{:#?} {:#?}", presynaptic, postsynaptic);
        self.incoming_connections[postsynaptic][presynaptic]
    }

    fn edit_weight(&mut self, presynaptic: &Position, postsynaptic: &Position, weight: Option<f64>) {
        // self.incoming_connections[presynaptic][postsynaptic] = weight;

        if let Some(positions_and_weights) = self.incoming_connections.get_mut(postsynaptic) {
            positions_and_weights.insert(*presynaptic, weight);
        }
    }

    // to be cached
    // or point to reference
    fn get_incoming_connections(&self, pos: &Position) -> Vec<Position> {
        self.incoming_connections[pos].keys().cloned().collect::<Vec<Position>>()
    }

    // to be cached
    // or point to reference
    fn get_outgoing_connections(&self, pos: &Position) -> Vec<Position> {
        // self.outgoing_connections[pos].clone()
        self.outgoing_connections.get(pos).unwrap_or(&vec![]).clone()
    }

    fn update_history(&mut self) {
        self.history.push(self.incoming_connections.clone());
    }

    fn write_current_weights(&self, tag: &str) {
        let json_string = serde_json::to_string(&self.incoming_connections)
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
                let wrapped_inner_key = format!("{}_{}", inner_key.0, inner_key.1);
                let mut innermost_map: KeyWeightPair = HashMap::new();

                for (innermost_key, innermost_value) in inner_value {
                    let wrapped_innermost_key = format!("{}_{}", innermost_key.0, innermost_key.1);
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
        }
    }
}

#[derive(Clone, Debug)]
pub enum Graph {
    Matrix,
    List,
}

impl Graph {
    pub fn from_str(string: &str) -> Result<Graph> {
        match string.to_ascii_lowercase().as_str() {
            "matrix" | "adjacency matrix" => Ok(Graph::Matrix),
            "list" | "adjacency list" => Ok(Graph::List),
            _ => { Err(Error::new(ErrorKind::InvalidInput, "Unknown graph type")) }
        }
    }
}

#[derive(Clone)]
pub struct GraphParameters {
    pub write_weights: bool,
    pub write_history: bool,
}
