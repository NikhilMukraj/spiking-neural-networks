use std::{
    collections::HashMap, 
    fs::{File, read_to_string}, 
    io::{Write, BufWriter, Result, Error, ErrorKind}, 
    env,
};
use rand::{Rng, seq::SliceRandom};
use exprtk_rs::{Expression, SymbolTable};
use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, IntoPyDict};
#[path = "distribution/mod.rs"]
mod distribution;
use crate::distribution::limited_distr;
mod neuron;
use crate::neuron::{
    IFParameters, IFType, PotentiationType, Cell, CellGrid,
    ScaledDefault, IzhikevichDefault, BayesianParameters, STDPParameters,
    Gate, HodgkinHuxleyCell
};
mod eeg;
use crate::eeg::{read_eeg_csv, get_power_density, power_density_comparison};
mod ga;
use crate::ga::{BitString, decode, genetic_algo};
mod graph;
use crate::graph::{Position, AdjacencyList, AdjacencyMatrix, Graph, GraphParameters, GraphFunctionality};
mod py_interface;
use crate::py_interface::{IFCell, HodgkinHuxleyModel};


fn positions_within_square(
    center_row: usize, 
    center_col: usize, 
    extent: usize, 
    size: Position
) -> Vec<Position> {
    let (row_length, col_length) = size;
    let mut positions = Vec::new();

    for row in center_row.saturating_sub(extent)..=(center_row + extent) {
        for col in center_col.saturating_sub(extent)..=(center_col + extent) {
            if (row != center_row || col != center_col) && (row < row_length && col < col_length) {
                positions.push((row, col));
            }
        }
    }

    positions
}

fn randomly_select_positions(mut positions: Vec<Position>, num_to_select: usize) -> Vec<Position> {
    let mut rng = rand::thread_rng();

    positions.shuffle(&mut rng);
    positions.truncate(num_to_select);

    positions
}

fn get_input_from_positions(
    cell_grid: &CellGrid, 
    input_positions: &Vec<Position>, 
    input_calculation: &mut dyn FnMut(f64, f64, f64, f64) -> f64,
    if_params: Option<&IFParameters>,
    averaged: bool,
) -> f64 {
    let mut input_val = input_positions
        .iter()
        .map(|input_position| {
            let (pos_x, pos_y) = input_position;
            let input_cell = &cell_grid[*pos_x][*pos_y];
            
            let sign = match input_cell.potentiation_type { 
                PotentiationType::Excitatory => -1., 
                PotentiationType::Inhibitory => 1.,
            };

            let final_input = input_calculation(
                sign,
                input_cell.current_voltage,
                input_cell.receptor_density,
                input_cell.neurotransmission_concentration,
            );
            
            final_input

        })
        .sum();

    match if_params {
        Some(params) => { 
            input_val *= limited_distr(
                params.bayesian_params.mean, 
                params.bayesian_params.std, 
                params.bayesian_params.min, 
                params.bayesian_params.max,
            ); 
        },
        None => {},
    }

    if averaged {
        input_val /= input_positions.len() as f64;
    }

    return input_val;
}

fn weighted_get_input_from_positions(
    cell_grid: &CellGrid, 
    graph: &dyn GraphFunctionality,
    position: &Position,
    input_positions: &Vec<(usize, usize)>, 
    input_calculation: &mut dyn FnMut(f64, f64, f64, f64) -> f64,
    if_params: Option<&IFParameters>,
    averaged: bool,
) -> f64 {
    let mut input_val = input_positions
        .iter()
        .map(|input_position| {
            let (pos_x, pos_y) = input_position;
            let input_cell = &cell_grid[*pos_x][*pos_y];
            
            let sign = match input_cell.potentiation_type { 
                PotentiationType::Excitatory => -1., 
                PotentiationType::Inhibitory => 1.,
            };

            let final_input = input_calculation(
                sign,
                input_cell.current_voltage,
                input_cell.receptor_density,
                input_cell.neurotransmission_concentration,
            );

            // do not account for neurotransmission just yet
            // let final_input = sign * current_voltage;
            
            final_input * graph.lookup_weight(&input_position, position).unwrap()

        })
        .sum();

    match if_params {
        Some(params) => { 
            input_val *= limited_distr(
                params.bayesian_params.mean, 
                params.bayesian_params.std, 
                params.bayesian_params.min, 
                params.bayesian_params.max,
            ); 
        },
        None => {},
    }

    if averaged {
        input_val /= input_positions.len() as f64;
    }

    return input_val;
}

fn get_volt_avg(cell_grid: &CellGrid) -> f64 {
    let volt_mean: f64 = cell_grid
        .iter()
        .flatten()
        .map(|x| x.current_voltage)
        .sum();

    volt_mean / ((cell_grid[0].len() * cell_grid.len()) as f64)
}

fn get_neuro_avg(cell_grid: &CellGrid) -> f64 {
    let neuro_mean: f64 = cell_grid
        .iter()
        .flatten()
        .map(|x| x.neurotransmission_concentration)
        .sum();

    neuro_mean / ((cell_grid[0].len() * cell_grid.len()) as f64) 
}

struct NeuroAndVolts {
    voltage: f64,
    neurotransmitter: f64,
}

enum Output {
    Grid(Vec<CellGrid>),
    GridBinary(Vec<CellGrid>),
    Averaged(Vec<NeuroAndVolts>),
    AveragedBinary(Vec<NeuroAndVolts>),
}

impl Output {
    fn add(&mut self, cell_grid: &CellGrid) {
        match self {
            Output::Grid(grids) | Output::GridBinary(grids) => { grids.push(cell_grid.clone()) }
            Output::Averaged(averages) | Output::AveragedBinary(averages) => { 
                averages.push(
                    NeuroAndVolts {
                        voltage: get_volt_avg(cell_grid),
                        neurotransmitter: get_neuro_avg(cell_grid),
                    }
                );
            }
        }
    }

    fn from_str(string: &str) -> Result<Output> {
        match string.to_ascii_lowercase().as_str() {
            "grid" => { Ok(Output::Grid(Vec::<CellGrid>::new())) },
            "grid binary" => { Ok(Output::GridBinary(Vec::<CellGrid>::new())) },
            "averaged" => { Ok(Output::Averaged(Vec::<NeuroAndVolts>::new())) },
            "averaged binary" => { Ok(Output::AveragedBinary(Vec::<NeuroAndVolts>::new())) },
            _ => { Err(Error::new(ErrorKind::InvalidInput, "Unknown output type")) }
        }
    }

    fn write_to_file(&self, voltage_file: &mut BufWriter<File>, neurotransmitter_file: &mut BufWriter<File>) {
        match &self {
            Output::Grid(grids) => {
                for grid in grids {
                    for row in grid {
                        for value in row {
                            write!(voltage_file, "{} ", value.current_voltage)
                                .expect("Could not write to file");
                            write!(neurotransmitter_file, "{} ", value.neurotransmission_concentration)
                                .expect("Could not write to file");
                        }
                        writeln!(voltage_file)
                            .expect("Could not write to file");
                        writeln!(neurotransmitter_file)
                            .expect("Could not write to file");
                    }
                    writeln!(voltage_file, "-----")
                        .expect("Could not write to file"); 
                    writeln!(neurotransmitter_file, "-----")
                        .expect("Could not write to file"); 
                }
            },
            Output::GridBinary(grids) => {
                for grid in grids {
                    for row in grid {
                        for value in row {
                            let bytes = value.current_voltage.to_le_bytes();
                            voltage_file
                                .write_all(&bytes)
                                .expect("Could not write to file");
                
                            let bytes = value.neurotransmission_concentration.to_le_bytes();
                            neurotransmitter_file
                                .write_all(&bytes)
                                .expect("Could not write to file");
                        }
                    }
                }
            },
            Output::Averaged(averages) => {
                for neuro_and_volt in averages {
                    writeln!(voltage_file, "{}", neuro_and_volt.voltage)
                        .expect("Could not write to file");
                    writeln!(neurotransmitter_file, "{}", neuro_and_volt.neurotransmitter)
                        .expect("Could not write to file");
                } 
            },
            Output::AveragedBinary(averages) => {
                for neuro_and_volt in averages {
                    let volt_mean_bytes = neuro_and_volt.voltage.to_le_bytes();
                    let neuro_mean_bytes = neuro_and_volt.neurotransmitter.to_le_bytes();

                    voltage_file.write_all(&volt_mean_bytes).expect("Could not write to file"); 
                    neurotransmitter_file.write_all(&neuro_mean_bytes).expect("Could not write to file");
                }
            }
        }
    }
}

// type AdaptiveDwAndGetSpikeFunction = Box::<dyn Fn(&mut Cell, &IFParameters) -> bool>;
// type AdaptiveDvFunction = Box<dyn Fn(&mut Cell, &IFParameters, f64) -> f64>;

// // use in run simulation and isolated stdp test
// fn determine_calculaton_function(if_type: IFType) -> Result<(AdaptiveDwAndGetSpikeFunction, AdaptiveDvFunction)> {
//     let regular_adaptive_dw_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool
//         {neuron.apply_dw_change_and_get_spike(if_params)};
//     let izhikevich_adaptive_dw_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool 
//         {neuron.izhikevich_apply_dw_and_get_spike(if_params)};
//     let leaky_izhikevich_adaptive_dw_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool 
//         {neuron.izhikevich_apply_dw_and_get_spike(if_params)};

//     let adaptive_apply_and_get_spike = match if_type {
//         IFType::Basic => return Err(Error::new(ErrorKind::InvalidData, "Non adaptive IF type")), 
//         IFType::Adaptive | IFType::AdaptiveExponential => regular_adaptive_dw_and_get_spike,
//         IFType::Izhikevich => izhikevich_adaptive_dw_and_get_spike,
//         IFType::IzhikevichLeaky => leaky_izhikevich_adaptive_dw_and_get_spike,
//     };

//     let adaptive_apply_and_get_spike = Box::new(adaptive_apply_and_get_spike);

//     let regular_adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64
//         {neuron.adaptive_get_dv_change(if_params, input_value)};
//     let exp_adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 
//         {neuron.exp_adaptive_get_dv_change(if_params, input_value)};
//     let izhikevich_adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 
//         {neuron.izhikevich_get_dv_change(if_params, input_value)};
//     let leaky_izhikevich_adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 
//         {neuron.izhikevich_leaky_get_dv_change(if_params, input_value)};

//     let adaptive_dv = match if_type {
//         IFType::Basic => unreachable!(), 
//         IFType::Adaptive => regular_adaptive_dv,
//         IFType::AdaptiveExponential => exp_adaptive_dv,
//         IFType::Izhikevich => izhikevich_adaptive_dv,
//         IFType::IzhikevichLeaky => leaky_izhikevich_adaptive_dv,
//     };

//     let adaptive_dv = Box::new(adaptive_dv);

//     return Ok((adaptive_apply_and_get_spike, adaptive_dv));
// }

type IFCellGrid = Vec<Vec<IFCell>>;

#[pyfunction]
#[pyo3(signature = (
    num_rows,
    num_cols,
    if_mode,
    dt,
    v_init,
    excitatory_chance=0.8,
    neurotransmission_release=0.0,
    neurotransmission_release_std=0.0,
    receptor_density=0.0,
    receptor_density_std=0.0,
    chance_of_releasing=0.0, 
    dissipation_rate=0.0, 
    dissipation_rate_std=0.0, 
    chance_of_random_release=0.0,
    random_release_concentration=0.0,
    random_release_concentration_std=0.0,
    w_init=30.0,
    alpha_init=0.02,
    beta_init=0.2,
    v_reset=-65.0,
    d_init=8.0,
    a_minus=2.0,
    a_plus=2.0,
    tau_minus=45.0,
    tau_plus=45.0,
    stdp_weight_mean=1.75,
    stdp_weight_std=0.0,
    stdp_weight_max=1.75,
    stdp_weight_min=5.25,
))]
fn create_cell_grid(
    num_rows: usize,
    num_cols: usize,
    if_mode: IFType,
    dt: f64,
    v_init: f64,
    excitatory_chance: f64,
    neurotransmission_release: f64,
    neurotransmission_release_std: f64,
    receptor_density: f64,
    receptor_density_std: f64,
    chance_of_releasing: f64, 
    dissipation_rate: f64, 
    dissipation_rate_std: f64, 
    chance_of_random_release: f64,
    random_release_concentration: f64,
    random_release_concentration_std: f64,
    w_init: f64,
    alpha_init: f64,
    beta_init: f64,
    v_reset: f64,
    d_init: f64,
    a_minus: f64,
    a_plus: f64,
    tau_minus: f64,
    tau_plus: f64,
    stdp_weight_mean: f64,
    stdp_weight_std: f64,
    stdp_weight_max: f64,
    stdp_weight_min: f64,
) -> IFCellGrid {
    let stdp_params = STDPParameters {
        a_minus: a_minus,
        a_plus: a_plus,
        tau_minus: tau_minus,
        tau_plus: tau_plus,
        weight_bayesian_params: BayesianParameters {
            mean: stdp_weight_mean,
            std: stdp_weight_std,
            max: stdp_weight_max,
            min: stdp_weight_min,
        }
    };

    let cell_grid = (0..num_rows)
        .map(|_| {
            (0..num_cols)
                .map(|_| Cell { 
                    current_voltage: v_init, 
                    refractory_count: 0.0,
                    leak_constant: -1.,
                    integration_constant: 1.,
                    potentiation_type: PotentiationType::weighted_random_type(excitatory_chance),
                    neurotransmission_concentration: 0., 
                    neurotransmission_release: limited_distr(neurotransmission_release, neurotransmission_release_std, 0.0, 1.0),
                    receptor_density: limited_distr(receptor_density, receptor_density_std, 0.0, 1.0),
                    chance_of_releasing: chance_of_releasing, 
                    dissipation_rate: limited_distr(dissipation_rate, dissipation_rate_std, 0.0, 1.0), 
                    chance_of_random_release: chance_of_random_release,
                    random_release_concentration: limited_distr(random_release_concentration, random_release_concentration_std, 0.0, 1.0),
                    w_value: w_init,
                    stdp_params: stdp_params.clone(),
                    last_firing_time: None,
                    alpha: alpha_init,
                    beta: beta_init,
                    c: v_reset,
                    d: d_init,
                })
                .collect::<Vec<Cell>>()
        })
        .collect::<CellGrid>();

    (0..num_rows)
        .map(|x| {
            (0..num_cols)
                .map(|y| {
                    match if_mode {
                        IFType::Basic | IFType::Adaptive | IFType::AdaptiveExponential => {
                            IFCell {
                                mode: if_mode.clone(),
                                cell_backend: cell_grid[x][y].clone(),
                                if_params: IFParameters {
                                    dt: dt,
                                    v_init: v_init,
                                    w_init: w_init,
                                    alpha_init: alpha_init,
                                    beta_init: beta_init,
                                    v_reset: v_reset,
                                    d_init: d_init,
                                    ..Default::default()
                                }
                            }
                        },
                        IFType::Izhikevich | IFType::IzhikevichLeaky => {
                            IFCell {
                                mode: if_mode.clone(),
                                cell_backend: cell_grid[x][y].clone(),
                                if_params: IFParameters {
                                    dt: dt,
                                    v_init: v_init,
                                    w_init: w_init,
                                    alpha_init: alpha_init,
                                    beta_init: beta_init,
                                    v_reset: v_reset,
                                    d_init: d_init,
                                    ..IzhikevichDefault::izhikevich_default()
                                }
                            }
                        }
                    }
                })
                .collect::<Vec<IFCell>>()
        })
        .collect::<IFCellGrid>()
}

fn generate_graph_from_connections(
    incoming_connections: PyDict,
    outgoing_connections: PyDict,
    py: Python
) -> Result<AdjacencyList> {
    let converted_incoming_connections: HashMap<Position, HashMap<Position, Option<f64>>> = 
        incoming_connections.extract()?;

    let converted_outgoing_connections: HashMap<Position, Vec<Position>> = 
        outgoing_connections.extract()?;

    Ok(
        AdjacencyList {
            incoming_connections: converted_incoming_connections,
            outgoing_connections: converted_outgoing_connections,
            history: vec![],
        }
    )
}

// maybe convert connections to matrix
// fn generate_graph_from_matrix(
   
// ) -> Result<AdjacencyMatrix> {
//     Ok(
//         AdjacencyMatrix {

//         }
//     )
// }

fn run_simulation(
    num_rows: usize, 
    num_cols: usize, 
    iterations: usize, 
    radius: usize, 
    random_volt_initialization: bool,
    averaged: bool,
    if_type: IFType,
    if_params: &IFParameters,
    do_stdp: bool,
    graph_params: &GraphParameters,
    stdp_params: &STDPParameters,
    default_cell_values: &HashMap<String, f64>,
    input_calculation: &mut dyn FnMut(f64, f64, f64, f64) -> f64,
    mut output_val: Output,
) -> Result<(Output, Box<dyn GraphFunctionality>)> {
    if radius / 2 > num_rows || radius / 2 > num_cols || radius == 0 {
        let err_msg = "Radius must be less than both number of rows or number of cols divided by 2 and greater than 0";
        return Err(Error::new(ErrorKind::InvalidInput, err_msg));
    }

    let neurotransmission_release = *default_cell_values.get("neurotransmission_release")
        .unwrap_or(&1.);
    let receptor_density = *default_cell_values.get("receptor_density")
        .unwrap_or(&1.);
    let chance_of_releasing = *default_cell_values.get("chance_of_releasing")
        .unwrap_or(&0.5);
    let dissipation_rate = *default_cell_values.get("dissipation_rate")
        .unwrap_or(&0.1);
    let chance_of_random_release = *default_cell_values.get("chance_of_random_release")
        .unwrap_or(&0.2);
    let random_release_concentration = *default_cell_values.get("random_release_concentration")
        .unwrap_or(&0.1);    
    let excitatory_chance = *default_cell_values.get("excitatory_chance")
        .unwrap_or(&0.5);

    let neurotransmission_release_std = *default_cell_values.get("neurotransmission_release_std")
        .unwrap_or(&0.);
    let receptor_density_std = *default_cell_values.get("receptor_density_std")
        .unwrap_or(&0.);
    let dissipation_rate_std = *default_cell_values.get("dissipation_rate_std")
        .unwrap_or(&0.);
    let random_release_concentration_std = *default_cell_values.get("random_release_concentration_std")
        .unwrap_or(&0.);

    let mean_change = &if_params.bayesian_params.mean != &BayesianParameters::default().mean;
    let std_change = &if_params.bayesian_params.std != &BayesianParameters::default().std;
    let bayesian = if mean_change || std_change {
        Some(if_params)
    } else {
        None
    };

    let mut cell_grid: CellGrid = (0..num_rows)
        .map(|_| {
            (0..num_cols)
                .map(|_| Cell { 
                    current_voltage: if_params.v_init, 
                    refractory_count: 0.0,
                    leak_constant: -1.,
                    integration_constant: 1.,
                    potentiation_type: PotentiationType::weighted_random_type(excitatory_chance),
                    neurotransmission_concentration: 0., 
                    neurotransmission_release: limited_distr(neurotransmission_release, neurotransmission_release_std, 0.0, 1.0),
                    receptor_density: limited_distr(receptor_density, receptor_density_std, 0.0, 1.0),
                    chance_of_releasing: chance_of_releasing, 
                    dissipation_rate: limited_distr(dissipation_rate, dissipation_rate_std, 0.0, 1.0), 
                    chance_of_random_release: chance_of_random_release,
                    random_release_concentration: limited_distr(random_release_concentration, random_release_concentration_std, 0.0, 1.0),
                    w_value: if_params.w_init,
                    stdp_params: stdp_params.clone(),
                    last_firing_time: None,
                    alpha: if_params.alpha_init,
                    beta: if_params.beta_init,
                    c: if_params.v_reset,
                    d: if_params.d_init,
                })
                .collect::<Vec<Cell>>()
        })
        .collect::<CellGrid>();

    let mut rng = rand::thread_rng();

    if random_volt_initialization {
        for section in &mut cell_grid {
            for neuron in section {
                neuron.current_voltage = rng.gen_range(if_params.v_init..=if_params.v_th);
                neuron.refractory_count = rng.gen_range(0.0..=if_params.tref);
            }
        }
    }

    let mut graph: Box<dyn GraphFunctionality> = match graph_params.graph_type {
        Graph::Matrix => {
            let matrix = AdjacencyMatrix::default();
            Box::new(matrix)
        },
        Graph::List => {
            let list = AdjacencyList::default();
            Box::new(list)
        },
    };
    
    for row in 0..num_rows {
        for col in 0..num_cols {
            let positions = positions_within_square(row, col, radius, (num_rows, num_cols));
            let num_to_select = rng.gen_range(1..positions.len());
            let positions = randomly_select_positions(positions, num_to_select);

            graph.initialize_connections((row, col), positions, do_stdp, stdp_params);
        }
    }

    if do_stdp && graph_params.write_history {
        graph.update_history();
    }

    match if_type {
        IFType::Basic => {
            for timestep in 0..iterations {
                let mut changes: HashMap<Position, (f64, bool)> = graph.get_every_node()
                    .iter()
                    .map(|key| (*key, (0.0, false)))
                    .collect();          

                // loop through every cell
                // calculate the dv given the inputs
                // write 
                // end loop

                for pos in graph.get_every_node() {
                    let (x, y) = pos;

                    let input_positions = graph.get_incoming_connections(&pos);

                    let input = if do_stdp {
                        weighted_get_input_from_positions(
                            &cell_grid,
                            &*graph,
                            &pos,
                            &input_positions,
                            input_calculation,
                            bayesian,
                            averaged,
                        )
                    } else {
                        get_input_from_positions(
                            &cell_grid, 
                            &input_positions, 
                            input_calculation, 
                            bayesian,
                            averaged,
                        )
                    };
                    
                    let (dv, is_spiking) = cell_grid[x][y].get_dv_change_and_spike(if_params, input);

                    changes.insert(pos, (dv, is_spiking));
                }

                // loop through every cell
                // modify the voltage
                // end loop

                for (pos, (dv_value, is_spiking_value)) in changes {
                    let (x, y) = pos;
                    
                    cell_grid[x][y].determine_neurotransmitter_concentration(is_spiking_value);
                    cell_grid[x][y].current_voltage += dv_value;

                    if do_stdp && is_spiking_value {
                        cell_grid[x][y].last_firing_time = Some(timestep);

                        let input_positions = graph.get_incoming_connections(&pos);
                        for i in input_positions {
                            let (x_in, y_in) = i;
                            let current_weight = graph.lookup_weight(&(x_in, y_in), &pos).unwrap();
                                                        
                            graph.edit_weight(
                                &(x_in, y_in), 
                                &pos, 
                                Some(current_weight + update_weight(&cell_grid[x_in][y_in], &cell_grid[x][y]))
                            );
                        }

                        let out_going_connections = graph.get_outgoing_connections(&pos);

                        for i in out_going_connections {
                            let (x_out, y_out) = i;
                            let current_weight = graph.lookup_weight(&pos, &(x_out, y_out)).unwrap();

                            graph.edit_weight(
                                &pos, 
                                &(x_out, y_out), 
                                Some(current_weight + update_weight(&cell_grid[x][y], &cell_grid[x_out][y_out]))
                            ); 
                        }
                    } // need to also update neurons on receiving end of spiking neuron
                    // create hashmap of what neurons existing neurons point to and use that
                    // generate that hashmap alongside current adjancency list
                }
                // repeat until simulation is over

                output_val.add(&cell_grid);

                if do_stdp && graph_params.write_history {
                    graph.update_history();
                }
            }
        },
        IFType::Adaptive | IFType::AdaptiveExponential | 
        IFType::Izhikevich | IFType::IzhikevichLeaky => {
            let adaptive_apply_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool {
                match if_type {
                    IFType::Basic => unreachable!(),
                    IFType::Adaptive | IFType::AdaptiveExponential => neuron.apply_dw_change_and_get_spike(if_params),
                    IFType::Izhikevich => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                    IFType::IzhikevichLeaky => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                }
            };

            let adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 {
                match if_type {
                    IFType::Basic => unreachable!(), 
                    IFType::Adaptive => neuron.adaptive_get_dv_change(if_params, input_value),
                    IFType::AdaptiveExponential => neuron.exp_adaptive_get_dv_change(if_params, input_value),
                    IFType::Izhikevich => neuron.izhikevich_get_dv_change(if_params, input_value),
                    IFType::IzhikevichLeaky => neuron.izhikevich_leaky_get_dv_change(if_params, input_value),
                }
            };

            for timestep in 0..iterations {
                let mut changes: HashMap<Position, (f64, bool)> = graph.get_every_node()
                    .iter()
                    .map(|key| (*key, (0.0, false)))
                    .collect();

                // loop through every cell
                // calculate the dv given the inputs
                // write 
                // end loop

                for pos in graph.get_every_node() {
                    let (x, y) = pos;

                    let input_positions = graph.get_incoming_connections(&pos);

                    let input = if do_stdp {
                        weighted_get_input_from_positions(
                            &cell_grid,
                            &*graph,
                            &pos,
                            &input_positions,
                            input_calculation,
                            bayesian,
                            averaged,
                        )
                    } else {
                        get_input_from_positions(
                            &cell_grid, 
                            &input_positions, 
                            input_calculation, 
                            bayesian,
                            averaged,
                        )
                    };

                    let is_spiking = adaptive_apply_and_get_spike(&mut cell_grid[x][y], if_params);

                    changes.insert(pos, (input, is_spiking));
                }

                // find dv change and apply it
                // find neurotransmitter change and apply it
                for (pos, (input_value, is_spiking_value)) in changes {
                    let (x, y) = pos;

                    let dv = adaptive_dv(&mut cell_grid[x][y], if_params, input_value);

                    cell_grid[x][y].determine_neurotransmitter_concentration(is_spiking_value);
                    cell_grid[x][y].current_voltage += dv;

                    if do_stdp && is_spiking_value {
                        cell_grid[x][y].last_firing_time = Some(timestep);

                        let input_positions = graph.get_incoming_connections(&pos);
                        for i in input_positions {
                            let (x_in, y_in) = i;
                            let current_weight = graph.lookup_weight(&(x_in, y_in), &pos).unwrap();
                                                        
                            graph.edit_weight(
                                &(x_in, y_in), 
                                &pos, 
                                Some(current_weight + update_weight(&cell_grid[x_in][y_in], &cell_grid[x][y]))
                            );

                        }

                        let out_going_connections = graph.get_outgoing_connections(&pos);

                        for i in out_going_connections {
                            let (x_out, y_out) = i;
                            let current_weight = graph.lookup_weight(&pos, &(x_out, y_out)).unwrap();
                            
                            graph.edit_weight(
                                &pos, 
                                &(x_out, y_out), 
                                Some(current_weight + update_weight(&cell_grid[x][y], &cell_grid[x_out][y_out]))
                            );  
                        }
                    } // need to also update neurons on receiving end of spiking neuron
                    // create hashmap of what neurons existing neurons point to and use that
                    // generate that hashmap alongside current adjancency list
                }

                output_val.add(&cell_grid);

                if do_stdp && graph_params.write_history {
                    graph.update_history();
                }
            }
        }
    }

    return Ok((output_val, graph));
}


#[derive(Clone)]
struct SimulationParameters {
    num_rows: usize, 
    num_cols: usize, 
    iterations: usize, 
    radius: usize, 
    random_volt_initialization: bool,
    averaged: bool,
    if_params: IFParameters,
    if_type: IFType,
    do_stdp: bool,
    stdp_params: STDPParameters,
    graph_params: GraphParameters,
    default_cell_values: HashMap<String, f64>,
}

#[derive(Clone)]
struct GASettings<'a> {
    equation: &'a str, 
    eeg: &'a Array1<f64>,
    sim_params: SimulationParameters,
    power_density_dt: f64,
}

fn objective(
    bitstring: &BitString, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: usize, 
    settings: &HashMap<&str, GASettings>
) -> Result<f64> {
    let decoded = match decode(bitstring, bounds, n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(e) => return Err(e),
    };

    let ga_settings = settings
        .get("ga_settings")
        .unwrap()
        .clone();
    let equation: &str = ga_settings.equation; // "sign * mp + x + rd * (nc^2 * y)"
    let eeg: &Array1<f64> = ga_settings.eeg;
    let sim_params: SimulationParameters = ga_settings.sim_params;
    let power_density_dt: f64 = ga_settings.power_density_dt;

    let mut symbol_table = SymbolTable::new();
    let sign_id = symbol_table.add_variable("sign", 0.).unwrap().unwrap();
    let mp_id = symbol_table.add_variable("mp", 0.).unwrap().unwrap();
    let rd_id = symbol_table.add_variable("rd", 0.).unwrap().unwrap();
    let nc_id = symbol_table.add_variable("nc", 0.).unwrap().unwrap();
    let x_id = symbol_table.add_variable("x", 0.).unwrap().unwrap();
    let y_id = symbol_table.add_variable("y", 0.).unwrap().unwrap();
    let z_id = symbol_table.add_variable("z", 0.).unwrap().unwrap();

    let (mut expr, _unknown_vars) = Expression::parse_vars(equation, symbol_table).unwrap();

    let mut input_func = |sign: f64, mp: f64, rd: f64, nc: f64| -> f64 {
        expr.symbols().value_cell(sign_id).set(sign);
        expr.symbols().value_cell(mp_id).set(mp);
        expr.symbols().value_cell(rd_id).set(rd);
        expr.symbols().value_cell(nc_id).set(nc);
        expr.symbols().value_cell(x_id).set(decoded[0]);
        expr.symbols().value_cell(y_id).set(decoded[1]);
        expr.symbols().value_cell(z_id).set(decoded[2]);

        expr.value()
    };

    let (output_value, _) = run_simulation(
        sim_params.num_rows, 
        sim_params.num_cols, 
        sim_params.iterations, 
        sim_params.radius, 
        sim_params.random_volt_initialization,
        sim_params.averaged,
        sim_params.if_type,
        &sim_params.if_params,
        sim_params.do_stdp,
        &sim_params.graph_params,
        &sim_params.stdp_params,
        &sim_params.default_cell_values,
        &mut input_func,
        Output::Averaged(vec![]),
    )?;

    let x: Vec<f64> = match output_value {
        Output::Averaged(value) | Output::AveragedBinary(value) => { 
            value.iter()
                .map(|val| val.voltage)
                .collect()
        },
        _ => { unreachable!() },
    };

    let total_time: f64 = sim_params.iterations as f64 * sim_params.if_params.dt;
    // let (_faxis, sxx) = get_power_density(x, sim_params.lif_params.dt, total_time);
    let (_faxis, sxx) = get_power_density(x, power_density_dt, total_time);
    let score = power_density_comparison(eeg, &sxx)?;

    if score.is_nan() || !score.is_finite() {
        return Ok(f64::MAX);
    } else {
        return Ok(score);
    }
}

fn test_coupled_neurons(
    if_type: IFType,
    pre_synaptic_neuron: &mut Cell, 
    post_synaptic_neuron: &mut Cell,
    pre_if_params: &IFParameters,
    post_if_params: &IFParameters,
    iterations: usize,
    input_voltage: f64,
    input_equation: &str,
) -> Vec<(f64, f64)> {
    let mut symbol_table = SymbolTable::new();
    let sign_id = symbol_table.add_variable("sign", 0.).unwrap().unwrap();
    let mp_id = symbol_table.add_variable("mp", 0.).unwrap().unwrap();
    let rd_id = symbol_table.add_variable("rd", 0.).unwrap().unwrap();
    let nc_id = symbol_table.add_variable("nc", 0.).unwrap().unwrap();

    let (mut expr, _unknown_vars) = Expression::parse_vars(input_equation, symbol_table).unwrap();

    let mut input_func = |sign: f64, mp: f64, rd: f64, nc: f64| -> f64 {
        expr.symbols().value_cell(sign_id).set(sign);
        expr.symbols().value_cell(mp_id).set(mp);
        expr.symbols().value_cell(rd_id).set(rd);
        expr.symbols().value_cell(nc_id).set(nc);

        expr.value()
    };

    let mut coupled_voltages: Vec<(f64, f64)> = Vec::new();

    let sign = match pre_synaptic_neuron.potentiation_type {
        PotentiationType::Excitatory => -1., 
        PotentiationType::Inhibitory => 1.,
    };

    let pre_mean_change = &pre_if_params.bayesian_params.mean != &BayesianParameters::default().mean;
    let pre_std_change = &pre_if_params.bayesian_params.std != &BayesianParameters::default().std;
    let pre_bayesian = if pre_mean_change || pre_std_change {
        true
    } else { 
        false
    };

    let post_mean_change = &post_if_params.bayesian_params.mean != &BayesianParameters::default().mean;
    let post_std_change = &post_if_params.bayesian_params.std != &BayesianParameters::default().std;
    let post_bayesian = if post_mean_change || post_std_change {
        true
    } else { 
        false
    };

    match if_type {
        IFType::Basic => { 
            for _ in 0..iterations {
                let (pre_dv, pre_is_spiking) = if pre_bayesian {
                    pre_synaptic_neuron.get_dv_change_and_spike(
                        &pre_if_params, 
                        input_voltage * limited_distr(pre_if_params.bayesian_params.mean, pre_if_params.bayesian_params.std, 0., 1.)
                    )
                } else {
                    pre_synaptic_neuron.get_dv_change_and_spike(&pre_if_params, input_voltage)
                };

                pre_synaptic_neuron.determine_neurotransmitter_concentration(pre_is_spiking);
        
                let input = input_func(
                    sign, 
                    pre_synaptic_neuron.current_voltage, 
                    post_synaptic_neuron.receptor_density,
                    post_synaptic_neuron.neurotransmission_concentration,
                );
        
                let (post_dv, post_is_spiking) = if post_bayesian {
                    post_synaptic_neuron.get_dv_change_and_spike(
                        &pre_if_params, 
                        input * limited_distr(post_if_params.bayesian_params.mean, post_if_params.bayesian_params.std, 0., 1.)
                    )
                } else {
                    post_synaptic_neuron.get_dv_change_and_spike(&post_if_params, input)
                };

                post_synaptic_neuron.determine_neurotransmitter_concentration(post_is_spiking);
            
                pre_synaptic_neuron.current_voltage += pre_dv;
                post_synaptic_neuron.current_voltage += post_dv;
        
                coupled_voltages.push((
                    pre_synaptic_neuron.current_voltage, 
                    post_synaptic_neuron.current_voltage
                ));        
           }
        },
        IFType::Adaptive | IFType::AdaptiveExponential |
        IFType::Izhikevich | IFType::IzhikevichLeaky => {
            let adaptive_apply_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool {
                match if_type {
                    IFType::Basic => unreachable!(),
                    IFType::Adaptive | IFType::AdaptiveExponential => neuron.apply_dw_change_and_get_spike(if_params),
                    IFType::Izhikevich => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                    IFType::IzhikevichLeaky => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                }
            };
    
            let adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 {
                match if_type {
                    IFType::Basic => unreachable!(), 
                    IFType::Adaptive => neuron.adaptive_get_dv_change(if_params, input_value),
                    IFType::AdaptiveExponential => neuron.exp_adaptive_get_dv_change(if_params, input_value),
                    IFType::Izhikevich => neuron.izhikevich_get_dv_change(if_params, input_value),
                    IFType::IzhikevichLeaky => neuron.izhikevich_leaky_get_dv_change(if_params, input_value),
                }
            };

            for _ in 0..iterations {
                let pre_is_spiking = adaptive_apply_and_get_spike(pre_synaptic_neuron, pre_if_params);
                let pre_dv = if pre_bayesian {
                    adaptive_dv(
                        pre_synaptic_neuron,
                        pre_if_params, 
                        input_voltage * limited_distr(pre_if_params.bayesian_params.mean, pre_if_params.bayesian_params.std, 0., 1.)
                    )
                } else {
                    adaptive_dv(
                        pre_synaptic_neuron,
                        pre_if_params, 
                        input_voltage
                    )
                };

                pre_synaptic_neuron.determine_neurotransmitter_concentration(pre_is_spiking);
        
                let input = input_func(
                    sign, 
                    pre_synaptic_neuron.current_voltage, 
                    post_synaptic_neuron.receptor_density,
                    post_synaptic_neuron.neurotransmission_concentration,
                );

                let post_is_spiking = adaptive_apply_and_get_spike(post_synaptic_neuron, post_if_params);
                let post_dv = if post_bayesian {
                    adaptive_dv(
                        post_synaptic_neuron,
                        post_if_params, 
                        input * limited_distr(post_if_params.bayesian_params.mean, post_if_params.bayesian_params.std, 0., 1.)
                    )
                } else {
                    adaptive_dv(
                        post_synaptic_neuron,
                        post_if_params, 
                        input
                    )
                };

                post_synaptic_neuron.determine_neurotransmitter_concentration(post_is_spiking);
            
                pre_synaptic_neuron.current_voltage += pre_dv;
                post_synaptic_neuron.current_voltage += post_dv;
        
                coupled_voltages.push((
                    pre_synaptic_neuron.current_voltage, 
                    post_synaptic_neuron.current_voltage
                )); 
           }
        }
    };

    coupled_voltages
}

#[pyfunction]
#[pyo3(signature = (pre_synaptic_neuron, post_synaptic_neuron, iterations, input_voltage, input_equation))]
pub fn test_coupled_if_cells(
    pre_synaptic_neuron: &mut IFCell, 
    post_synaptic_neuron: &mut IFCell,
    iterations: usize,
    input_voltage: f64,
    input_equation: &str
) -> PyResult<Vec<(f64, f64)>> {
    if pre_synaptic_neuron.mode != post_synaptic_neuron.mode {
        return Err(PyValueError::new_err("Both modes must be the same").into());
    } 

    let output = test_coupled_neurons(
        post_synaptic_neuron.mode.clone(),
        &mut pre_synaptic_neuron.cell_backend, 
        &mut post_synaptic_neuron.cell_backend,
        &pre_synaptic_neuron.if_params,
        &post_synaptic_neuron.if_params,
        iterations,
        input_voltage,
        input_equation,
    );

    Ok(output)
}

fn update_stdp_output(
    output_data: &mut Vec<Vec<f64>>,
    pre_synaptic_neurons: &Vec<Cell>, 
    post_synaptic_neuron: &Cell,
    weights: &Vec<f64>
) {
    for (n, i) in pre_synaptic_neurons.iter().enumerate() {
        output_data[n].push(i.current_voltage);
    }

    output_data[pre_synaptic_neurons.len()].push(post_synaptic_neuron.current_voltage);

    for (n, i) in weights.iter().enumerate() {
        output_data[n + pre_synaptic_neurons.len()].push(*i);
    }
}

// // https://github.com/Abtinmy/computational-neuroscience/blob/main/SNN/snn.py
// // line 61
// fn update_dopamine_decay(
//     dopamine_decay: f64, 
//     weight_change: f64,
//     stdp_params: &STDPParameters, 
//     if_params: &IFParameters,
// ) -> f64 {
//     // might need to multiply by delta dirac of difference in spike times
//     (-dopamine_decay/stdp_params.tau_c + weight_change) * if_params.dt
// }

// fn update_doamine(
//     reward: f64, 
//     dopamine: f64, 
//     spike_time_difference: f64,
//     stdp_params: &STDPParameters, 
//     if_params: &IFParameters
// ) -> f64 {
//     // might wanna multiply tau_d by spike time difference
//     (-dopamine / (stdp_params.tau_d + reward)) * if_params.dt 
// }

// weight * dopamine

fn update_weight(presynaptic_neuron: &Cell, postsynaptic_neuron: &Cell) -> f64 {
    let mut delta_w: f64 = 0.;

    match (presynaptic_neuron.last_firing_time, postsynaptic_neuron.last_firing_time) {
        (Some(t_pre), Some(t_post)) => {
            let (t_pre, t_post): (f64, f64) = (t_pre as f64, t_post as f64);

            if t_pre < t_post {
                delta_w = postsynaptic_neuron.stdp_params.a_plus * (-1. * (t_pre - t_post).abs() / postsynaptic_neuron.stdp_params.tau_plus).exp();
            } else if t_pre > t_post {
                delta_w = -1. * postsynaptic_neuron.stdp_params.a_minus * (-1. * (t_post - t_pre).abs() / postsynaptic_neuron.stdp_params.tau_minus).exp();
            }
        },
        _ => {}
    };

    return delta_w;
}

#[pyfunction]
fn get_weight_change_from_if_cells(pre_synaptic_neuron: &IFCell, post_synaptic_neuron_init: &IFCell) -> f64 {
    update_weight(&pre_synaptic_neuron.cell_backend, &post_synaptic_neuron_init.cell_backend)
}

fn update_isolated_presynaptic_neuron_weights(
    neurons: &mut Vec<Cell>,
    neuron: &Cell,
    weights: &mut Vec<f64>,
    delta_ws: &mut Vec<f64>,
    timestep: usize,
    dvs: Vec<f64>,
    is_spikings: Vec<bool>,
) {
    for (input_neuron, input_dv) in neurons.iter_mut().zip(dvs.iter()) {
        input_neuron.current_voltage += *input_dv;
    }

    for (n, i) in is_spikings.iter().enumerate() {
        if *i {
            neurons[n].last_firing_time = Some(timestep);
            delta_ws[n] = update_weight(&neurons[n], &neuron);
            weights[n] += delta_ws[n];
        }
    }
}

fn run_isolated_stdp_test(
    mut pre_synaptic_neurons: Vec<Cell>,
    pre_synaptic_if_params: Vec<&IFParameters>,
    mut post_synaptic_neuron: Cell,
    post_synaptic_if_params: &IFParameters,
    if_type: IFType,
    iterations: usize,
    input_voltages: Vec<f64>,
    input_equation: &str,
) -> Vec<Vec<f64>> { 
    let mut symbol_table = SymbolTable::new();
    let sign_id = symbol_table.add_variable("sign", 0.).unwrap().unwrap();
    let mp_id = symbol_table.add_variable("mp", 0.).unwrap().unwrap();
    let rd_id = symbol_table.add_variable("rd", 0.).unwrap().unwrap();
    let nc_id = symbol_table.add_variable("nc", 0.).unwrap().unwrap();
    let weight_id = symbol_table.add_variable("weight", 0.).unwrap().unwrap();

    let (mut expr, _unknown_vars) = Expression::parse_vars(input_equation, symbol_table).unwrap();

    let mut input_func = |sign: f64, mp: f64, rd: f64, nc: f64, weight: f64| -> f64 {
        expr.symbols().value_cell(sign_id).set(sign);
        expr.symbols().value_cell(mp_id).set(mp);
        expr.symbols().value_cell(rd_id).set(rd);
        expr.symbols().value_cell(nc_id).set(nc);
        expr.symbols().value_cell(weight_id).set(weight);

        expr.value()
    };

    let mut pre_fires: Vec<Option<usize>> = (0..pre_synaptic_neurons.len()).map(|_| None).collect();
    let mut weights: Vec<f64> = (0..pre_synaptic_neurons.len()).map( // get weights from toml and set them higher
        |_| limited_distr(
            post_synaptic_neuron.stdp_params.weight_bayesian_params.mean, 
            post_synaptic_neuron.stdp_params.weight_bayesian_params.std, 
            post_synaptic_neuron.stdp_params.weight_bayesian_params.min, 
            post_synaptic_neuron.stdp_params.weight_bayesian_params.max,
        )
    ).collect();

    let mut delta_ws: Vec<f64> = (0..pre_synaptic_neurons.len())
        .map(|_| 0.0)
        .collect();

    let mut output: Vec<Vec<f64>> = (0..pre_synaptic_neurons.len() * 2 + 1)
        .map(|_| vec![])
        .collect();

    match if_type {
        IFType::Basic => {
            for timestep in 0..iterations {
                let (mut dvs, mut is_spikings): (Vec<f64>, Vec<bool>) = (Vec::new(), Vec::new()); 

                for (n_neuron, input_neuron) in pre_synaptic_neurons.iter_mut().enumerate() {
                    let input_if_params = pre_synaptic_if_params[n_neuron];
                    let (dv, is_spiking) = if input_if_params.bayesian_params.std != 0. {
                        input_neuron.get_dv_change_and_spike(
                            &input_if_params, 
                            input_voltages[n_neuron] * limited_distr(
                                input_if_params.bayesian_params.mean, 
                                input_if_params.bayesian_params.std, 
                                0., 
                                1.
                            )
                        )
                    } else {
                        input_neuron.get_dv_change_and_spike(&input_if_params, input_voltages[n_neuron])
                    };

                    dvs.push(dv);
                    is_spikings.push(is_spiking);

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }

                    input_neuron.determine_neurotransmitter_concentration(is_spiking);                    
                }

                let calculated_voltage: f64 = (0..pre_synaptic_neurons.len())
                    .map(
                        |i| {
                            let sign = match pre_synaptic_neurons[i].potentiation_type { 
                                PotentiationType::Excitatory => -1., 
                                PotentiationType::Inhibitory => 1.,
                            };

                            input_func(
                                sign, 
                                pre_synaptic_neurons[i].current_voltage, 
                                pre_synaptic_neurons[i].receptor_density,
                                pre_synaptic_neurons[i].neurotransmission_concentration,
                                weights[i]
                            ) / (pre_synaptic_neurons.len() as f64)
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
                
                let noise_factor = limited_distr(
                    post_synaptic_if_params.bayesian_params.mean, 
                    post_synaptic_if_params.bayesian_params.std, 
                    0., 
                    1.
                );
                let (dv, is_spiking) = post_synaptic_neuron.get_dv_change_and_spike(
                    &post_synaptic_if_params, noise_factor * calculated_voltage
                );

                post_synaptic_neuron.determine_neurotransmitter_concentration(is_spiking);                    

                update_isolated_presynaptic_neuron_weights(
                    &mut pre_synaptic_neurons, 
                    &post_synaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
                    dvs, 
                    is_spikings,
                );

                post_synaptic_neuron.current_voltage += dv;

                if is_spiking {
                    post_synaptic_neuron.last_firing_time = Some(timestep);
                    for (n, i) in pre_synaptic_neurons.iter().enumerate() {
                        delta_ws[n] = update_weight(&i, &post_synaptic_neuron);
                        weights[n] += delta_ws[n];
                    }
                }

                update_stdp_output(
                    &mut output, 
                    &pre_synaptic_neurons, 
                    &post_synaptic_neuron, 
                    &weights
                );
            }
        }
        IFType::Adaptive | IFType::AdaptiveExponential | 
        IFType::Izhikevich | IFType::IzhikevichLeaky => {
            let adaptive_apply_and_get_spike = |neuron: &mut Cell, if_params: &IFParameters| -> bool {
                match if_type {
                    IFType::Basic => unreachable!(),
                    IFType::Adaptive | IFType::AdaptiveExponential => neuron.apply_dw_change_and_get_spike(if_params),
                    IFType::Izhikevich => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                    IFType::IzhikevichLeaky => neuron.izhikevich_apply_dw_and_get_spike(if_params),
                }
            };

            let adaptive_dv = |neuron: &mut Cell, if_params: &IFParameters, input_value: f64| -> f64 {
                match if_type {
                    IFType::Basic => unreachable!(), 
                    IFType::Adaptive => neuron.adaptive_get_dv_change(if_params, input_value),
                    IFType::AdaptiveExponential => neuron.exp_adaptive_get_dv_change(if_params, input_value),
                    IFType::Izhikevich => neuron.izhikevich_get_dv_change(if_params, input_value),
                    IFType::IzhikevichLeaky => neuron.izhikevich_leaky_get_dv_change(if_params, input_value),
                }
            };

            for timestep in 0..iterations {
                let (mut dvs, mut is_spikings): (Vec<f64>, Vec<bool>) = (Vec::new(), Vec::new()); 

                for (n_neuron, input_neuron) in pre_synaptic_neurons.iter_mut().enumerate() {
                    let input_if_params = pre_synaptic_if_params[n_neuron];
                    let is_spiking = adaptive_apply_and_get_spike(input_neuron, &input_if_params);

                    let dv = if input_if_params.bayesian_params.std != 0. {
                        adaptive_dv(
                            input_neuron,
                            &input_if_params, 
                            input_voltages[n_neuron] * limited_distr(
                                input_if_params.bayesian_params.mean, 
                                input_if_params.bayesian_params.std, 
                                0., 
                                1.
                            )
                        )
                    } else {
                        adaptive_dv(input_neuron, &input_if_params, input_voltages[n_neuron])
                    };

                    dvs.push(dv);
                    is_spikings.push(is_spiking);

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }

                    input_neuron.determine_neurotransmitter_concentration(is_spiking);                    
                }

                let is_spiking = adaptive_apply_and_get_spike(&mut post_synaptic_neuron, &post_synaptic_if_params);

                post_synaptic_neuron.determine_neurotransmitter_concentration(is_spiking);  

                let calculated_voltage: f64 = (0..pre_synaptic_neurons.len())
                    .map(
                        |i| {
                            let sign = match pre_synaptic_neurons[i].potentiation_type { 
                                PotentiationType::Excitatory => -1., 
                                PotentiationType::Inhibitory => 1.,
                            };

                            input_func(
                                sign, 
                                pre_synaptic_neurons[i].current_voltage, 
                                pre_synaptic_neurons[i].receptor_density,
                                pre_synaptic_neurons[i].neurotransmission_concentration,
                                weights[i]
                            ) / (pre_synaptic_neurons.len() as f64)
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();                  

                let noise_factor = limited_distr(
                    post_synaptic_if_params.bayesian_params.mean, 
                    post_synaptic_if_params.bayesian_params.std, 
                    0., 
                    1.
                );
                let dv = adaptive_dv(&mut post_synaptic_neuron, &post_synaptic_if_params, noise_factor * calculated_voltage);

                update_isolated_presynaptic_neuron_weights(
                    &mut pre_synaptic_neurons, 
                    &post_synaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
                    dvs, 
                    is_spikings,
                );

                post_synaptic_neuron.current_voltage += dv;

                if is_spiking {
                    post_synaptic_neuron.last_firing_time = Some(timestep);
                    for (n, i) in pre_synaptic_neurons.iter().enumerate() {
                        delta_ws[n] = update_weight(&i, &post_synaptic_neuron);
                        weights[n] += delta_ws[n];
                    }
                }

                update_stdp_output(
                    &mut output, 
                    &pre_synaptic_neurons, 
                    &post_synaptic_neuron, 
                    &weights
                );
            }
        }
    };

    output
}

// mut pre_synaptic_neurons: Vec<Cell>,
// pre_synaptic_if_params: Vec<&IFParameters>,
// mut post_synaptic_neuron: Cell,
// post_synaptic_if_params: &IFParameters,
// if_type: IFType,
// iterations: usize,
// input_voltages: Vec<f64>,
// input_equation: &str,
#[pyfunction]
#[pyo3(signature = (pre_synaptic_neurons_init, post_synaptic_neuron_init, iterations, input_voltages, input_equation))]
fn test_isolated_stdp(
    pre_synaptic_neurons_init: Vec<IFCell>, 
    post_synaptic_neuron_init: &IFCell,
    iterations: usize,
    input_voltages: Vec<f64>,
    input_equation: &str
) -> PyResult<Vec<Vec<f64>>> {
    if pre_synaptic_neurons_init.iter().map(|i| i.mode.clone()).any(|i| i != post_synaptic_neuron_init.mode) {
        return Err(PyValueError::new_err("Both modes must be the same").into());
    }
    
    // try directly inputting ifcell into stdp test so you dont have to clone
    let output = run_isolated_stdp_test(
        pre_synaptic_neurons_init.iter().map(|i| i.cell_backend.clone()).collect::<Vec<Cell>>(), 
        pre_synaptic_neurons_init.iter().map(|i| &i.if_params).collect::<Vec<&IFParameters>>(), 
        post_synaptic_neuron_init.cell_backend.clone(), 
        &post_synaptic_neuron_init.if_params, 
        post_synaptic_neuron_init.mode.clone(), 
        iterations, 
        input_voltages, 
        input_equation
    );
    
    Ok(output)
}

#[pymodule]
#[pyo3(name = "lixirnet")]
fn lixirnet(_py: Python, m: &PyModule) -> PyResult<()> {
    // python3 -m venv .venv
    // source .venv/bin/activate
    // pip install -U pip maturin

    m.add_class::<IFType>()?;
    m.add_class::<IFCell>()?;
    m.add_class::<HodgkinHuxleyModel>()?;

    m.add_function(wrap_pyfunction!(test_coupled_if_cells, m)?)?;

    m.add_function(wrap_pyfunction!(get_weight_change_from_if_cells, m)?)?;
    m.add_function(wrap_pyfunction!(test_isolated_stdp, m)?)?;

    m.add_function(wrap_pyfunction!(create_cell_grid, m)?)?;

    Ok(())
}
