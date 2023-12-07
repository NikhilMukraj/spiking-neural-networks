use std::{
    collections::HashMap, 
    fs::{File, read_to_string}, 
    io::{Write, BufWriter, Result, Error, ErrorKind}, 
    env,
};
use rand::{Rng, seq::SliceRandom};
use toml::{from_str, Value};
use exprtk_rs::{Expression, SymbolTable};
use ndarray::Array1;
#[path = "distribution/mod.rs"]
mod distribution;
use crate::distribution::limited_distr;
mod neuron;
use crate::neuron::{
    IFParameters, IFType, PotentiationType, Cell, CellGrid, 
    ScaledDefault, IzhikevichDefault, BayesianParameters, STDPParameters
};
mod eeg;
use crate::eeg::{read_eeg_csv, get_power_density, power_density_comparison};
mod ga;
use crate::ga::{BitString, decode, genetic_algo};
mod graph;
use crate::graph::{Position, AdjacencyList, AdjacencyMatrix, Graph, GraphParameters, GraphFunctionality};


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
                params.bayesian_mean, 
                params.bayesian_std, 
                params.bayesian_min, 
                params.bayesian_max
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
                params.bayesian_mean, 
                params.bayesian_std, 
                params.bayesian_min, 
                params.bayesian_max
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
    default_cell_values: &HashMap<&str, f64>,
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

    let mean_change = &if_params.bayesian_mean != &BayesianParameters::default().mean;
    let std_change = &if_params.bayesian_std != &BayesianParameters::default().std;
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
                    a_plus: stdp_params.a_plus,
                    a_minus: stdp_params.a_minus,
                    tau_plus: stdp_params.tau_plus,
                    tau_minus: stdp_params.tau_minus,
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

fn parse_bool(value: &Value, field_name: &str) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Cannot parse {} as boolean", field_name)))
        .map(|v| v as bool)
}

fn parse_usize(value: &Value, field_name: &str) -> Result<usize> {
    value
        .as_integer()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Cannot parse {} as unsigned integer", field_name)))
        .map(|v| v as usize)
}

fn parse_f64(value: &Value, field_name: &str) -> Result<f64> {
    value
        .as_float()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Cannot parse {} as float32", field_name)))
        .map(|v| v as f64)
}

fn parse_string(value: &Value, field_name: &str) -> Result<String> {
    value
        .as_str()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Cannot parse {} as string", field_name)))
        .map(|v| String::from(v))
}

fn parse_value_with_default<T>(
    table: &Value,
    key: &str,
    parser: impl Fn(&Value, &str) -> Result<T>,
    default: T,
) -> Result<T> {
    table
        .get(key)
        .map_or(Ok(default), |value| parser(value, key))
}

#[derive(Clone)]
struct SimulationParameters<'a> {
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
    default_cell_values: HashMap<&'a str, f64>,
}

fn get_if_params(if_params: &mut IFParameters, table: &Value) -> Result<()> {
    if_params.dt = parse_value_with_default(table, "dt", parse_f64, if_params.dt)?;
    if_params.exp_dt = parse_value_with_default(table, "exp_dt", parse_f64, if_params.exp_dt)?;
    if_params.tau_m = parse_value_with_default(table, "tau_m", parse_f64, if_params.tau_m)?;
    if_params.tref = parse_value_with_default(table, "tref", parse_f64, if_params.tref)?;
    if_params.alpha_init = parse_value_with_default(table, "alpha_init", parse_f64, if_params.alpha_init)?;
    if_params.beta_init = parse_value_with_default(table, "beta_init", parse_f64, if_params.beta_init)?;
    if_params.v_reset = parse_value_with_default(table, "v_reset", parse_f64, if_params.v_reset)?; 
    if_params.d_init = parse_value_with_default(table, "d_init", parse_f64, if_params.d_init)?;
    if_params.w_init = parse_value_with_default(table, "w_init", parse_f64, if_params.w_init)?;
    if_params.bayesian_mean = parse_value_with_default(table, "bayesian_mean", parse_f64, if_params.bayesian_mean)?;
    if_params.bayesian_std = parse_value_with_default(table, "bayesian_std", parse_f64, if_params.bayesian_std)?;
    if_params.bayesian_max = parse_value_with_default(table, "bayesian_max", parse_f64, if_params.bayesian_max)?;
    if_params.bayesian_min = parse_value_with_default(table, "bayesian_min", parse_f64, if_params.bayesian_min)?;

    Ok(())
}

fn get_stdp_params(stdp: &mut STDPParameters, table: &Value) -> Result<()> {
    stdp.a_plus = parse_value_with_default(
        table, 
        "a_plus", 
        parse_f64, 
        STDPParameters::default().a_plus
    )?;
    println!("a_plus: {}", stdp.a_plus);

    stdp.a_minus = parse_value_with_default(
        table, 
        "a_minus", 
        parse_f64, 
        STDPParameters::default().a_minus
    )?;
    println!("a_minus: {}", stdp.a_minus);

    stdp.tau_plus = parse_value_with_default(
        table, 
        "tau_plus", 
        parse_f64, 
        stdp.tau_plus
    )?; 
    println!("tau_plus: {}", stdp.tau_plus);

    stdp.tau_minus = parse_value_with_default(
        table, 
        "tau_minus", 
        parse_f64, 
        stdp.tau_minus
    )?; 
    println!("tau_minus: {}", stdp.tau_minus);

    stdp.weight_init = parse_value_with_default(
        table, 
        "weight_init", 
        parse_f64, 
        stdp.weight_init
    )?;
    println!("weight_init: {}", stdp.weight_init);

    stdp.weight_std = parse_value_with_default(
        table, 
        "weight_std", 
        parse_f64, 
        stdp.weight_std
    )?;
    println!("weight_std: {}", stdp.weight_std);

    stdp.weight_min = parse_value_with_default(
        table, 
        "weight_min", 
        parse_f64, 
        stdp.weight_min
    )?;
    println!("weight_min: {}", stdp.weight_min);

    stdp.weight_max = parse_value_with_default(
        table, 
        "weight_max", 
        parse_f64, 
        stdp.weight_max
    )?;
    println!("weight_max: {}", stdp.weight_max);

    Ok(())
}

fn get_default_cell_parameters(table: &Value) -> Result<HashMap<&str, f64>> {
    let mut default_cell_values: HashMap<&str, f64> = HashMap::new();
    default_cell_values.insert("neurotransmission_release", 1.);
    default_cell_values.insert("receptor_density", 1.);
    default_cell_values.insert("chance_of_releasing", 0.5);
    default_cell_values.insert("dissipation_rate", 0.1);
    default_cell_values.insert("chance_of_random_release", 0.2);
    default_cell_values.insert("random_release_concentration", 0.1);
    default_cell_values.insert("excitatory_chance", 0.5);

    default_cell_values.insert("neurotransmission_release_std", 0.);
    default_cell_values.insert("receptor_density_std", 0.);
    default_cell_values.insert("dissipation_rate_std", 0.);
    default_cell_values.insert("random_release_concentration_std", 0.);

    let updates: Vec<(&str, Result<f64>)> = default_cell_values
        .iter()
        .map(|(&key, &default_value)| {
            let value_to_update = parse_value_with_default(
                &table, key, parse_f64, default_value
            );

            (key, value_to_update)
        })
        .collect();

    for (key, value_to_update) in updates {
        let value_to_update = match value_to_update {
            Ok(output_value) => output_value,
            Err(e) => { 
                let err_msg = format!("Error with key '{}'\nError: {}", key, e.to_string());
                return Err(Error::new(ErrorKind::InvalidInput, err_msg)); 
            }
        };

        default_cell_values.insert(key, value_to_update);
        println!("{}: {}", key, value_to_update);
    }

    return Ok(default_cell_values);
}

fn get_parameters(table: &Value) -> Result<SimulationParameters> {
    let num_rows: usize = parse_value_with_default(&table, "num_rows", parse_usize, 10)?;
    println!("num_rows: {}", num_rows);

    let num_cols: usize = parse_value_with_default(&table, "num_cols", parse_usize, 10)?;
    println!("num_cols: {}", num_cols);

    let radius: usize = parse_value_with_default(&table, "radius", parse_usize, 1)?;
    println!("radius: {}", radius);

    let random_volt_initialization = parse_value_with_default(&table, "random_volt_initialization", parse_bool, false)?;
    println!("random_volt_initialization: {}", random_volt_initialization);

    let if_type: String = parse_value_with_default(table, "if_type", parse_string, String::from("basic"))?;

    let if_type = match IFType::from_str(&if_type) {
        Ok(if_type_val) => if_type_val,
        Err(_e) => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'if_type' as one of the valid types")) }
    };
    println!("if_type: {:#?}", if_type);

    let averaged: bool = parse_value_with_default(table, "averaged", parse_bool, false)?;
    println!("averaged: {}", averaged);

    let output_type: String = parse_value_with_default(table, "output_type", parse_string, String::from("averaged"))?;
    println!("output_type: {}", output_type);

    let do_stdp: bool = parse_value_with_default(&table, "do_stdp", parse_bool, false)?;
    println!("do_stdp: {}", do_stdp);

    let mut stdp_params = STDPParameters::default();

    get_stdp_params(&mut stdp_params, &table)?;

    let graph_type: String = parse_value_with_default(&table, "graph_type", parse_string, String::from("list"))?;

    let graph_type = match Graph::from_str(&graph_type) {
        Ok(graph_type_val) => graph_type_val,
        Err(_e) => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'graph_type' as one of the valid types")) }
    };
    println!("graph_type: {:#?}", graph_type);
    
    let write_weights = parse_value_with_default(&table, "write_weights", parse_bool, false)?;
    println!("write_weights: {}", write_weights);    

    let write_history = parse_value_with_default(&table, "write_history", parse_bool, false)?;
    println!("write_history: {}", write_history);

    let graph_params = GraphParameters {
        graph_type: graph_type,
        write_weights: write_weights,
        write_history: write_history,
    };

    let default_cell_values: HashMap<&str, f64> = get_default_cell_parameters(&table)?;

    let scaling_type_default = match if_type {
        IFType::Izhikevich | IFType::IzhikevichLeaky => "izhikevich",
        _ => "regular",
    };
    let scaling_type: String = parse_value_with_default(
        table, 
        "scaling_type", 
        parse_string, 
        String::from(scaling_type_default)
    )?;
    println!("scaling_type: {}", scaling_type);

    let mut if_params = match scaling_type.as_str() {
        "regular" => IFParameters { ..IFParameters::default() },
        "scaled" => IFParameters { ..IFParameters::scaled_default() },
        "izhikevich" | "adaptive quadratic" => IFParameters { ..IFParameters::izhikevich_default() },
        _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown scaling")) }
    };

    get_if_params(&mut if_params, table)?;
    
    println!("{:#?}", if_params);

    // in ms
    let total_time: Option<usize> = match table.get("total_time") {
        Some(value) => {
            match value.as_integer() {
                Some(output_value) => Some(output_value as usize),
                None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'total_time' as unsigned integer")); }
            }
        },
        None => None,
    };

    let iterations: usize = match (table.get("iterations"), total_time) {
        (Some(_), Some(_)) => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot have both 'iterations' and 'total_time' argument")); }
        (Some(value), None) => {
            match value.as_integer() {
                Some(output_value) => output_value as usize,
                None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'iterations' as unsigned integer")); }
            }
        },
        (None, Some(total_time_value)) => { (total_time_value as f64 / if_params.dt) as usize },
        (None, None) => { return Err(Error::new(ErrorKind::InvalidInput, "Missing 'iterations' or 'total_time' argument")); },
    };
    println!("iterations: {}", iterations);

    return Ok(SimulationParameters {
        num_rows: num_rows, 
        num_cols: num_cols, 
        iterations: iterations, 
        radius: radius, 
        random_volt_initialization: random_volt_initialization,
        averaged: averaged,
        if_params: if_params,
        if_type: if_type,
        do_stdp: do_stdp,
        stdp_params: stdp_params,
        graph_params: graph_params,
        default_cell_values: default_cell_values,
    });
}

#[derive(Clone)]
struct GASettings<'a> {
    equation: &'a str, 
    eeg: &'a Array1<f64>,
    sim_params: SimulationParameters<'a>,
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

fn write_row(
    file: &mut File, 
    presynaptic_neurons: &Vec<Cell>, 
    postsynaptic_neuron: &Cell, 
    weights: &Vec<f64>,
) {
    write!(
        file, 
        "{}, ", 
        presynaptic_neurons.iter()
            .map(|i| i.current_voltage)
            .collect::<Vec<f64>>()
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    ).expect("Cannot write to file");
    write!(file, "{}, ", postsynaptic_neuron.current_voltage)
        .expect("Cannot write to file");
    write!(
        file, 
        "{}, ", 
        presynaptic_neurons.iter()
            .map(|i| i.neurotransmission_concentration)
            .collect::<Vec<f64>>()
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    ).expect("Cannot write to file");
    write!(file, "{}, ", postsynaptic_neuron.neurotransmission_concentration)
        .expect("Cannot write to file");
    write!(
        file, 
        "{}, ", 
        weights.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    ).expect("Cannot write to file");
    write!(
        file, 
        "{}, ", 
        presynaptic_neurons.iter()
            .map(|x| {
                match x.last_firing_time {
                    Some(value) => value.to_string(),
                    None => String::from("None"),
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    ).expect("Cannot write to file");
    writeln!(
        file, 
        "{}", 
        match postsynaptic_neuron.last_firing_time {
            Some(value) => value.to_string(),
            None => String::from("None"),
        }
    ).expect("Cannot write to file");
}

fn update_weight(presynaptic_neuron: &Cell, postsynaptic_neuron: &Cell) -> f64 {
    let mut delta_w: f64 = 0.;

    match (presynaptic_neuron.last_firing_time, postsynaptic_neuron.last_firing_time) {
        (Some(t_pre), Some(t_post)) => {
            let (t_pre, t_post): (f64, f64) = (t_pre as f64, t_post as f64);

            if t_pre < t_post {
                delta_w = postsynaptic_neuron.a_plus * (-1. * (t_pre - t_post).abs() / postsynaptic_neuron.tau_plus).exp();
            } else if t_pre > t_post {
                delta_w = -1. * postsynaptic_neuron.a_minus * (-1. * (t_post - t_pre).abs() / postsynaptic_neuron.tau_minus).exp();
            }
        },
        _ => {}
    };

    return delta_w;
}

// fn neuromodulate_by_correlation(presynaptic_neuron: &Cell, postsynaptic_neuron: &Cell) -> f64 {
//     // tonic
//     // dt = 0.5
//     // a = 0.01
//     // b = 0.25
//     // w_init = 30.0
//     // v_reset = -55.0
//     // d = 8.0

//     // bursting
//     // dt = 0.5
//     // a = 0.035
//     // w_init = 10.0
//     // v_reset = -50.0
//     // d = 4.0

//     // determine correlation between pre and post
//     // if positive, move toward bursting, scale appropriately
//     // if negative, move toward tonic, scale appropriately
// }

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
    stdp_table: &Value,
    stdp_params: &STDPParameters,
    if_type: IFType,
    iterations: usize,
    n: usize,
    input_voltage: f64,
    default_cell_values: &HashMap<&str, f64>,
    input_equation: &str,
    filename: &str,
) -> Result<()> {
    let mut if_params = match if_type {
        IFType::Basic | IFType::Adaptive |
        IFType::AdaptiveExponential => {
            IFParameters {
                ..IFParameters::default()
            }
        },
        IFType::Izhikevich | IFType::IzhikevichLeaky => {
            IFParameters {
                ..IzhikevichDefault::izhikevich_default()
            }
        }
    };

    get_if_params(&mut if_params, &stdp_table)?;

    println!("{:#?}", if_params);

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

    let mut postsynaptic_neuron = match if_type {
        IFType::Basic | IFType::Adaptive |
        IFType::AdaptiveExponential => {
            Cell {
                ..Cell::default()
            }
        },
        IFType::Izhikevich | IFType::IzhikevichLeaky => {
            Cell {
                ..Cell::izhikevich_default()
            }
        }
    };

    // let mut postsynaptic_neuron = Cell { 
    //     current_voltage: if_params.v_init, 
    //     refractory_count: 0.0,
    //     leak_constant: -1.,
    //     integration_constant: 1.,
    //     potentiation_type: PotentiationType::Excitatory,
    //     neurotransmission_concentration: 0., 
    //     neurotransmission_release: *default_cell_values.get("neurotransmission_release").unwrap_or(&0.),
    //     receptor_density: *default_cell_values.get("receptor_density").unwrap_or(&0.),
    //     chance_of_releasing: *default_cell_values.get("chance_of_releasing").unwrap_or(&0.), 
    //     dissipation_rate: *default_cell_values.get("dissipation_rate").unwrap_or(&0.), 
    //     chance_of_random_release: *default_cell_values.get("chance_of_random_release").unwrap_or(&0.),
    //     random_release_concentration: *default_cell_values.get("random_release_concentration").unwrap_or(&0.),
    //     w_value: if_params.w_init,
    //     a_plus: stdp_params.a_plus,
    //     a_minus: stdp_params.a_minus,
    //     tau_plus: stdp_params.tau_plus,
    //     tau_minus: stdp_params.tau_minus,
    //     last_firing_time: None,
    // };

    postsynaptic_neuron.w_value = if_params.v_init;
    postsynaptic_neuron.neurotransmission_release = *default_cell_values.get("neurotransmission_release").unwrap_or(&0.);
    postsynaptic_neuron.receptor_density = *default_cell_values.get("receptor_density").unwrap_or(&0.);
    postsynaptic_neuron.chance_of_releasing = *default_cell_values.get("chance_of_releasing").unwrap_or(&0.);
    postsynaptic_neuron.dissipation_rate = *default_cell_values.get("dissipation_rate").unwrap_or(&0.);
    postsynaptic_neuron.chance_of_random_release = *default_cell_values.get("chance_of_random_release").unwrap_or(&0.);
    postsynaptic_neuron.random_release_concentration = *default_cell_values.get("random_release_concentration").unwrap_or(&0.);
    postsynaptic_neuron.w_value = if_params.w_init;
    postsynaptic_neuron.a_plus = stdp_params.a_plus;
    postsynaptic_neuron.a_minus = stdp_params.a_minus;
    postsynaptic_neuron.tau_plus = stdp_params.tau_plus;
    postsynaptic_neuron.tau_minus = stdp_params.tau_minus;
    postsynaptic_neuron.alpha = if_params.alpha_init;
    postsynaptic_neuron.beta = if_params.beta_init;
    postsynaptic_neuron.c = if_params.v_reset;
    postsynaptic_neuron.d = if_params.d_init;

    let mut neurons: Vec<Cell> = (0..n).map(|_| postsynaptic_neuron.clone())
        .collect();

    for i in neurons.iter_mut() {
        if rand::thread_rng().gen_range(0.0..=1.0) < *default_cell_values.get("excitatory_chance").unwrap_or(&0.) {
            i.potentiation_type = PotentiationType::Excitatory;
        } else {
            i.potentiation_type = PotentiationType::Inhibitory;
        }
    }

    let input_voltages: Vec<f64> = (0..n).map(|_| input_voltage * limited_distr(1.0, 0.1, 0., 2.))
        .collect();

    let mut pre_fires: Vec<Option<usize>> = (0..n).map(|_| None).collect();
    let mut weights: Vec<f64> = (0..n).map( // get weights from toml and set them higher
        |_| limited_distr(
            stdp_params.weight_init, 
            0.1, 
            stdp_params.weight_init * 0.5, 
            stdp_params.weight_init * 1.5
        )
    ).collect();

    let mut delta_ws: Vec<f64> = (0..n)
        .map(|_| 0.0)
        .collect();

    let mut file = File::create(&filename)
        .expect("Unable to create file");

    write_row(&mut file, &neurons, &postsynaptic_neuron, &weights);

    match if_type {
        IFType::Basic => {
            for timestep in 0..iterations {
                let (mut dvs, mut is_spikings): (Vec<f64>, Vec<bool>) = (Vec::new(), Vec::new()); 

                for (n_neuron, input_neuron) in neurons.iter_mut().enumerate() {
                    let (dv, is_spiking) = if if_params.bayesian_std != 0. {
                        input_neuron.get_dv_change_and_spike(
                            &if_params, 
                            input_voltages[n_neuron] * limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.)
                        )
                    } else {
                        input_neuron.get_dv_change_and_spike(&if_params, input_voltages[n_neuron])
                    };

                    dvs.push(dv);
                    is_spikings.push(is_spiking);

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }

                    input_neuron.determine_neurotransmitter_concentration(is_spiking);                    
                }

                let calculated_voltage: f64 = (0..n)
                    .map(
                        |i| {
                            let sign = match neurons[i].potentiation_type { 
                                PotentiationType::Excitatory => -1., 
                                PotentiationType::Inhibitory => 1.,
                            };

                            input_func(
                                sign, 
                                neurons[i].current_voltage, 
                                neurons[i].receptor_density,
                                neurons[i].neurotransmission_concentration,
                                weights[i]
                            ) / (n as f64)
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
                
                let noise_factor = limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.);
                let (dv, is_spiking) = postsynaptic_neuron.get_dv_change_and_spike(&if_params, noise_factor * calculated_voltage);

                postsynaptic_neuron.determine_neurotransmitter_concentration(is_spiking);                    

                update_isolated_presynaptic_neuron_weights(
                    &mut neurons, 
                    &postsynaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
                    dvs, 
                    is_spikings,
                );

                postsynaptic_neuron.current_voltage += dv;

                if is_spiking {
                    postsynaptic_neuron.last_firing_time = Some(timestep);
                    for (n_neuron, i) in neurons.iter().enumerate() {
                        delta_ws[n_neuron] = update_weight(&i, &postsynaptic_neuron);
                        weights[n_neuron] += delta_ws[n_neuron];
                    }
                }

                write_row(&mut file, &neurons, &postsynaptic_neuron, &weights);
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

                for (n_neuron, input_neuron) in neurons.iter_mut().enumerate() {
                    let is_spiking = adaptive_apply_and_get_spike(input_neuron, &if_params);

                    let dv = if if_params.bayesian_std != 0. {
                        adaptive_dv(
                            input_neuron,
                            &if_params, 
                            input_voltages[n_neuron] * limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.)
                        )
                    } else {
                        adaptive_dv(input_neuron, &if_params, input_voltages[n_neuron])
                    };

                    dvs.push(dv);
                    is_spikings.push(is_spiking);

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }

                    input_neuron.determine_neurotransmitter_concentration(is_spiking);                    
                }

                let is_spiking = adaptive_apply_and_get_spike(&mut postsynaptic_neuron, &if_params);

                postsynaptic_neuron.determine_neurotransmitter_concentration(is_spiking);  

                let calculated_voltage: f64 = (0..n)
                    .map(
                        |i| {
                            let sign = match neurons[i].potentiation_type { 
                                PotentiationType::Excitatory => -1., 
                                PotentiationType::Inhibitory => 1.,
                            };

                            input_func(
                                sign, 
                                neurons[i].current_voltage, 
                                neurons[i].receptor_density,
                                neurons[i].neurotransmission_concentration,
                                weights[i]
                            ) / (n as f64)
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();                  

                let noise_factor = limited_distr(if_params.bayesian_mean, if_params.bayesian_std, 0., 1.);
                let dv = adaptive_dv(&mut postsynaptic_neuron, &if_params, noise_factor * calculated_voltage);

                update_isolated_presynaptic_neuron_weights(
                    &mut neurons, 
                    &postsynaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
                    dvs, 
                    is_spikings,
                );

                postsynaptic_neuron.current_voltage += dv;

                if is_spiking {
                    postsynaptic_neuron.last_firing_time = Some(timestep);
                    for (n, i) in neurons.iter().enumerate() {
                        delta_ws[n] = update_weight(&i, &postsynaptic_neuron);
                        weights[n] += delta_ws[n];
                    }
                }

                write_row(&mut file, &neurons, &postsynaptic_neuron, &weights);
            }
        }
    };

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Requires .toml argument file");
        return Err(Error::new(ErrorKind::InvalidInput, "Requires .toml argument file"));
    }

    let toml_content = read_to_string(&args[1]).expect("Cannot read file");
    let config: Value = from_str(&toml_content).expect("Cannot read config");

    if let Some(simulation_table) = config.get("lattice_simulation") {
        let output_type: String = parse_value_with_default(
            &simulation_table, 
            "output_type", 
            parse_string, 
            String::from("averaged")
        )?;
        println!("output_type: {}", output_type);

        let output_type = Output::from_str(&output_type)?;

        let tag: &str = match simulation_table.get("tag") {
            Some(value) => { 
                match value.as_str() {
                    Some(str_value) => str_value,
                    None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'tag'")) },
                }
            },
            None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'tag'")) },
        };
        println!("tag: {}", tag);

        let sim_params = get_parameters(&simulation_table)?;

        let default_eq = match sim_params.if_type { // izhikevich currently untested with neurotransmitter
            IFType::Izhikevich | IFType::IzhikevichLeaky => String::from("(sign * mp + 65) / 15."),
            _ => String::from("sign * mp + 100 + rd * (nc^2 * 200)")
        };

        let equation: String = parse_value_with_default(
            &simulation_table, 
            "input_equation", 
            parse_string, 
            default_eq
        )?;
        let equation: &str = equation.trim();
        println!("input equation: {}", equation);
    
        let mut symbol_table = SymbolTable::new();
        let sign_id = symbol_table.add_variable("sign", 0.).unwrap().unwrap();
        let mp_id = symbol_table.add_variable("mp", 0.).unwrap().unwrap();
        let rd_id = symbol_table.add_variable("rd", 0.).unwrap().unwrap();
        let nc_id = symbol_table.add_variable("nc", 0.).unwrap().unwrap();
    
        let (mut expr, _unknown_vars) = Expression::parse_vars(equation, symbol_table).unwrap();
    
        let mut input_func = |sign: f64, mp: f64, rd: f64, nc: f64| -> f64 {
            expr.symbols().value_cell(sign_id).set(sign);
            expr.symbols().value_cell(mp_id).set(mp);
            expr.symbols().value_cell(rd_id).set(rd);
            expr.symbols().value_cell(nc_id).set(nc);
    
            expr.value()
        };

        let (output_value, output_graph) = run_simulation(
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
            output_type,
        )?;

        let (mut voltage_file, mut neurotransmitter_file) = match output_value {
            Output::Grid(_) | Output::Averaged(_) => { 
                (   
                    BufWriter::new(File::create(format!("{}_voltage.txt", tag))
                        .expect("Could not create file")),
                    BufWriter::new(File::create(format!("{}_neurotransmitter.txt", tag))
                        .expect("Could not create file"))
                )
            },
            Output::GridBinary(_) | Output::AveragedBinary(_) => { 
                (   
                    BufWriter::new(File::create(format!("{}_voltage.bin", tag))
                        .expect("Could not create file")),
                    BufWriter::new(File::create(format!("{}_neurotransmitter.bin", tag))
                        .expect("Could not create file"))
                )
            },
        };

        output_value.write_to_file(&mut voltage_file, &mut neurotransmitter_file);

        if sim_params.graph_params.write_history {
            output_graph.write_history(&tag);
        } else if sim_params.graph_params.write_weights {
            output_graph.write_current_weights(&tag);
        }

        println!("Finished lattice simulation");
    } else if let Some(ga_table) = config.get("ga") {
        let n_bits: usize = parse_value_with_default(&ga_table, "n_bits", parse_usize, 10)?;
        println!("n_bits: {}", n_bits);

        let n_iter: usize = parse_value_with_default(&ga_table, "n_iter", parse_usize, 100)?;
        println!("n_iter: {}", n_iter);

        let n_pop: usize = parse_value_with_default(&ga_table, "n_pop", parse_usize, 100)?;
        println!("n_pop: {}", n_pop);

        let r_cross: f64 = parse_value_with_default(&ga_table, "r_cross", parse_f64, 0.9)?;
        println!("r_cross: {}", r_cross);

        let r_mut: f64 = parse_value_with_default(&ga_table, "r_mut", parse_f64, 0.1)?;
        println!("r_mut: {}", r_mut);

        let k: usize = 3;

        let equation: String = parse_value_with_default(
            &ga_table, 
            "input_equation", 
            parse_string, 
            String::from("(sign * mp + x + rd * (nc^2 * y)) * z")
        )?;
        let equation: &str = equation.trim();
        println!("equation: {}", equation);

        let sim_params = get_parameters(&ga_table)?;

        // make sure lif_params.exp_dt = lif_params.dt
        // sim_params.lif_params.exp_dt = sim_params.lif_params.dt;

        let eeg_file: &str = match ga_table.get("eeg_file") {
            Some(value) => {
                match value.as_str() {
                    Some(output_value) => output_value,
                    None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'eeg_file' as string")); }
                }
            },
            None => { return Err(Error::new(ErrorKind::InvalidInput, "Requires 'eeg_file' argument")); },
        };

        // eeg should have column specifying dt and total time
        let (x, dt, total_time) = read_eeg_csv(eeg_file)?;
        let (_faxis, sxx) = get_power_density(x, dt, total_time);

        let power_density_dt: f64 = parse_value_with_default(
            &ga_table, 
            "power_density_dt", 
            parse_f64, 
            dt
        )?;
        println!("power density calculation dt: {}", power_density_dt);

        let ga_settings = GASettings {
            equation: equation, 
            eeg: &sxx,
            sim_params: sim_params,
            power_density_dt: power_density_dt,
        };

        let mut settings: HashMap<&str, GASettings<'_>> = HashMap::new();
        settings.insert("ga_settings", ga_settings);

        let bounds_min: f64 = parse_value_with_default(&ga_table, "bounds_min", parse_f64, 0.)?;
        let bounds_max: f64 = parse_value_with_default(&ga_table, "bounds_max", parse_f64, 100.)?;

        let bounds: Vec<Vec<f64>> = (0..3)
            .map(|_| vec![bounds_min, bounds_max])
            .collect();

        println!("\nstarting genetic algorithm...");
        let (best_bitstring, best_score, _scores) = genetic_algo(
            objective, 
            &bounds, 
            n_bits, 
            n_iter, 
            n_pop, 
            r_cross,
            r_mut, 
            k, 
            &settings,
        )?;

        println!("best bitstring: {}", best_bitstring.string);
        println!("best score: {}", best_score);

        let decoded = match decode(&best_bitstring, &bounds, n_bits) {
            Ok(decoded_value) => decoded_value,
            Err(e) => return Err(e),
        };

        println!("decoded values: {:#?}", decoded);

        // option to run a simulation and return the eeg signals
        // option to write custom bounds
    } else if let Some(single_neuron_test) = config.get("single_neuron_test") {
        let filename: &str = match single_neuron_test.get("filename") {
            Some(value) => { 
                match value.as_str() {
                    Some(str_value) => str_value,
                    None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse filename")) },
                }
            },
            None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse filename")) },
        };
        println!("filename: {}", filename);

        let iterations: usize = match single_neuron_test.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);

        let input: f64 = match single_neuron_test.get("input") {
            Some(value) => parse_f64(value, "input")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input' value not found")); },
        };
        println!("input: {}", input);  
        
        let if_type: String = parse_value_with_default(
            single_neuron_test, 
            "if_type", 
            parse_string, 
            String::from("basic")
        )?;
        println!("if_type: {}", if_type);

        let if_type = IFType::from_str(&if_type)?;

        let scaling_type_default = match if_type {
            IFType::Izhikevich | IFType::IzhikevichLeaky => "izhikevich",
            _ => "regular",
        };
        let scaling_type: String = parse_value_with_default(
            single_neuron_test, 
            "scaling_type", 
            parse_string, 
            String::from(scaling_type_default)
        )?;
        println!("scaling_type: {}", scaling_type);

        let mut if_params = match scaling_type.as_str() {
            "regular" => IFParameters { ..IFParameters::default() },
            "scaled" => IFParameters { ..IFParameters::scaled_default() },
            "izhikevich" | "adaptive quadratic" => IFParameters { ..IFParameters::izhikevich_default() },
            _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown scaling")) }
        };

        get_if_params(&mut if_params, &single_neuron_test)?;

        // let bayesian: bool = parse_value_with_default(single_neuron_test, "bayesian", parse_bool, false)?; 

        let mean_change = &if_params.bayesian_mean != &BayesianParameters::default().mean;
        let std_change = &if_params.bayesian_std != &BayesianParameters::default().std;
        let bayesian = if mean_change || std_change {
            true
        } else {
            false
        };
        
        println!("{:#?}", if_params);

        let mut test_cell = Cell { 
            current_voltage: if_params.v_init, 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            potentiation_type: PotentiationType::Excitatory,
            neurotransmission_concentration: 0., 
            neurotransmission_release: 1.,
            receptor_density: 1.,
            chance_of_releasing: 0.5, 
            dissipation_rate: 0.1, 
            chance_of_random_release: 0.2,
            random_release_concentration: 0.1,
            w_value: if_params.w_init,
            a_plus: STDPParameters::default().a_plus,
            a_minus: STDPParameters::default().a_minus,
            tau_plus: STDPParameters::default().tau_plus,
            tau_minus: STDPParameters::default().tau_minus,
            last_firing_time: None,
            alpha: if_params.alpha_init,
            beta: if_params.beta_init,
            c: if_params.v_reset,
            d: if_params.d_init,
        };

        match if_type {
            IFType::Basic => { 
                test_cell.run_static_input(&if_params, input, bayesian, iterations, filename); 
            },
            IFType::Adaptive => { 
                test_cell.run_adaptive_static_input(&if_params, input, bayesian, iterations, filename); 
            },
            IFType::AdaptiveExponential => { 
                test_cell.run_exp_adaptive_static_input(&if_params, input, bayesian, iterations, filename);
            },
            IFType::Izhikevich => { 
                test_cell.run_izhikevich_static_input(&if_params, input, bayesian, iterations, filename); 
            },
            IFType::IzhikevichLeaky => {
                test_cell.run_izhikevich_leaky_static_input(&if_params, input, bayesian, iterations, filename);
            },
        };

        println!("\nFinished volt test");
    } else if let Some(stdp_table) = config.get("stdp_test") {
        let if_type: String = parse_value_with_default(
            stdp_table, 
            "if_type", 
            parse_string, 
            String::from("basic")
        )?;
        println!("if_type: {}", if_type);

        let if_type = IFType::from_str(&if_type)?;

        let iterations: usize = match stdp_table.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);
    
        let filename: String = match stdp_table.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);
    
        let n: usize = match stdp_table.get("n") {
            Some(value) => parse_usize(value, "n")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'n' value not found")); },
        };
        println!("n: {}", n); 
    
        let input_voltage: f64 = match stdp_table.get("input_voltage") {
            Some(value) => parse_f64(value, "input_voltage")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_voltage' value not found")); },
        };
        println!("input_voltage: {}", input_voltage);

        let default_cell_values: HashMap<&str, f64> = get_default_cell_parameters(&stdp_table)?;

        let default_eq = match if_type {
            IFType::Izhikevich | IFType::IzhikevichLeaky => String::from("(sign * mp + 65) / 15."),
            _ => String::from("weight * (sign * mp + 100)")
        };

        let equation: String = parse_value_with_default(
            &stdp_table, 
            "input_equation", 
            parse_string, 
            default_eq
        )?;
        let equation: &str = equation.trim();
        println!("\ninput equation: {}", equation);

        let mut stdp_params = STDPParameters::default();

        get_stdp_params(&mut stdp_params, stdp_table)?;

        run_isolated_stdp_test(
            stdp_table,
            &stdp_params,
            if_type,
            iterations,
            n,
            input_voltage,
            &default_cell_values,
            &equation,
            &filename,
        )?;

        println!("\nFinished STDP test");
    } else {
        return Err(Error::new(ErrorKind::InvalidInput, "Simulation config not found"));
    }

    Ok(())
}
