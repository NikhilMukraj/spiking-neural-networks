use std::{
    collections::HashMap, 
    env, 
    f64::consts::PI, 
    fs::{read_to_string, File}, 
    io::{BufWriter, Error, ErrorKind, Result, Write}
};
use rand::{Rng, seq::SliceRandom};
use toml::{from_str, Value};
// use ndarray::Array1;
mod distribution;
use crate::distribution::limited_distr;
mod neuron;
use crate::neuron::{
    IFParameters, IFType, PotentiationType, Cell, CellGrid, 
    ScaledDefault, IzhikevichDefault, BayesianParameters, STDPParameters, 
    if_params_bayesian, gap_junction, iterate_coupled_hodgkin_huxley,
    Gate, HodgkinHuxleyCell, GeneralLigandGatedChannel, AMPADefault, GABAaDefault, 
    GABAbDefault, GABAbDefault2, NMDAWithBV, BV, AdditionalGates, HighThresholdCalciumChannel,
    HighVoltageActivatedCalciumChannel, handle_receptor_kinetics
};
// mod eeg;
// use crate::eeg::{read_eeg_csv, get_power_density, power_density_comparison};
mod ga;
use crate::ga::{decode, genetic_algo};
mod fitting;
use crate::fitting::{
    FittingSettings, fitting_objective, 
    get_hodgkin_huxley_voltages, get_reference_scale, get_izhikevich_summary,
    print_action_potential_summaries, ActionPotentialSummary,
    SummaryScalingDefaults, SummaryScalingFactors,
};
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

fn signed_gap_junction(presynaptic_neuron: &Cell, postsynaptic_neuron: &Cell, sign: f64) -> f64 {
    sign * gap_junction(presynaptic_neuron, postsynaptic_neuron)
}

fn get_sign(cell: &Cell) -> f64 {
    match cell.potentiation_type {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    }
}

fn handle_bayesian_modifier(if_params: Option<&IFParameters>, input_val: f64) -> f64 {
    match if_params {
        Some(params) => { 
            input_val * if_params_bayesian(params)
        },
        None => input_val,
    }
}

fn get_input_from_positions(
    cell_grid: &CellGrid, 
    postsynaptic_neuron: &Cell,
    input_positions: &Vec<Position>, 
    if_params: Option<&IFParameters>,
    averaged: bool,
) -> f64 {
    let mut input_val = input_positions
        .iter()
        .map(|input_position| {
            let (pos_x, pos_y) = input_position;
            let input_cell = &cell_grid[*pos_x][*pos_y];
            
            let sign = get_sign(&input_cell);

            let final_input = signed_gap_junction(&input_cell, &postsynaptic_neuron, sign);
            
            final_input
        })
        .sum();

    input_val = handle_bayesian_modifier(if_params, input_val);

    if averaged {
        input_val /= input_positions.len() as f64;
    }

    return input_val;
}

fn weighted_get_input_from_positions(
    cell_grid: &CellGrid, 
    graph: &dyn GraphFunctionality,
    position: &Position,
    input_positions: &Vec<Position>, 
    if_params: Option<&IFParameters>,
    averaged: bool,
) -> f64 {
    let (x, y) = position;
    let postsynaptic_neuron = &cell_grid[*x][*y];

    let mut input_val = input_positions
        .iter()
        .map(|input_position| {
            let (pos_x, pos_y) = input_position;
            let input_cell = &cell_grid[*pos_x][*pos_y];
            
            let sign = get_sign(&input_cell);

            let final_input = signed_gap_junction(&input_cell, postsynaptic_neuron, sign);
            
            final_input * graph.lookup_weight(&input_position, position).unwrap()

        })
        .sum();

    input_val = handle_bayesian_modifier(if_params, input_val);

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

// fn get_neuro_avg(cell_grid: &CellGrid) -> f64 {
//     let neuro_mean: f64 = cell_grid
//         .iter()
//         .flatten()
//         .map(|x| x.neurotransmission_concentration)
//         .sum();

//     neuro_mean / ((cell_grid[0].len() * cell_grid.len()) as f64) 
// }

// struct NeuroAndVolts {
//     voltage: f64,
//     neurotransmitter: f64,
// }

// distance: 6.8 mm
// conductivity: 0.251 S/m 
// refence voltage: 7 uV (microvolt)
// either convert dist to m, or conductivity to S/mm
// should be inputtable from user as well
fn convert_to_eeg(cell_grid: &CellGrid, distance: f64, conductivity: f64, reference_voltage: f64) -> f64 {
    let mut total_current: f64 = 0.;

    for row in cell_grid {
        for value in row {
            total_current += value.current_voltage - reference_voltage;
        }
    }

    (1. / (4. * PI * conductivity * distance)) * total_current
}

// fn create_voltage_and_neuro_files(tag: &str, extension: &str) -> (BufWriter<File>, BufWriter<File>) {
//     let voltage_file = BufWriter::new(File::create(format!("{}_voltage.{}", tag, extension))
//         .expect("Could not create file"));
//     let neurotransmitter_file = BufWriter::new(File::create(format!("{}_neurotransmitter.{}", tag, extension))
//         .expect("Could not create file"));

//     (voltage_file, neurotransmitter_file)
// }

enum Output {
    Grid(Vec<CellGrid>),
    GridBinary(Vec<CellGrid>),
    Averaged(Vec<f64>), // NeuroAndVolts
    AveragedBinary(Vec<f64>), // NeuroAndVolts
    EEG(Vec<f64>, f64, f64, f64), // Vec<(f64, f64)>, f64, f64, f64
}

impl Output {
    fn add(&mut self, cell_grid: &CellGrid) {
        match self {
            Output::Grid(grids) | Output::GridBinary(grids) => { grids.push(cell_grid.clone()) }
            Output::Averaged(averages) | Output::AveragedBinary(averages) => { 
                averages.push(
                    // NeuroAndVolts {
                    //     voltage: get_volt_avg(cell_grid),
                    //     neurotransmitter: get_neuro_avg(cell_grid),
                    // }
                    get_volt_avg(cell_grid)
                );
            },
            Output::EEG(signals, distance, conductivity, reference_voltage) => {
                signals.push(
                    convert_to_eeg(cell_grid, *distance, *conductivity, *reference_voltage)
                    // get_neuro_avg(cell_grid),
                )
            }
        }
    }

    fn from_str(string: &str, distance: f64, conductivity: f64, reference_voltage: f64) -> Result<Output> {
        match string.to_ascii_lowercase().as_str() {
            "grid" => { Ok(Output::Grid(Vec::<CellGrid>::new())) },
            "grid binary" => { Ok(Output::GridBinary(Vec::<CellGrid>::new())) },
            "averaged" => { Ok(Output::Averaged(Vec::<f64>::new())) },
            "averaged binary" => { Ok(Output::AveragedBinary(Vec::<f64>::new())) },
            "eeg" => { Ok(Output::EEG(Vec::<f64>::new(), distance, conductivity, reference_voltage)) }
            _ => { Err(Error::new(ErrorKind::InvalidInput, "Unknown output type")) }
        }
    }

    fn write_to_file(&self, tag: &str) {
        match &self {
            Output::Grid(grids) => {
                // let (mut voltage_file, mut neurotransmitter_file) = create_voltage_and_neuro_files(tag, "txt");
                let mut voltage_file = BufWriter::new(File::create(format!("{}_voltage.{}", tag, ".txt"))
                    .expect("Could not create file"));

                for grid in grids {
                    for row in grid {
                        for value in row {
                            write!(voltage_file, "{} ", value.current_voltage)
                                .expect("Could not write to file");
                            // write!(neurotransmitter_file, "{} ", value.neurotransmission_concentration)
                            //     .expect("Could not write to file");
                        }
                        writeln!(voltage_file)
                            .expect("Could not write to file");
                        // writeln!(neurotransmitter_file)
                        //     .expect("Could not write to file");
                    }
                    writeln!(voltage_file, "-----")
                        .expect("Could not write to file"); 
                    // writeln!(neurotransmitter_file, "-----")
                    //     .expect("Could not write to file"); 
                }
            },
            Output::GridBinary(grids) => {
                // let (mut voltage_file, mut neurotransmitter_file) = create_voltage_and_neuro_files(tag, "bin");
                let mut voltage_file = BufWriter::new(File::create(format!("{}_voltage.{}", tag, ".bin"))
                    .expect("Could not create file"));

                for grid in grids {
                    for row in grid {
                        for value in row {
                            let bytes = value.current_voltage.to_le_bytes();
                            voltage_file
                                .write_all(&bytes)
                                .expect("Could not write to file");
                
                            // let bytes = value.neurotransmission_concentration.to_le_bytes();
                            // neurotransmitter_file
                            //     .write_all(&bytes)
                            //     .expect("Could not write to file");
                        }
                    }
                }
            },
            Output::Averaged(averages) => {
                // let (mut voltage_file, mut neurotransmitter_file) = create_voltage_and_neuro_files(tag, "txt");
                let mut voltage_file = BufWriter::new(File::create(format!("{}_voltage.{}", tag, ".txt"))
                    .expect("Could not create file"));

                for voltage in averages {
                    writeln!(voltage_file, "{}", voltage)
                        .expect("Could not write to file");
                    // writeln!(neurotransmitter_file, "{}", neuro_and_volt.neurotransmitter)
                    //     .expect("Could not write to file");
                } 
            },
            Output::AveragedBinary(averages) => {
                // let (mut voltage_file, mut neurotransmitter_file) = create_voltage_and_neuro_files(tag, "bin");
                let mut voltage_file = BufWriter::new(File::create(format!("{}_voltage.{}", tag, ".bin"))
                    .expect("Could not create file"));

                for voltage in averages {
                    let volt_mean_bytes = voltage.to_le_bytes();
                    // let neuro_mean_bytes = neuro_and_volt.neurotransmitter.to_le_bytes();

                    voltage_file.write_all(&volt_mean_bytes).expect("Could not write to file"); 
                    // neurotransmitter_file.write_all(&neuro_mean_bytes).expect("Could not write to file");
                }
            },
            Output::EEG(signals, _, _, _) => {
                let mut eeg_file = BufWriter::new(File::create(format!("{}_eeg.txt", tag))
                    .expect("Could not create file"));

                for value in signals {
                    writeln!(eeg_file, "{}", value)
                        .expect("Could not write to file");
                }
            }
        }
    }
}

fn run_simulation(
    num_rows: usize, 
    num_cols: usize, 
    iterations: usize, 
    radius: usize, 
    random_volt_initialization: bool,
    averaged: bool,
    excitatory_chance: f64,
    if_type: IFType,
    if_params: &IFParameters,
    do_stdp: bool,
    graph_params: &GraphParameters,
    stdp_params: &STDPParameters,
    mut output_val: Output,
) -> Result<(Output, Box<dyn GraphFunctionality>)> {
    if radius / 2 > num_rows || radius / 2 > num_cols || radius == 0 {
        let err_msg = "Radius must be less than both number of rows or number of cols divided by 2 and greater than 0";
        return Err(Error::new(ErrorKind::InvalidInput, err_msg));
    }

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
                    gap_conductance: if_params.gap_conductance_init,
                    potentiation_type: PotentiationType::weighted_random_type(excitatory_chance),
                    w_value: if_params.w_init,
                    stdp_params: stdp_params.clone(),
                    last_firing_time: None,
                    alpha: if_params.alpha_init,
                    beta: if_params.beta_init,
                    c: if_params.v_reset,
                    d: if_params.d_init,
                    ligand_gates: if_params.ligand_gates_init.clone(),
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
                let mut changes: HashMap<Position, bool> = graph.get_every_node()
                    .iter()
                    .map(|key| (*key, false))
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
                            bayesian,
                            averaged,
                        )
                    } else {
                        get_input_from_positions(
                            &cell_grid, 
                            &cell_grid[x][y],
                            &input_positions, 
                            bayesian,
                            averaged,
                        )
                    };
                    
                    let (dv, is_spiking) = cell_grid[x][y].get_dv_change_and_spike(if_params, input);
                    cell_grid[x][y].current_voltage += dv;

                    changes.insert(pos, is_spiking);
                }

                // loop through every cell
                // modify the voltage
                // end loop

                for (pos, is_spiking_value) in changes {
                    let (x, y) = pos;

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
                            bayesian,
                            averaged,
                        )
                    } else {
                        get_input_from_positions(
                            &cell_grid, 
                            &cell_grid[x][y],
                            &input_positions, 
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
struct SimulationParameters {
    num_rows: usize, 
    num_cols: usize, 
    iterations: usize, 
    radius: usize, 
    random_volt_initialization: bool,
    averaged: bool,
    excitatory_chance: f64,
    if_params: IFParameters,
    if_type: IFType,
    do_stdp: bool,
    stdp_params: STDPParameters,
    graph_params: GraphParameters,
}

fn get_if_params(if_params: &mut IFParameters, prefix: Option<&str>, table: &Value) -> Result<()> {
    let prefix_value = match prefix {
        Some(value) => format!("{}_", value),
        None => String::from(""),
    };

    if_params.gap_conductance_init = parse_value_with_default(table, &format!("{}gap_conductance_init", prefix_value), parse_f64, if_params.gap_conductance_init)?;
    if_params.dt = parse_value_with_default(table, &format!("{}dt", prefix_value), parse_f64, if_params.dt)?;
    if_params.slope_factor = parse_value_with_default(table, &format!("{}slope_factor", prefix_value), parse_f64, if_params.slope_factor)?;
    if_params.tau_m = parse_value_with_default(table, &format!("{}tau_m", prefix_value), parse_f64, if_params.tau_m)?;
    if_params.c_m = parse_value_with_default(table, &format!("{}c_m", prefix_value), parse_f64, if_params.c_m)?;
    if_params.tref = parse_value_with_default(table, &format!("{}tref", prefix_value), parse_f64, if_params.tref)?;
    if_params.alpha_init = parse_value_with_default(table, &format!("{}alpha_init", prefix_value), parse_f64, if_params.alpha_init)?;
    if_params.beta_init = parse_value_with_default(table, &format!("{}beta_init", prefix_value), parse_f64, if_params.beta_init)?;
    if_params.v_reset = parse_value_with_default(table, &format!("{}v_reset", prefix_value), parse_f64, if_params.v_reset)?; 
    if_params.d_init = parse_value_with_default(table, &format!("{}d_init", prefix_value), parse_f64, if_params.d_init)?;
    if_params.w_init = parse_value_with_default(table, &format!("{}w_init", prefix_value), parse_f64, if_params.w_init)?;
    if_params.v_init = parse_value_with_default(table, &format!("{}v_init", prefix_value), parse_f64, if_params.w_init)?;
    if_params.bayesian_params.mean = parse_value_with_default(table, &format!("{}bayesian_mean", prefix_value), parse_f64, if_params.bayesian_params.mean)?;
    if_params.bayesian_params.std = parse_value_with_default(table, &format!("{}bayesian_std", prefix_value), parse_f64, if_params.bayesian_params.std)?;
    if_params.bayesian_params.max = parse_value_with_default(table, &format!("{}bayesian_max", prefix_value), parse_f64, if_params.bayesian_params.max)?;
    if_params.bayesian_params.min = parse_value_with_default(table, &format!("{}bayesian_min", prefix_value), parse_f64, if_params.bayesian_params.min)?;

    if_params.ligand_gates_init = get_ligand_gated_channel(table, &format!("{}", prefix_value))?;

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

    stdp.weight_bayesian_params.mean = parse_value_with_default(
        table, 
        "weight_init", 
        parse_f64, 
        stdp.weight_bayesian_params.mean
    )?;
    println!("weight_init: {}", stdp.weight_bayesian_params.mean);

    stdp.weight_bayesian_params.std = parse_value_with_default(
        table, 
        "weight_std", 
        parse_f64, 
        stdp.weight_bayesian_params.std
    )?;
    println!("weight_std: {}", stdp.weight_bayesian_params.std);

    stdp.weight_bayesian_params.min = parse_value_with_default(
        table, 
        "weight_min", 
        parse_f64, 
        stdp.weight_bayesian_params.min
    )?;
    println!("weight_min: {}", stdp.weight_bayesian_params.min);

    stdp.weight_bayesian_params.max = parse_value_with_default(
        table, 
        "weight_max", 
        parse_f64, 
        stdp.weight_bayesian_params.max
    )?;
    println!("weight_max: {}", stdp.weight_bayesian_params.max);

    Ok(())
}

fn get_bayesian_params(
    bayesian_params: &mut BayesianParameters, 
    table: &Value, 
    prefix: Option<&str>
) -> Result<()> {
    let (mean_string, std_string, min_string, max_string) = match prefix {
        Some(prefix_value) => (
            format!("{}_bayesian_mean", prefix_value),
            format!("{}_bayesian_std", prefix_value),
            format!("{}_bayesian_min", prefix_value),
            format!("{}_bayesian_max", prefix_value),
        ),
        None => (
            String::from("bayesian_mean"),
            String::from("bayesian_std"), 
            String::from("bayesian_min"), 
            String::from("bayesian_max"),
        )
    };

    bayesian_params.mean = parse_value_with_default(
        table, 
        &mean_string, 
        parse_f64, 
        bayesian_params.mean
    )?;
    println!("{}: {}", mean_string, bayesian_params.mean);

    bayesian_params.std = parse_value_with_default(
        table, 
        &std_string, 
        parse_f64, 
        bayesian_params.std
    )?;
    println!("{}: {}", std_string, bayesian_params.std);

    bayesian_params.min = parse_value_with_default(
        table, 
        &min_string, 
        parse_f64, 
        bayesian_params.min
    )?;
    println!("{}: {}", min_string, bayesian_params.min);

    bayesian_params.max = parse_value_with_default(
        table, 
        &max_string, 
        parse_f64, 
        bayesian_params.max
    )?;
    println!("{}: {}", max_string, bayesian_params.max);

    Ok(())
}

fn get_simulation_parameters(table: &Value) -> Result<SimulationParameters> {
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

    let excitatory_chance = parse_value_with_default(&table, "excitatory_chance", parse_f64, 0.8)?;
    println!("excitatory_chance: {}", excitatory_chance);

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

    get_if_params(&mut if_params, None, table)?;
    
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
        excitatory_chance: excitatory_chance,
        if_params: if_params,
        if_type: if_type,
        do_stdp: do_stdp,
        stdp_params: stdp_params,
        graph_params: graph_params,
    });
}

// #[derive(Clone)]
// struct GASettings<'a> {
//     equation: &'a str, 
//     eeg: &'a Array1<f64>,
//     sim_params: SimulationParameters,
//     power_density_dt: f64,
// }

// fn objective(
//     bitstring: &BitString, 
//     bounds: &Vec<Vec<f64>>, 
//     n_bits: usize, 
//     settings: &HashMap<&str, GASettings>
// ) -> Result<f64> {
//     let decoded = match decode(bitstring, bounds, n_bits) {
//         Ok(decoded_value) => decoded_value,
//         Err(e) => return Err(e),
//     };

//     let ga_settings = settings
//         .get("ga_settings")
//         .unwrap()
//         .clone();
//     let eeg: &Array1<f64> = ga_settings.eeg;
//     let sim_params: SimulationParameters = ga_settings.sim_params;
//     let power_density_dt: f64 = ga_settings.power_density_dt;

//     let (output_value, _) = run_simulation(
//         sim_params.num_rows, 
//         sim_params.num_cols, 
//         sim_params.iterations, 
//         sim_params.radius, 
//         sim_params.random_volt_initialization,
//         sim_params.averaged,
//         sim_params.if_type,
//         &sim_params.if_params,
//         sim_params.do_stdp,
//         &sim_params.graph_params,
//         &sim_params.stdp_params,
//         &sim_params.default_cell_values,
//         Output::Averaged(vec![]),
//     )?;

//     let x: Vec<f64> = match output_value {
//         Output::Averaged(value) | Output::AveragedBinary(value) => { 
//             value.iter()
//                 .map(|val| val.voltage)
//                 .collect()
//         },
//         _ => { unreachable!() },
//     };

//     let total_time: f64 = sim_params.iterations as f64 * sim_params.if_params.dt;
//     // let (_faxis, sxx) = get_power_density(x, sim_params.lif_params.dt, total_time);
//     let (_faxis, sxx) = get_power_density(x, power_density_dt, total_time);
//     let score = power_density_comparison(eeg, &sxx)?;

//     if score.is_nan() || !score.is_finite() {
//         return Ok(f64::MAX);
//     } else {
//         return Ok(score);
//     }
// }

fn test_coupled_neurons(
    if_type: IFType,
    pre_if_params: &IFParameters, 
    post_if_params: &IFParameters,
    pre_potentiation_type: PotentiationType,
    iterations: usize,
    input_current: f64,
    do_receptor_kinetics: bool,
    full: bool,
    filename: &str,
) {
    let mut presynaptic_neuron = Cell { 
        current_voltage: pre_if_params.v_init, 
        refractory_count: 0.0,
        leak_constant: -1.,
        integration_constant: 1.,
        gap_conductance: pre_if_params.gap_conductance_init,
        potentiation_type: pre_potentiation_type,
        w_value: pre_if_params.w_init,
        stdp_params: STDPParameters::default(),
        last_firing_time: None,
        alpha: pre_if_params.alpha_init,
        beta: pre_if_params.beta_init,
        c: pre_if_params.v_reset,
        d: pre_if_params.d_init,
        ligand_gates: pre_if_params.ligand_gates_init.clone(),
    };

    let mut postsynaptic_neuron = Cell { 
        current_voltage: post_if_params.v_init, 
        refractory_count: 0.0,
        leak_constant: -1.,
        integration_constant: 1.,
        gap_conductance: post_if_params.gap_conductance_init,
        potentiation_type: PotentiationType::Excitatory,
        w_value: post_if_params.w_init,
        stdp_params: STDPParameters::default(),
        last_firing_time: None,
        alpha: post_if_params.alpha_init,
        beta: post_if_params.beta_init,
        c: post_if_params.v_reset,
        d: post_if_params.d_init,
        ligand_gates: post_if_params.ligand_gates_init.clone(),
    };

    if !do_receptor_kinetics {
        presynaptic_neuron.ligand_gates.iter_mut().for_each(|i| {
            i.neurotransmitter.r = 0.8;
        });
        postsynaptic_neuron.ligand_gates.iter_mut().for_each(|i| {
            i.neurotransmitter.r = 0.8;
        });
    }

    let mut file = BufWriter::new(File::create(filename)
        .expect("Unable to create file"));
    write!(file, "pre_voltage,post_voltage").expect("Unable to write to file");
    if full && postsynaptic_neuron.ligand_gates.len() != 0{
        for i in postsynaptic_neuron.ligand_gates.iter() {
            let name = i.to_str();
            write!(file, ",g_{},r_{},T_{}", name, name, name).expect("Unable to write to file");
        }
    }
    write!(file, "\n").expect("Unable to write to file");

    let sign = get_sign(&presynaptic_neuron);

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
                let (pre_dv, _pre_is_spiking) = if pre_bayesian {
                    let bayesian_factor = limited_distr(pre_if_params.bayesian_params.mean, pre_if_params.bayesian_params.std, 0., 1.);
                    let bayesian_input = input_current * bayesian_factor;
                    handle_receptor_kinetics(&mut presynaptic_neuron, &pre_if_params, bayesian_input, do_receptor_kinetics);

                    let (dv, is_spiking) = presynaptic_neuron.get_dv_change_and_spike(
                        &pre_if_params, 
                        bayesian_input
                    );

                    (
                        dv + presynaptic_neuron.get_neurotransmitter_currents(&pre_if_params),
                        is_spiking,
                    )
                } else {
                    handle_receptor_kinetics(&mut presynaptic_neuron, &pre_if_params, input_current, do_receptor_kinetics);

                    let (dv, is_spiking) = presynaptic_neuron.get_dv_change_and_spike(&pre_if_params, input_current);

                    (
                        dv + presynaptic_neuron.get_neurotransmitter_currents(&pre_if_params),
                        is_spiking,
                    )
                };
        
                let input = signed_gap_junction(&presynaptic_neuron, &postsynaptic_neuron, sign);
        
                let (post_dv, _post_is_spiking) = if post_bayesian {
                    let bayesian_factor = limited_distr(post_if_params.bayesian_params.mean, post_if_params.bayesian_params.std, 0., 1.);
                    let bayesian_input = input * bayesian_factor;
                    handle_receptor_kinetics(&mut postsynaptic_neuron, &post_if_params, bayesian_input, do_receptor_kinetics);

                    let (dv, is_spiking) = postsynaptic_neuron.get_dv_change_and_spike(
                        &pre_if_params, 
                        bayesian_input
                    );

                    (
                        dv + postsynaptic_neuron.get_neurotransmitter_currents(&post_if_params),
                        is_spiking,
                    )
                } else {
                    handle_receptor_kinetics(&mut postsynaptic_neuron, &post_if_params, input, do_receptor_kinetics);

                    let (dv, is_spiking) = postsynaptic_neuron.get_dv_change_and_spike(&post_if_params, input);

                    (
                        dv + postsynaptic_neuron.get_neurotransmitter_currents(&post_if_params),
                        is_spiking,
                    )
                };
            
                presynaptic_neuron.current_voltage += pre_dv;
                postsynaptic_neuron.current_voltage += post_dv;
           
                if !full || postsynaptic_neuron.ligand_gates.len() == 0 {
                    writeln!(file, "{}, {}", 
                        presynaptic_neuron.current_voltage,
                        postsynaptic_neuron.current_voltage,
                    ).expect("Unable to write to file");
                } else {
                    write!(file, "{}, {}", 
                        presynaptic_neuron.current_voltage, 
                        postsynaptic_neuron.current_voltage,
                    ).expect("Unable to write to file");
        
                    for i in postsynaptic_neuron.ligand_gates.iter() {
                        write!(file, ", {}, {}, {}", 
                            i.current,
                            i.neurotransmitter.r,
                            i.neurotransmitter.t,
                        ).expect("Unable to write to file");
                    }
        
                    write!(file, "\n").expect("Unable to write to file");
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
                    IFType::Adaptive => neuron.adaptive_get_dv_change(if_params, input_value) + neuron.get_neurotransmitter_currents(if_params),
                    IFType::AdaptiveExponential => neuron.exp_adaptive_get_dv_change(if_params, input_value) + neuron.get_neurotransmitter_currents(if_params),
                    IFType::Izhikevich => neuron.izhikevich_get_dv_change(if_params, input_value) + neuron.get_neurotransmitter_currents(if_params),
                    IFType::IzhikevichLeaky => neuron.izhikevich_leaky_get_dv_change(if_params, input_value) + neuron.get_neurotransmitter_currents(if_params),
                }
            };

            for _ in 0..iterations {
                let _pre_is_spiking = adaptive_apply_and_get_spike(&mut presynaptic_neuron, &pre_if_params);
                let pre_dv = if pre_bayesian {
                    let bayesian_factor = limited_distr(pre_if_params.bayesian_params.mean, pre_if_params.bayesian_params.std, 0., 1.);
                    let bayesian_input = input_current * bayesian_factor;
                    handle_receptor_kinetics(&mut presynaptic_neuron, &pre_if_params, bayesian_input, do_receptor_kinetics);

                    let dv = adaptive_dv(
                        &mut presynaptic_neuron,
                        &pre_if_params, 
                        bayesian_input
                    );

                    dv
                } else {
                    handle_receptor_kinetics(&mut presynaptic_neuron, &pre_if_params, input_current, do_receptor_kinetics);

                    let dv = adaptive_dv(
                        &mut presynaptic_neuron,
                        &pre_if_params, 
                        input_current
                    );

                    dv
                };

                let pre_i_syn = presynaptic_neuron.get_neurotransmitter_currents(&pre_if_params);
        
                let input = signed_gap_junction(&presynaptic_neuron, &postsynaptic_neuron, sign);

                let _post_is_spiking = adaptive_apply_and_get_spike(&mut postsynaptic_neuron, &post_if_params);
                let post_dv = if post_bayesian {
                    let bayesian_factor = limited_distr(post_if_params.bayesian_params.mean, post_if_params.bayesian_params.std, 0., 1.);
                    let bayesian_input = input * bayesian_factor;
                    handle_receptor_kinetics(&mut postsynaptic_neuron, &post_if_params, bayesian_input, do_receptor_kinetics);

                    let dv = adaptive_dv(
                        &mut postsynaptic_neuron,
                        &post_if_params, 
                        bayesian_input
                    );

                    dv
                } else {
                    handle_receptor_kinetics(&mut postsynaptic_neuron, &post_if_params, input, do_receptor_kinetics);

                    let dv = adaptive_dv(
                        &mut postsynaptic_neuron,
                        &post_if_params, 
                        input
                    );

                    dv
                };

                let post_i_syn = postsynaptic_neuron.get_neurotransmitter_currents(&post_if_params);
            
                presynaptic_neuron.current_voltage += pre_dv + pre_i_syn;
                postsynaptic_neuron.current_voltage += post_dv + post_i_syn;
        
                if !full || postsynaptic_neuron.ligand_gates.len() == 0 {
                    writeln!(file, "{}, {}", 
                        presynaptic_neuron.current_voltage,
                        postsynaptic_neuron.current_voltage,
                    ).expect("Unable to write to file");
                } else {
                    write!(file, "{}, {}", 
                        presynaptic_neuron.current_voltage, 
                        postsynaptic_neuron.current_voltage,
                    ).expect("Unable to write to file");
        
                    for i in postsynaptic_neuron.ligand_gates.iter() {
                        write!(file, ", {}, {}, {}", 
                            i.current,
                            i.neurotransmitter.r,
                            i.neurotransmitter.t,
                        ).expect("Unable to write to file");
                    }
        
                    write!(file, "\n").expect("Unable to write to file");
                }
            }
        }
    };
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

// // https://github.com/Abtinmy/computational-neuroscience/blob/main/SNN/snn.py
// // line 61
// fn update_dopamine_decay(
//     dopamine_decay: f64, 
//     weight_change: f64,
//     stdp_params: &STDPParameters, 
//     if_params: &IFParameters,
// ) -> f64 {
//     // need to multiply weight change by delta dirac of current time - last spike time pre or post
//     // pre or post is determined by same thing in update_weight, t_pre < t_post, => pre, else post
//     // reward prediction error = t post - t pre / t post
//     (-dopamine_decay / (stdp_params.tau_c - dopamine_decay) + weight_change) * if_params.dt
// }

// fn update_dopamine(
//     reward: f64, 
//     dopamine: f64, 
//     spike_time_difference: f64,
//     stdp_params: &STDPParameters, 
//     if_params: &IFParameters
// ) -> f64 {
//     // might wanna multiply tau_d by spike time difference
//     ((-dopamine / (stdp_params.tau_d - dopamine)) + reward) * if_params.dt 
// }

// weight change = weight * dopamine

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

fn update_isolated_presynaptic_neuron_weights(
    neurons: &mut Vec<Cell>,
    neuron: &Cell,
    weights: &mut Vec<f64>,
    delta_ws: &mut Vec<f64>,
    timestep: usize,
    is_spikings: Vec<bool>,
) {
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
    input_current: f64,
    excitatory_chance: f64,
    averaged: bool,
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

    get_if_params(&mut if_params, None, &stdp_table)?;

    println!("{:#?}", if_params);

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

    postsynaptic_neuron.gap_conductance = if_params.gap_conductance_init;
    postsynaptic_neuron.w_value = if_params.v_init;
    postsynaptic_neuron.w_value = if_params.w_init;
    postsynaptic_neuron.stdp_params = stdp_params.clone();
    postsynaptic_neuron.alpha = if_params.alpha_init;
    postsynaptic_neuron.beta = if_params.beta_init;
    postsynaptic_neuron.c = if_params.v_reset;
    postsynaptic_neuron.d = if_params.d_init;

    let mut neurons: Vec<Cell> = (0..n).map(|_| postsynaptic_neuron.clone())
        .collect();

    for i in neurons.iter_mut() {
        if rand::thread_rng().gen_range(0.0..=1.0) < excitatory_chance {
            i.potentiation_type = PotentiationType::Excitatory;
        } else {
            i.potentiation_type = PotentiationType::Inhibitory;
        }
    }

    let input_currents: Vec<f64> = (0..n).map(|_| input_current * limited_distr(1.0, 0.1, 0., 2.))
        .collect();

    let mut pre_fires: Vec<Option<usize>> = (0..n).map(|_| None).collect();
    let mut weights: Vec<f64> = (0..n).map( // get weights from toml and set them higher
        |_| limited_distr(
            stdp_params.weight_bayesian_params.mean, 
            stdp_params.weight_bayesian_params.std, 
            stdp_params.weight_bayesian_params.min, 
            stdp_params.weight_bayesian_params.max,
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
                let mut is_spikings: Vec<bool> = Vec::new(); 

                for (n_neuron, input_neuron) in neurons.iter_mut().enumerate() {
                    let (dv, is_spiking) = if if_params.bayesian_params.std != 0. {
                        input_neuron.get_dv_change_and_spike(
                            &if_params, 
                            input_currents[n_neuron] * limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.)
                        )
                    } else {
                        input_neuron.get_dv_change_and_spike(&if_params, input_currents[n_neuron])
                    };

                    is_spikings.push(is_spiking);

                    input_neuron.current_voltage += dv;

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }                   
                }

                let calculated_voltage: f64 = (0..n)
                    .map(
                        |i| {
                            let sign = get_sign(&neurons[i]);

                            let output = weights[i] * signed_gap_junction(&neurons[i], &postsynaptic_neuron, sign);

                            if averaged {
                                output / (n as f64)
                            } else {
                                output
                            }
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
                
                let noise_factor = limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.);
                let (dv, is_spiking) = postsynaptic_neuron.get_dv_change_and_spike(&if_params, noise_factor * calculated_voltage);                  

                update_isolated_presynaptic_neuron_weights(
                    &mut neurons, 
                    &postsynaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
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
                let mut is_spikings: Vec<bool> = Vec::new(); 

                for (n_neuron, input_neuron) in neurons.iter_mut().enumerate() {
                    let is_spiking = adaptive_apply_and_get_spike(input_neuron, &if_params);

                    let dv = if if_params.bayesian_params.std != 0. {
                        adaptive_dv(
                            input_neuron,
                            &if_params, 
                            input_currents[n_neuron] * limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.)
                        )
                    } else {
                        adaptive_dv(input_neuron, &if_params, input_currents[n_neuron])
                    };

                    is_spikings.push(is_spiking);

                    input_neuron.current_voltage += dv;

                    if is_spiking {
                        pre_fires[n_neuron] = Some(timestep);
                    }               
                }

                let is_spiking = adaptive_apply_and_get_spike(&mut postsynaptic_neuron, &if_params);

                let calculated_voltage: f64 = (0..n)
                    .map(
                        |i| {
                            let sign = get_sign(&neurons[i]);

                            let output = weights[i] * signed_gap_junction(&neurons[i], &postsynaptic_neuron, sign);

                            if averaged {
                                output / (n as f64)
                            } else {
                                output
                            }
                        }
                    ) 
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();                  

                let noise_factor = limited_distr(if_params.bayesian_params.mean, if_params.bayesian_params.std, 0., 1.);
                let dv = adaptive_dv(&mut postsynaptic_neuron, &if_params, noise_factor * calculated_voltage);

                update_isolated_presynaptic_neuron_weights(
                    &mut neurons, 
                    &postsynaptic_neuron,
                    &mut weights, 
                    &mut delta_ws, 
                    timestep, 
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
    };

    Ok(())
}

fn get_ligand_gated_channel(table: &Value, prefix_value: &str) -> Result<Vec<GeneralLigandGatedChannel>> {
    let ampa: bool = parse_value_with_default(
        table,
        format!("{}AMPA", prefix_value).as_str(), 
        parse_bool, 
        false
    )?;

    let gabaa: bool = parse_value_with_default(
        table,
        format!("{}GABAa", prefix_value).as_str(), 
        parse_bool, 
        false
    )?;

    let gabab: bool = parse_value_with_default(
        table,
        format!("{}GABAb", prefix_value).as_str(), 
        parse_bool, 
        false
    )?;

    let gabab_2: bool = parse_value_with_default(
        table,
        format!("{}GABAb (secondary)", prefix_value).as_str(), 
        parse_bool, 
        false
    )?;

    if gabab && gabab_2 {
        return Err(Error::new(ErrorKind::InvalidInput, "Cannot use 'GABAb' and 'GABAb (secondary)' simultaneously"))
    }

    let nmda: bool = parse_value_with_default(
        table,
        format!("{}NMDA", prefix_value).as_str(), 
        parse_bool, 
        false
    )?;

    let mut ligand_gates: Vec<GeneralLigandGatedChannel> = vec![];
    if ampa {
        ligand_gates.push(GeneralLigandGatedChannel::ampa_default());
    }
    if gabaa {
        ligand_gates.push(GeneralLigandGatedChannel::gabaa_default());
    }
    if gabab {
        ligand_gates.push(GeneralLigandGatedChannel::gabab_default());
    }
    if gabab_2 {
        ligand_gates.push(GeneralLigandGatedChannel::gabab_default2())
    }
    if nmda {
        let mg_conc: f64 = parse_value_with_default(
            table,
            format!("{}mg_conc", prefix_value).as_str(), 
            parse_f64, 
            BV::default().mg_conc
        )?;

        ligand_gates.push(GeneralLigandGatedChannel::nmda_with_bv(BV { mg_conc: mg_conc }));
    }

    if ligand_gates.len() != 0 {
        println!("general ligand gated channels: {}", 
            ligand_gates.iter()
                .map(|i| i.to_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
    } else {
        println!("general ligand gated channels: none")
    }

    Ok(ligand_gates)
}

fn get_additional_gates(table: &Value, prefix: &str) -> Result<Vec<AdditionalGates>> {
    let mut additional_gates: Vec<AdditionalGates> = Vec::new();

    let ltype_calcium: bool = parse_value_with_default(
        table,
        format!("{}ltype_calcium", prefix).as_str(), 
        parse_bool, 
        false
    )?;

    let hva_ltype_calcium: bool = parse_value_with_default(
        table,
        format!("{}hva_ltype_calcium", prefix).as_str(), 
        parse_bool, 
        false
    )?;

    if ltype_calcium {
        // maybe make calcium permeability editable
        additional_gates.push(AdditionalGates::LTypeCa(HighThresholdCalciumChannel::default()));
    }
    if hva_ltype_calcium {
        additional_gates.push(AdditionalGates::HVACa(HighVoltageActivatedCalciumChannel::default()))
    }

    if additional_gates.len() != 0 {
        println!("additional gated channels: {}", 
            additional_gates.iter()
                .map(|i| i.to_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
    } else {
        println!("additional gated channels: none")
    }

    Ok(additional_gates)
}

fn get_hodgkin_huxley_params(hodgkin_huxley_table: &Value, prefix: Option<&str>) -> Result<HodgkinHuxleyCell> {
    let prefix = match prefix {
        Some(prefix_value) => format!("{}_", prefix_value),
        None => String::from(""),
    };

    let v_init: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}v_init", prefix).as_str(), 
        parse_f64, 
        0.
    )?;
    println!("{}v_init: {}", prefix, v_init);

    let dt: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}dt", prefix).as_str(), 
        parse_f64, 
        0.1
    )?;
    println!("{}dt: {}", prefix, dt);

    let c_m: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}c_m", prefix).as_str(), 
        parse_f64, 
        1.
    )?;
    println!("{}c_m: {}", prefix, c_m);

    let gap_conductance: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}gap_conductance", prefix).as_str(), 
        parse_f64, 
        7.
    )?;
    println!("{}gap_conductance: {}", prefix, gap_conductance);

    let e_na: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}e_na", prefix).as_str(), 
        parse_f64, 
        115.
    )?;
    println!("{}e_na: {}", prefix, e_na);

    let e_k: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}e_k", prefix).as_str(), 
        parse_f64, 
        -12.
    )?;
    println!("{}e_k: {}", prefix, e_k);

    let e_k_leak: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}e_k_leak", prefix).as_str(), 
        parse_f64, 
        10.6
    )?;
    println!("{}e_k_leak: {}", prefix, e_k_leak);

    let g_na: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}g_na", prefix).as_str(), 
        parse_f64, 
        120.
    )?;
    println!("{}g_na: {}", prefix, g_na);

    let g_k: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}g_k", prefix).as_str(), 
        parse_f64, 
        36.
    )?;
    println!("{}g_k: {}", prefix, g_k);

    let g_k_leak: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}g_k_leak", prefix).as_str(), 
        parse_f64, 
        0.3
    )?;
    println!("{}g_k_leak: {}", prefix, g_k_leak);

    let alpha_init: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}alpha_init", prefix).as_str(), 
        parse_f64, 
        0.
    )?;
    println!("{}alpha_init: {}", prefix, alpha_init);

    let beta_init: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}beta_init", prefix).as_str(), 
        parse_f64, 
        0.
    )?;
    println!("{}beta_init: {}", prefix, beta_init);

    let state_init: f64 = parse_value_with_default(
        &hodgkin_huxley_table, 
        format!("{}state_init", prefix).as_str(), 
        parse_f64, 
        0.
    )?;
    println!("{}state_init: {}", prefix, state_init);

    let mut bayesian_params = BayesianParameters::default();
    get_bayesian_params(&mut bayesian_params, hodgkin_huxley_table, None)?;

    let gate = Gate { 
        alpha: alpha_init, 
        beta: beta_init, 
        state: state_init, 
    };

    let ligand_gates = get_ligand_gated_channel(&hodgkin_huxley_table, &prefix)?;
    let additional_gates = get_additional_gates(&hodgkin_huxley_table, &prefix)?;
    
    Ok(
        HodgkinHuxleyCell {
            current_voltage: v_init,
            gap_condutance: gap_conductance,
            dt: dt,
            c_m: c_m,
            e_na: e_na,
            e_k: e_k,
            e_k_leak: e_k_leak,
            g_na: g_na,
            g_k: g_k,
            g_k_leak: g_k_leak,
            m: gate.clone(),
            n: gate.clone(),
            h: gate,
            ligand_gates: ligand_gates,
            additional_gates: additional_gates,
            bayesian_params: bayesian_params,
        }
    )
}

fn coupled_hodgkin_huxley<'a>(
    presynaptic_neuron: &'a mut HodgkinHuxleyCell, 
    postsynaptic_neuron: &'a mut HodgkinHuxleyCell,
    input_current: f64,
    iterations: usize,
    filename: &str,
    bayesian: bool,
    full: bool,
) -> Result<()> {
    let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));

    presynaptic_neuron.initialize_parameters(presynaptic_neuron.current_voltage);
    postsynaptic_neuron.initialize_parameters(postsynaptic_neuron.current_voltage);

    write!(file, "pre_voltage,post_voltage").expect("Unable to write to file");
    
    if full && postsynaptic_neuron.ligand_gates.len() != 0{
        for i in postsynaptic_neuron.ligand_gates.iter() {
            let name = i.to_str();
            write!(file, ",g_{},r_{},T_{}", name, name, name)?;
        }
    } 
    
    write!(file, "\n").expect("Unable to write to file");
        
    for _ in 0..iterations {
        iterate_coupled_hodgkin_huxley(presynaptic_neuron, postsynaptic_neuron, bayesian, input_current);

        if !full || postsynaptic_neuron.ligand_gates.len() == 0 {
            writeln!(file, "{}, {}", 
                presynaptic_neuron.current_voltage,
                postsynaptic_neuron.current_voltage,
            ).expect("Unable to write to file");
        } else {
            write!(file, "{}, {}", 
                presynaptic_neuron.current_voltage, 
                postsynaptic_neuron.current_voltage,
            ).expect("Unable to write to file");

            for i in postsynaptic_neuron.ligand_gates.iter() {
                write!(file, ", {}, {}, {}", 
                    i.current,
                    i.neurotransmitter.r,
                    i.neurotransmitter.t,
                )?;
            }

            write!(file, "\n")?;
        }
    }

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

        let sim_params = get_simulation_parameters(&simulation_table)?;

        let output_type: String = parse_value_with_default(
            &simulation_table, 
            "output_type", 
            parse_string, 
            String::from("averaged")
        )?;
        println!("output_type: {}", output_type);

        let distance = parse_value_with_default(
            &simulation_table, 
            "eeg_distance", 
            parse_f64, 
            6.8,
        )?;
        println!("eeg_distance: {}", distance);

        let conductivity = parse_value_with_default(
            &simulation_table, 
            "eeg_conductivity", 
            parse_f64, 
            251.,
        )?;
        println!("eeg_conductivity: {}", conductivity);

        let reference_voltage = parse_value_with_default(
            &simulation_table, 
            "eeg_reference_voltage", 
            parse_f64, 
            0.007,
        )?;
        println!("eeg_reference_voltage: {}", reference_voltage);

        let output_type = Output::from_str(
            &output_type, 
            distance, 
            conductivity,
            reference_voltage,
        )?;

        let (output_value, output_graph) = run_simulation(
            sim_params.num_rows, 
            sim_params.num_cols, 
            sim_params.iterations, 
            sim_params.radius, 
            sim_params.random_volt_initialization,
            sim_params.averaged,
            sim_params.excitatory_chance,
            sim_params.if_type,
            &sim_params.if_params,
            sim_params.do_stdp,
            &sim_params.graph_params,
            &sim_params.stdp_params,
            output_type,
        )?;

        output_value.write_to_file(tag);

        if sim_params.graph_params.write_history {
            output_graph.write_history(&tag);
        } else if sim_params.graph_params.write_weights {
            output_graph.write_current_weights(&tag);
        }

        println!("Finished lattice simulation");
    // } else if let Some(ga_table) = config.get("ga") {
    //     let n_bits: usize = parse_value_with_default(&ga_table, "n_bits", parse_usize, 10)?;
    //     println!("n_bits: {}", n_bits);

    //     let n_iter: usize = parse_value_with_default(&ga_table, "n_iter", parse_usize, 100)?;
    //     println!("n_iter: {}", n_iter);

    //     let n_pop: usize = parse_value_with_default(&ga_table, "n_pop", parse_usize, 100)?;
    //     println!("n_pop: {}", n_pop);

    //     let r_cross: f64 = parse_value_with_default(&ga_table, "r_cross", parse_f64, 0.9)?;
    //     println!("r_cross: {}", r_cross);

    //     let r_mut: f64 = parse_value_with_default(&ga_table, "r_mut", parse_f64, 0.1)?;
    //     println!("r_mut: {}", r_mut);

    //     let k: usize = 3;

    //     let equation: String = parse_value_with_default(
    //         &ga_table, 
    //         "input_equation", 
    //         parse_string, 
    //         String::from("(sign * mp + x + rd * (nc^2 * y)) * z")
    //     )?;
    //     let equation: &str = equation.trim();
    //     println!("equation: {}", equation);

    //     let sim_params = get_parameters(&ga_table)?;

    //     // make sure lif_params.exp_dt = lif_params.dt
    //     // sim_params.lif_params.exp_dt = sim_params.lif_params.dt;

    //     let eeg_file: &str = match ga_table.get("eeg_file") {
    //         Some(value) => {
    //             match value.as_str() {
    //                 Some(output_value) => output_value,
    //                 None => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'eeg_file' as string")); }
    //             }
    //         },
    //         None => { return Err(Error::new(ErrorKind::InvalidInput, "Requires 'eeg_file' argument")); },
    //     };

    //     // eeg should have column specifying dt and total time
    //     let (x, dt, total_time) = read_eeg_csv(eeg_file)?;
    //     let (_faxis, sxx) = get_power_density(x, dt, total_time);

    //     let power_density_dt: f64 = parse_value_with_default(
    //         &ga_table, 
    //         "power_density_dt", 
    //         parse_f64, 
    //         dt
    //     )?;
    //     println!("power density calculation dt: {}", power_density_dt);

    //     let ga_settings = GASettings {
    //         equation: equation, 
    //         eeg: &sxx,
    //         sim_params: sim_params,
    //         power_density_dt: power_density_dt,
    //     };

    //     let mut settings: HashMap<&str, GASettings<'_>> = HashMap::new();
    //     settings.insert("ga_settings", ga_settings);

    //     let bounds_min: f64 = parse_value_with_default(&ga_table, "bounds_min", parse_f64, 0.)?;
    //     let bounds_max: f64 = parse_value_with_default(&ga_table, "bounds_max", parse_f64, 100.)?;

    //     let bounds: Vec<Vec<f64>> = (0..3)
    //         .map(|_| vec![bounds_min, bounds_max])
    //         .collect();

    //     println!("\nStarting genetic algorithm...");
    //     let (best_bitstring, best_score, _scores) = genetic_algo(
    //         objective, 
    //         &bounds, 
    //         n_bits, 
    //         n_iter, 
    //         n_pop, 
    //         r_cross,
    //         r_mut, 
    //         k, 
    //         &settings,
    //     )?;

    //     println!("Best bitstring: {}", best_bitstring.string);
    //     println!("Best score: {}", best_score);

    //     let decoded = match decode(&best_bitstring, &bounds, n_bits) {
    //         Ok(decoded_value) => decoded_value,
    //         Err(e) => return Err(e),
    //     };

    //     println!("Decoded values: {:#?}", decoded);

    //     // option to run a simulation and return the eeg signals
    //     // option to write custom bounds
    } else if let Some(single_neuron_test) = config.get("single_neuron_test") {
        let filename: String = match single_neuron_test.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);

        let iterations: usize = match single_neuron_test.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);

        let input_current: f64 = match single_neuron_test.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);  
        
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

        get_if_params(&mut if_params, None, &single_neuron_test)?;

        // let bayesian: bool = parse_value_with_default(single_neuron_test, "bayesian", parse_bool, false)?; 

        let mean_change = &if_params.bayesian_params.mean != &BayesianParameters::default().mean;
        let std_change = &if_params.bayesian_params.std != &BayesianParameters::default().std;
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
            gap_conductance: if_params.gap_conductance_init,
            potentiation_type: PotentiationType::Excitatory,
            w_value: if_params.w_init,
            stdp_params: STDPParameters::default(),
            last_firing_time: None,
            alpha: if_params.alpha_init,
            beta: if_params.beta_init,
            c: if_params.v_reset,
            d: if_params.d_init,
            ligand_gates: if_params.ligand_gates_init.clone(),
        };

        match if_type {
            IFType::Basic => { 
                test_cell.run_static_input(&if_params, input_current, bayesian, iterations, &filename); 
            },
            IFType::Adaptive => { 
                test_cell.run_adaptive_static_input(&if_params, input_current, bayesian, iterations, &filename); 
            },
            IFType::AdaptiveExponential => { 
                test_cell.run_exp_adaptive_static_input(&if_params, input_current, bayesian, iterations, &filename);
            },
            IFType::Izhikevich => { 
                test_cell.run_izhikevich_static_input(&if_params, input_current, bayesian, iterations, &filename); 
            },
            IFType::IzhikevichLeaky => {
                test_cell.run_izhikevich_leaky_static_input(&if_params, input_current, bayesian, iterations, &filename);
            },
        };

        println!("\nFinished volt test");
    } else if let Some(coupled_table) = config.get("coupled_test") {
        let if_type: String = parse_value_with_default(
            coupled_table, 
            "if_type", 
            parse_string, 
            String::from("basic")
        )?;
        println!("if_type: {}", if_type);

        let if_type = IFType::from_str(&if_type)?;

        let iterations: usize = match coupled_table.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);
    
        let filename: String = match coupled_table.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);
    
        let input_current: f64 = match coupled_table.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);

        let full: bool = parse_value_with_default(
            &coupled_table, 
            "full", 
            parse_bool, 
            false
        )?;
        println!("full: {}", full);

        let do_receptor_kinetics: bool = parse_value_with_default(
            &coupled_table, 
            "do_receptor_kinetics", 
            parse_bool, 
            false
        )?;
        println!("do_receptor_kinetics: {}", do_receptor_kinetics);

        let scaling_type_default = match if_type {
            IFType::Izhikevich | IFType::IzhikevichLeaky => "izhikevich",
            _ => "regular",
        };
        
        let scaling_type: String = parse_value_with_default(
            coupled_table, 
            "scaling_type", 
            parse_string, 
            String::from(scaling_type_default)
        )?;
        println!("scaling_type: {}", scaling_type);

        let mut pre_if_params = match scaling_type.as_str() {
            "regular" => IFParameters { ..IFParameters::default() },
            "scaled" => IFParameters { ..IFParameters::scaled_default() },
            "izhikevich" | "adaptive quadratic" => IFParameters { ..IFParameters::izhikevich_default() },
            _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown scaling")) }
        };
        
        let mut post_if_params = pre_if_params.clone();

        get_if_params(&mut pre_if_params, Some("pre"), coupled_table)?;
        println!("pre if params: {:#?}", pre_if_params);
        get_if_params(&mut post_if_params, Some("post"), coupled_table)?;
        println!("post if params: {:#?}", post_if_params);

        let pre_potentiation_type = parse_value_with_default(
            coupled_table, 
            "pre_potentiation_type", 
            parse_string, 
            String::from("excitatory")
        )?;
        let pre_potentiation_type = match pre_potentiation_type.to_ascii_lowercase().as_str() {
            "excitatory" => PotentiationType::Excitatory,
            "inhibitory" => PotentiationType::Inhibitory,
            _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown potentiation type")) }
        };     

        test_coupled_neurons(
            if_type,
            &pre_if_params, 
            &post_if_params,
            pre_potentiation_type,
            iterations,
            input_current,
            do_receptor_kinetics,
            full,
            &filename,
        );   
        
        println!("Finished coupling test");
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
    
        let input_current: f64 = match stdp_table.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);

        let excitatory_chance = parse_value_with_default(stdp_table, "excitatory_chance", parse_f64, 1.0)?;
        println!("excitatory_chance: {}", excitatory_chance);

        let averaged: bool = parse_value_with_default(stdp_table, "averaged", parse_bool, false)?;
        println!("averaged: {}", averaged);

        let mut stdp_params = STDPParameters::default();

        get_stdp_params(&mut stdp_params, stdp_table)?;

        run_isolated_stdp_test(
            stdp_table,
            &stdp_params,
            if_type,
            iterations,
            n,
            input_current,
            excitatory_chance,
            averaged,
            &filename,
        )?;

        println!("\nFinished STDP test");
    } else if let Some(hodgkin_huxley_table) = config.get("hodgkin_huxley") {
        let iterations: usize = match hodgkin_huxley_table.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);
    
        let filename: String = match hodgkin_huxley_table.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);
    
        let input_current: f64 = match hodgkin_huxley_table.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);

        let full: bool = parse_value_with_default(
            &hodgkin_huxley_table, 
            "full", 
            parse_bool, 
            false
        )?;
        println!("full: {}", full);

        let mut hodgkin_huxley = get_hodgkin_huxley_params(hodgkin_huxley_table, None)?;

        let mean_change = &hodgkin_huxley.bayesian_params.mean != &BayesianParameters::default().mean;
        let std_change = &hodgkin_huxley.bayesian_params.std != &BayesianParameters::default().std;
        let bayesian = if mean_change || std_change {
            true
        } else {
            false
        };

        hodgkin_huxley.run_static_input(input_current, bayesian, iterations, &filename, full);

        println!("\nFinished Hodgkin Huxley test");
    } else if let Some(hodgkin_huxley_peaks) = config.get("hodgkin_huxley_peaks") {
        let iterations: usize = match hodgkin_huxley_peaks.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);
    
        let filename: String = match hodgkin_huxley_peaks.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);
    
        let input_current: f64 = match hodgkin_huxley_peaks.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);

        let tolerance: f64 = match hodgkin_huxley_peaks.get("tolerance") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'tolerance' value not found")); },
        };
        println!("tolerance: {}", tolerance);

        let full: bool = parse_value_with_default(
            &hodgkin_huxley_peaks, 
            "full", 
            parse_bool, 
            false
        )?;
        println!("full: {}", full);

        let mut hodgkin_huxley = get_hodgkin_huxley_params(hodgkin_huxley_peaks, None)?;

        let mean_change = &hodgkin_huxley.bayesian_params.mean != &BayesianParameters::default().mean;
        let std_change = &hodgkin_huxley.bayesian_params.std != &BayesianParameters::default().std;
        let bayesian = if mean_change || std_change {
            true
        } else {
            false
        };

        hodgkin_huxley.peaks_test(input_current, bayesian, iterations, tolerance, &filename);

        println!("\nFinished Hodgkin Huxley peaks test");
    } else if let Some(coupled_hodgkin_huxley_table) = config.get("coupled_hodgkin_huxley") {
        let iterations: usize = match coupled_hodgkin_huxley_table.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);
    
        let filename: String = match coupled_hodgkin_huxley_table.get("filename") {
            Some(value) => parse_string(value, "filename")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'filename' value not found")); },
        };
        println!("filename: {}", filename);
    
        let input_current: f64 = match coupled_hodgkin_huxley_table.get("input_current") {
            Some(value) => parse_f64(value, "input_current")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current' value not found")); },
        };
        println!("input_current: {}", input_current);

        let bayesian: bool = parse_value_with_default(
            &coupled_hodgkin_huxley_table, 
            "bayesian", 
            parse_bool, 
            false
        )?;
        println!("bayesian: {}", bayesian);

        let full: bool = parse_value_with_default(
            &coupled_hodgkin_huxley_table, 
            "full", 
            parse_bool, 
            false
        )?;
        println!("full: {}", full);

        let mut presynaptic_neuron = get_hodgkin_huxley_params(coupled_hodgkin_huxley_table, Some("pre"))?;
        let mut postsynaptic_neuron = get_hodgkin_huxley_params(coupled_hodgkin_huxley_table, Some("post"))?;

        coupled_hodgkin_huxley(
            &mut presynaptic_neuron, 
            &mut postsynaptic_neuron,
            input_current,
            iterations,
            &filename,
            bayesian,
            full,
        )?;

        println!("\nFinished coupled Hodgkin Huxley test");
    } else if let Some(fit_neuron_models_table) = config.get("fit_neuron_models") {
        let iterations: usize = match fit_neuron_models_table.get("iterations") {
            Some(value) => parse_usize(value, "iterations")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'iterations' value not found")); },
        };
        println!("iterations: {}", iterations);

        let input_current_lower_bound: f64 = match fit_neuron_models_table.get("input_current_lower_bound") {
            Some(value) => parse_f64(value, "input_current_lower_bound")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current_lower_bound' value not found")); },
        };
        println!("input_current_lower_bound: {}", input_current_lower_bound); 

        let input_current_upper_bound: f64 = match fit_neuron_models_table.get("input_current_upper_bound") {
            Some(value) => parse_f64(value, "input_current_lower_bound")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'input_current_upper_bound' value not found")); },
        };
        println!("input_current_upper_bound: {}", input_current_upper_bound); 

        let input_current_step: f64 = parse_value_with_default(&fit_neuron_models_table, "input_current_step", parse_f64, 5.)?;
        println!("input_current_step: {}", input_current_step); 

        let tolerance: f64 = match fit_neuron_models_table.get("tolerance") {
            Some(value) => parse_f64(value, "tolerance")?,
            None => { return Err(Error::new(ErrorKind::InvalidInput, "'tolerance' value not found")); },
        };
        println!("tolerance: {}", tolerance); 

        let spike_amplitude_default: f64 = parse_value_with_default(&fit_neuron_models_table, "spike_amplitude_default", parse_f64, 0.)?;
        println!("spike_amplitude_default: {}", spike_amplitude_default); 

        let amplitude_scale_default: f64 = parse_value_with_default(&fit_neuron_models_table, "amplitude_scale_default", parse_f64, 70.)?;
        println!("amplitude_scale_default: {}", amplitude_scale_default); 

        let time_difference_scale_default: f64 = parse_value_with_default(&fit_neuron_models_table, "time_difference_scale_default", parse_f64, 800.)?;
        println!("time_difference_scale_default: {}", time_difference_scale_default); 

        let num_peaks_scale_default: f64 = parse_value_with_default(&fit_neuron_models_table, "num_peaks_scale_default", parse_f64, 10.)?;
        println!("num_peaks_scale_default: {}", num_peaks_scale_default); 

        let scaling_defaults = SummaryScalingDefaults {
            default_amplitude_scale: amplitude_scale_default,
            default_time_difference_scale: time_difference_scale_default,
            default_num_peaks_scale: num_peaks_scale_default,
        };

        let do_scaling: bool = parse_value_with_default(fit_neuron_models_table, "do_scaling", parse_bool, false)?; 
        println!("do_scaling: {}", do_scaling); 

        let use_amplitude: bool = parse_value_with_default(fit_neuron_models_table, "use_amplitude", parse_bool, false)?; 
        println!("use_amplitude: {}", use_amplitude);

        let bayesian: bool = parse_value_with_default(fit_neuron_models_table, "bayesian", parse_bool, false)?; 
        println!("bayesian: {}", bayesian); 

        let do_receptor_kinetics: bool = parse_value_with_default(&fit_neuron_models_table, "do_receptor_kinetics", parse_bool, false)?;
        println!("do_receptor_kinetics: {}", do_receptor_kinetics);

        let print_scaled: bool = parse_value_with_default(fit_neuron_models_table, "print_scaled", parse_bool, false)?; 
        println!("print_scaled: {}", print_scaled); 

        let a_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "a_lower_bound", parse_f64, 0.)?;
        println!("a_lower_bound: {}", a_lower_bound);

        let a_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "a_upper_bound", parse_f64, 0.25)?;
        println!("a_upper_bound: {}", a_upper_bound);

        let b_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "b_lower_bound", parse_f64, 0.)?;
        println!("b_lower_bound: {}", b_lower_bound);

        let b_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "b_upper_bound", parse_f64, 10.)?;
        println!("b_upper_bound: {}", b_upper_bound);

        let c_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "c_lower_bound", parse_f64, -70.)?;
        println!("c_lower_bound: {}", c_lower_bound);

        let c_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "c_upper_bound", parse_f64, 0.)?;
        println!("c_upper_bound: {}", c_upper_bound);

        let d_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "d_lower_bound", parse_f64, 0.)?;
        println!("d_lower_bound: {}", d_lower_bound);

        let d_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "d_upper_bound", parse_f64, 10.)?;
        println!("d_upper_bound: {}", d_upper_bound);

        let v_th_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "v_th_lower_bound", parse_f64, 0.)?;
        println!("v_th_lower_bound: {}", v_th_lower_bound);

        let v_th_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "v_th_upper_bound", parse_f64, 200.)?;
        println!("v_th_upper_bound: {}", v_th_upper_bound);

        let gap_conductance_lower_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "gap_conductance_lower_bound", parse_f64, 0.)?;
        println!("gap_conductance_lower_bound: {}", gap_conductance_lower_bound);

        let gap_conductance_upper_bound: f64 = parse_value_with_default(&fit_neuron_models_table, "gap_conductance_upper_bound", parse_f64, 10.)?;
        println!("gap_conductance_upper_bound: {}", gap_conductance_upper_bound);
        
        let bounds: Vec<Vec<f64>> = vec![
            vec![a_lower_bound, a_upper_bound], 
            vec![b_lower_bound, b_upper_bound], 
            vec![c_lower_bound, c_upper_bound], 
            vec![d_lower_bound, d_upper_bound], 
            vec![v_th_lower_bound, v_th_upper_bound],
            vec![gap_conductance_lower_bound, gap_conductance_upper_bound]
        ];

        let n_bits: usize = parse_value_with_default(&fit_neuron_models_table, "n_bits", parse_usize, 10)?;
        println!("n_bits: {}", n_bits);

        let n_iter: usize = parse_value_with_default(&fit_neuron_models_table, "n_iter", parse_usize, 100)?;
        println!("n_iter: {}", n_iter);

        let n_pop: usize = parse_value_with_default(&fit_neuron_models_table, "n_pop", parse_usize, 100)?;
        println!("n_pop: {}", n_pop);

        let r_cross: f64 = parse_value_with_default(&fit_neuron_models_table, "r_cross", parse_f64, 0.9)?;
        println!("r_cross: {}", r_cross);

        let r_mut: f64 = parse_value_with_default(&fit_neuron_models_table, "r_mut", parse_f64, 0.1)?;
        println!("r_mut: {}", r_mut);

        let k: usize = 3;

        let hodgkin_huxley_model = get_hodgkin_huxley_params(fit_neuron_models_table, Some("reference"))?;
        let reference_dt = hodgkin_huxley_model.dt;

        match fit_neuron_models_table.get("dt") {
            Some(_) => {
                return Err(
                    Error::new(
                        ErrorKind::InvalidInput, "Cannot have two 'dt' values, use only 'reference_dt'"
                    )
                );
            },
            None => {}
        };

        let mut if_params = IzhikevichDefault::izhikevich_default();
        get_if_params(&mut if_params, None, fit_neuron_models_table)?;

        if_params.dt = reference_dt;

        if !do_receptor_kinetics {
            if_params.ligand_gates_init.iter_mut().for_each(|i| {
                i.neurotransmitter.r = 0.8;
            });
        }

        println!("{:#?}", if_params);

        let num_steps = ((input_current_upper_bound - input_current_lower_bound) / input_current_step)
            .ceil() as usize + 1;
        let input_currents_vector: Vec<f64> = (0..num_steps)
            .map(|i| input_current_lower_bound + i as f64 * input_current_step)
            .collect();
        let input_currents: &[f64] = input_currents_vector.as_slice();

        let (hodgkin_huxley_summaries, scaling_factors) = if do_scaling {
            let mut hodgkin_huxley_summaries: Vec<ActionPotentialSummary> = vec![];
            let mut scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![];

            for current in input_currents.iter() {
                let hodgkin_huxley_summary = get_hodgkin_huxley_voltages(
                    &hodgkin_huxley_model, *current, iterations, bayesian, tolerance, spike_amplitude_default
                )?;

                let (hodgkin_huxley_summary, scaling_factors) = get_reference_scale(
                    &hodgkin_huxley_summary, &scaling_defaults
                );

                hodgkin_huxley_summaries.push(hodgkin_huxley_summary);
                scaling_factors_vector.push(Some(scaling_factors));
            }

            (hodgkin_huxley_summaries, scaling_factors_vector)
        } else {
            let mut hodgkin_huxley_summaries: Vec<ActionPotentialSummary> = vec![];
            let scaling_factors_vector: Vec<Option<SummaryScalingFactors>> = vec![None; input_currents.len()];

            for current in input_currents.iter() {
                let hodgkin_huxley_summary = get_hodgkin_huxley_voltages(
                    &hodgkin_huxley_model, *current, iterations, bayesian, tolerance, spike_amplitude_default
                )?;

                hodgkin_huxley_summaries.push(hodgkin_huxley_summary);
            }

            (hodgkin_huxley_summaries, scaling_factors_vector)            
        };
    
        let fitting_settings = FittingSettings {
            hodgkin_huxley_model: hodgkin_huxley_model,
            if_params: &if_params,
            action_potential_summary: &hodgkin_huxley_summaries.as_slice(),
            scaling_factors: &scaling_factors.as_slice(),
            use_amplitude: use_amplitude,
            spike_amplitude_default: spike_amplitude_default,
            input_currents: input_currents,
            iterations: iterations,
            bayesian: bayesian,
            do_receptor_kinetics: do_receptor_kinetics,
        };

        let mut fitting_settings_map: HashMap<&str, FittingSettings> = HashMap::new();
        fitting_settings_map.insert("settings", fitting_settings.clone());

        println!("\nStarting genetic algorithm...");
        let (best_bitstring, best_score, _scores) = genetic_algo(
            fitting_objective, 
            &bounds, 
            n_bits, 
            n_iter, 
            n_pop, 
            r_cross,
            r_mut, 
            k, 
            &fitting_settings_map,
        )?;

        println!("Best bitstring: {}", best_bitstring.string);
        println!("Best score: {}", best_score);

        let decoded = match decode(&best_bitstring, &bounds, n_bits) {
            Ok(decoded_value) => decoded_value,
            Err(e) => return Err(e),
        };

        println!("Decoded values (a, b, c, d, v_th, weight): {:#?}", decoded);

        println!("\nReference summaries:");
        if !print_scaled {
            print_action_potential_summaries(&hodgkin_huxley_summaries, &scaling_factors, use_amplitude);
        } else {
            print_action_potential_summaries(&hodgkin_huxley_summaries, &vec![None; input_currents.len()], use_amplitude);
        }

        let a: f64 = decoded[0];
        let b: f64 = decoded[1];
        let c: f64 = decoded[2];
        let d: f64 = decoded[3];
        let v_th: f64 = decoded[4];
        let gap_conductance: f64 = decoded[5];

        let mut generated_if_params: IFParameters = IzhikevichDefault::izhikevich_default();
        generated_if_params.dt = reference_dt;
        generated_if_params.v_th = v_th;

        let test_cell = Cell { 
            current_voltage: if_params.v_init, 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: gap_conductance,
            potentiation_type: PotentiationType::Excitatory,
            w_value: if_params.w_init,
            stdp_params: STDPParameters::default(),
            last_firing_time: None,
            alpha: a,
            beta: b,
            c: c,
            d: d,
            ligand_gates: if_params.ligand_gates_init.clone(),
        };

        let summaries_results = (0..fitting_settings.input_currents.len())
            .map(|i| {
                get_izhikevich_summary(
                    &mut test_cell.clone(), 
                    &mut test_cell.clone(), 
                    &if_params,
                    &fitting_settings,
                    i
                )
            })
            .collect::<Vec<Result<ActionPotentialSummary>>>();

        for result in summaries_results.iter() {
            if let Err(_) = result {
                return Err(Error::new(ErrorKind::InvalidData, "Summary calculation could not be completed"));
            }
        }

        let generated_summaries = summaries_results.into_iter().map(|res| res.unwrap())
            .collect::<Vec<ActionPotentialSummary>>();

        println!("\nGenerated summaries:");
        if !print_scaled {
            print_action_potential_summaries(&generated_summaries, &scaling_factors, use_amplitude);
        } else {
            print_action_potential_summaries(&generated_summaries, &vec![None; input_currents.len()], use_amplitude);
        }

        // match fit_neuron_models_table.get("filename") {
        //     Some(value) => {
        //         let parsed_filename = parse_string(value, "filename")?;

        //         println!("Running static test, output at {}", parsed_filename);

        //         test_cell.clone().run_izhikevich_static_input(&if_params, input_currents[0], bayesian, iterations, &parsed_filename);
                
        //         println!("Finished static test");

        //         let mut refreshed_hodgkin_huxley = get_hodgkin_huxley_params(fit_neuron_models_table, Some("reference"))?;
        //         refreshed_hodgkin_huxley.run_static_input(input_currents[0], bayesian, iterations, "output/ga_hodgkin_huxley.csv", false);
        //     },
        //     None => {},
        // };

        println!("Finished fitting");
    } else {
        return Err(Error::new(ErrorKind::InvalidInput, "Simulation config not found"));
    }

    Ok(())
}
