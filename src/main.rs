use std::{
    collections::HashMap, 
    fs::read_to_string, 
    io::{Result, Error, ErrorKind}, 
    env,
};
use rand::{Rng, seq::SliceRandom};
use rand_distr::{Normal, Distribution};
use toml::{from_str, Value};
use exprtk_rs::{Expression, SymbolTable};
use ndarray::Array1;
mod eeg;
use eeg::{read_eeg_csv, get_power_density, power_density_comparison};
mod ga;
use ga::{BitString, decode, genetic_algo};


#[derive(Debug, Clone)]
struct LIFParameters {
    v_th: f64,
    v_reset: f64,
    tau_m: f64,
    g_l: f64,
    v_init: f64,
    e_l: f64,
    tref: f64,
    w_init: f64,
    a: f64,
    b: f64,
    dt: f64,
    exp_dt: f64,
    bayesian_mean: f64,
    bayesian_std: f64,
    bayesian_max: f64,
    bayesian_min: f64,
    // total_time: f64,
}

impl Default for LIFParameters {
    fn default() -> Self {
        LIFParameters { 
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            a: 6., // arbitrary a value
            b: 10., // arbitrary b value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: 1.0, // center of norm distr
            bayesian_std: 0.0, // std of norm distr
            bayesian_max: 2.0, // maximum cutoff for norm distr
            bayesian_min: 0.0, // minimum cutoff for norm distr
        }
    }
}

pub trait ScaledDefault {
    fn scaled_default() -> Self;
}

impl ScaledDefault for LIFParameters {
    fn scaled_default() -> Self {
        LIFParameters { 
            v_th: 1., // spike threshold (mV)
            v_reset: 0., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            g_l: 4.25, // leak conductance (nS) ((10 - (-75)) / ((-55) - (-75))) * (1 - 0)) + 1
            v_init: 0., // initial potential (mV)
            e_l: 0., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            a: 6., // arbitrary a value
            b: 10., // arbitrary b value
            dt: 0.1, // simulation time step (ms)
            exp_dt: 1., // exponential time step (ms)
            bayesian_mean: 1.0, // center of norm distr
            bayesian_std: 0.0, // std of norm distr
            bayesian_max: 2.0, // maximum cutoff for norm distr
            bayesian_min: 0.0, // minimum cutoff for norm distr
        }
    }
}

#[derive(Clone, Debug)]
enum LIFType {
    Basic,
    Adaptive,
    AdaptiveExponentatial,
}

impl LIFType {
    fn from_str(string: &str) -> Result<LIFType> {
        let output = match string.to_ascii_lowercase().as_str() {
            "basic" => { LIFType::Basic },
            "adaptive" => { LIFType::Adaptive },
            "adaptive exponential" => { LIFType::AdaptiveExponentatial },
            _ => { return Err(Error::new(ErrorKind::InvalidInput, "Unknown string")); }
        };

        Ok(output)
    }
}

#[derive(Clone)]
enum PotentiationType {
    Excitatory,
    Inhibitory,
}

impl PotentiationType {
    fn weighted_random_type(prob: f64) -> PotentiationType {
        if rand::thread_rng().gen_range(0.0..=1.0) <= prob {
            PotentiationType::Excitatory
        } else {
            PotentiationType::Inhibitory
        }
    }
}

#[derive(Clone)]
struct Cell {
    current_voltage: f64, // membrane potential
    refractory_count: f64, // keeping track of refractory period
    leak_constant: f64, // leak constant gene
    integration_constant: f64, // integration constant gene
    potentiation_type: PotentiationType,
    neurotransmission_concentration: f64, // concentration of neurotransmitter in synapse
    neurotransmission_release: f64, // concentration of neurotransmitter released at spiking
    receptor_density: f64, // factor of how many receiving receptors for a given neurotransmitter
    chance_of_releasing: f64, // chance cell can produce neurotransmitter
    dissipation_rate: f64, // how quickly neurotransmitter concentration decreases
    chance_of_random_release: f64, // likelyhood of neuron randomly releasing neurotransmitter
    random_release_concentration: f64, // how much neurotransmitter is randomly released
    w_value: f64, // adaptive value 
}

type CellGrid = Vec<Vec<Cell>>;

impl Cell {
    fn get_dv_change_and_spike(&mut self, lif: &LIFParameters, i: f64) -> (f64, bool) {
        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = lif.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = lif.v_reset;
            self.refractory_count = lif.tref / lif.dt
        }

        // let dv = (-1. * (self.current_voltage - lif.e_l) + i / lif.g_l) * (lif.dt / lif.tau_m);
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (self.integration_constant * (i / lif.g_l))
        ) * (lif.dt / lif.tau_m);
        // could be varied with a leak constant instead of -1 *
        // input could be varied with a integration constant times the input

        return (dv, is_spiking);
    }

    fn apply_dw_change_and_get_spike(&mut self, lif: &LIFParameters) -> bool {
        // dw = (self.a * (v[it]-self.V_L) - w[it]) * (self.dt/self.tau_m)
        let dw = (
            lif.a * (self.current_voltage - lif.e_l) -
            self.w_value
        ) * (lif.dt / lif.tau_m);

        self.w_value += dw;

        let mut is_spiking = false;

        if self.refractory_count > 0. {
            self.current_voltage = lif.v_reset;
            self.refractory_count -= 1.;
        } else if self.current_voltage >= lif.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = lif.v_reset;
            self.w_value += lif.b;
            self.refractory_count = lif.tref / lif.dt
        }

        return is_spiking;
    }

    fn adaptive_get_dv_change(&mut self, lif: &LIFParameters, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (self.integration_constant * (i / lif.g_l)) - 
            (self.w_value / lif.g_l)
        ) * (lif.dt / lif.tau_m);

        dv
    }

    fn exp_adaptive_get_dv_change(&mut self, lif: &LIFParameters, i: f64) -> f64 {
        let dv = (
            (self.leak_constant * (self.current_voltage - lif.e_l)) +
            (lif.exp_dt * ((self.current_voltage - lif.v_th) / lif.exp_dt).exp()) +
            (self.integration_constant * (i / lif.g_l)) - 
            (self.w_value / lif.g_l)
        ) * (lif.dt / lif.tau_m);

        dv
    }

    fn determine_neurotransmitter_concentration(&mut self, is_spiking: bool) {
        // (excitatory should increase voltage)
        // (inhibitory should decrease voltage)
        // (may also depend on kind of receptor)
        let prob = rand::thread_rng().gen_range(0.0..=1.0);
        if is_spiking && (prob <= self.chance_of_releasing) {
            self.neurotransmission_concentration += self.neurotransmission_release;
        } else if self.neurotransmission_concentration > 0. {
            let concentration = (
                    self.neurotransmission_concentration - self.dissipation_rate
                )
                .max(0.0); // reduce concentration until 0
            self.neurotransmission_concentration = concentration;
        }
        
        let prob = rand::thread_rng().gen_range(0.0..=1.0);
        if self.refractory_count <= 0. && prob <= self.chance_of_random_release {
            self.neurotransmission_concentration += self.random_release_concentration;
        }
    }

    // voltage of cell should be initial voltage + this change
    // fn run_static_input(&mut self, lif: &LIFParameters, i: f64, bayesian: bool, iterations: usize, filename: &str) {
    //     let mut file = File::create(filename)
    //         .expect("Unable to create file");
    //     writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");

    //     for _ in 0..iterations {
    //         let (dv, _is_spiking) = if bayesian {
    //             self.get_dv_change_and_spike(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
    //         } else {
    //             self.get_dv_change_and_spike(lif, i)
    //         };
    //         self.current_voltage += dv;

    //         writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
    //     }
    // }

    // fn run_adaptive_static_input(&mut self, lif: &LIFParameters, i: f64, bayesian: bool, iterations: usize, filename: &str) {
    //     let mut file = File::create(filename)
    //         .expect("Unable to create file");
    //     writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
    //     for _ in 0..iterations {
    //         let _is_spiking = self.apply_dw_change_and_get_spike(lif);
    //         let dv = if bayesian {
    //             self.adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
    //         } else {
    //             self.adaptive_get_dv_change(lif, i)
    //         };
    //         self.current_voltage += dv;

    //         writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
    //     }
    // }

    // fn run_exp_adaptive_static_input(&mut self, lif: &LIFParameters, i: f64, bayesian: bool, iterations: usize, filename: &str) {
    //     let mut file = File::create(filename)
    //         .expect("Unable to create file");
    //     writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        
    //     for _ in 0..iterations {
    //         let _is_spiking = self.apply_dw_change_and_get_spike(lif);
    //         let dv = if bayesian {
    //             self.exp_adaptive_get_dv_change(lif, i * limited_distr(lif.bayesian_mean, lif.bayesian_std, 0., 1.))
    //         } else {
    //             self.exp_adaptive_get_dv_change(lif, i)
    //         };
    //         self.current_voltage += dv;

    //         writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
    //     }
    // }
}

fn positions_within_square(
    center_row: usize, 
    center_col: usize, 
    extent: usize, 
    size: (usize, usize)
) -> Vec<(usize, usize)> {
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

fn randomly_select_positions(mut positions: Vec<(usize, usize)>, num_to_select: usize) -> Vec<(usize, usize)> {
    let mut rng = rand::thread_rng();

    positions.shuffle(&mut rng);
    positions.truncate(num_to_select);

    positions
}

fn limited_distr(mean: f64, std_dev: f64, minimum: f64, maximum: f64) -> f64 {
    if std_dev == 0.0 {
        return mean;
    }

    let normal = Normal::new(mean, std_dev).unwrap();
    let output: f64 = normal.sample(&mut rand::thread_rng());
   
    output.max(minimum).min(maximum)
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

fn get_input_from_positions(
    cell_grid: &CellGrid, 
    input_positions: &Vec<(usize, usize)>, 
    input_calculation: &mut dyn FnMut(f64, f64, f64, f64) -> f64,
    bayesian_params: Option<&LIFParameters>,
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

            // could weight certain connections alongside adjacency list
        })
        .sum();

    match bayesian_params {
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

    return input_val;
}


// #[derive(Debug)]
// #[allow(dead_code)]
// struct NeuroAndVoltage {
//     neurotransmitter_concentration: f64,
//     voltage: f64,
// }

enum Output {
    Grid(Vec<CellGrid>),
    Averaged(Vec<f64>)
}

impl Output {
    fn add(&mut self, cell_grid: &CellGrid) {
        match self {
            Output::Grid(grids) => { grids.push(cell_grid.clone()) }
            Output::Averaged(averages) => { 
                // averages.push(NeuroAndVoltage {
                //     neurotransmitter_concentration: get_neuro_avg(cell_grid),
                //     voltage: get_volt_avg(cell_grid)
                // });

                averages.push(get_volt_avg(cell_grid));
            }
        }
    }

    fn from_str(string: &str) -> Result<Output> {
        match string.to_ascii_lowercase().as_str() {
            "grid" => { Ok(Output::Grid(Vec::<CellGrid>::new())) },
            "averaged" => { Ok(Output::Averaged(Vec::<f64>::new())) },
            _ => { Err(Error::new(ErrorKind::InvalidInput, "Unknown output type")) }
        }
    }
}

type AdjacencyList = HashMap<(usize, usize), Vec<(usize, usize)>>;

fn run_simulation(
    num_rows: usize, 
    num_cols: usize, 
    iterations: usize, 
    radius: usize, 
    random_volt_initialization: bool,
    lif_type: LIFType,
    lif_params: &LIFParameters,
    default_cell_values: &HashMap<&str, f64>,
    input_calculation: &mut dyn FnMut(f64, f64, f64, f64) -> f64,
    mut output_val: Output,
) -> Result<Output> {
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

    let mean_change = &lif_params.bayesian_mean != &LIFParameters::default().bayesian_mean;
    let std_change = &lif_params.bayesian_std != &LIFParameters::default().bayesian_std;
    let bayesian = if mean_change || std_change {
        Some(lif_params)
    } else {
        None
    };

    let mut cell_grid: CellGrid = (0..num_rows)
        .map(|_| {
            (0..num_cols)
                .map(|_| Cell { 
                    current_voltage: lif_params.v_init, 
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
                    w_value: lif_params.w_init,
                })
                .collect::<Vec<Cell>>()
        })
        .collect::<CellGrid>();

    let mut rng = rand::thread_rng();

    if random_volt_initialization {
        for section in &mut cell_grid {
            for neuron in section {
                neuron.current_voltage = rng.gen_range(lif_params.v_init..=lif_params.v_th);
            }
        }
    }

    let mut adjacency_list: AdjacencyList = HashMap::new(); 
    
    for row in 0..num_rows {
        for col in 0..num_cols {
            let positions = positions_within_square(row, col, radius, (num_rows, num_cols));
            let num_to_select = rng.gen_range(1..positions.len());
            let positions = randomly_select_positions(positions, num_to_select);
            adjacency_list
                .entry((row, col))
                .or_insert(positions);
        }
    }

    match lif_type {
        LIFType::Basic => {
            for _ in 0..iterations {
                let mut changes: HashMap<(usize, usize), (f64, bool)> = adjacency_list.keys()
                    .cloned()
                    .map(|key| (key, (0.0, false)))
                    .collect();

                // loop through every cell
                // calculate the dv given the inputs
                // write 
                // end loop

                for pos in adjacency_list.keys() {
                    let (x, y) = pos;
                    let input_positions = adjacency_list.get(&pos).unwrap();

                    
                    let input = get_input_from_positions(&cell_grid, input_positions, input_calculation, bayesian);
                    let (dv, is_spiking) = cell_grid[*x][*y].get_dv_change_and_spike(lif_params, input);

                    changes.insert(*pos, (dv, is_spiking));
                }

                // loop through every cell
                // modify the voltage
                // end loop

                for (pos, (dv_value, is_spiking_value)) in changes {
                    let (x, y) = pos;
                    
                    cell_grid[x][y].determine_neurotransmitter_concentration(is_spiking_value);
                    cell_grid[x][y].current_voltage += dv_value;
                }

                // repeat until simulation is over

                output_val.add(&cell_grid);
            }
        },
        LIFType::Adaptive => {
            for _ in 0..iterations {
                let mut changes: HashMap<(usize, usize), (f64, bool)> = adjacency_list.keys()
                    .cloned()
                    .map(|key| (key, (0.0, false)))
                    .collect();

                // loop through every cell
                // calculate the dv given the inputs
                // write 
                // end loop

                for pos in adjacency_list.keys() {
                    let (x, y) = pos;
                    let input_positions = adjacency_list.get(&pos).unwrap();

                    let input = get_input_from_positions(&cell_grid, input_positions, input_calculation, bayesian);
                    let is_spiking = cell_grid[*x][*y].apply_dw_change_and_get_spike(lif_params);

                    changes.insert(*pos, (input, is_spiking));
                }

                // find dv change and apply it
                // find neurotransmitter change and apply it
                for (pos, (input_value, is_spiking_value)) in changes {
                    let (x, y) = pos;

                    let dv = cell_grid[x][y].adaptive_get_dv_change(lif_params, input_value);

                    cell_grid[x][y].determine_neurotransmitter_concentration(is_spiking_value);
                    cell_grid[x][y].current_voltage += dv;
                }

                output_val.add(&cell_grid);
            }
        },
        LIFType::AdaptiveExponentatial => {
            for _ in 0..iterations {
                let mut changes: HashMap<(usize, usize), (f64, bool)> = adjacency_list.keys()
                    .cloned()
                    .map(|key| (key, (0.0, false)))
                    .collect();

                // loop through every cell
                // calculate the dv given the inputs
                // write 
                // end loop

                for pos in adjacency_list.keys() {
                    let (x, y) = pos;
                    let input_positions = adjacency_list.get(&pos).unwrap();

                    let input = get_input_from_positions(&cell_grid, input_positions, input_calculation, bayesian);

                    let is_spiking = cell_grid[*x][*y].apply_dw_change_and_get_spike(lif_params);

                    changes.insert(*pos, (input, is_spiking));
                }

                // find dv change and apply it
                // find neurotransmitter change and apply it
                for (pos, (input_value, is_spiking_value)) in changes {
                    let (x, y) = pos;

                    let dv = cell_grid[x][y].exp_adaptive_get_dv_change(lif_params, input_value);

                    cell_grid[x][y].determine_neurotransmitter_concentration(is_spiking_value);
                    cell_grid[x][y].current_voltage += dv;
                }

                output_val.add(&cell_grid);
            }
        },
    }

    return Ok(output_val);
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
    lif_params: LIFParameters,
    lif_type: LIFType,
    default_cell_values: HashMap<&'a str, f64>,
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

    let lif_type: String = parse_value_with_default(table, "lif_type", parse_string, String::from("basic"))?;

    let lif_type = match LIFType::from_str(&lif_type) {
        Ok(lif_type_val) => lif_type_val,
        Err(_e) => { return Err(Error::new(ErrorKind::InvalidInput, "Cannot parse 'lif_type' as one of the valid types")) }
    };
    println!("lif_type: {:#?}", lif_type);

    let output_type: String = parse_value_with_default(table, "output_type", parse_string, String::from("averaged"))?;
    println!("output_type: {}", output_type);

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

    let mut lif_params = LIFParameters {
        ..LIFParameters::default()
    };

    lif_params.dt = parse_value_with_default(table, "dt", parse_f64, lif_params.dt)?;
    lif_params.exp_dt = parse_value_with_default(table, "exp_dt", parse_f64, lif_params.exp_dt)?;
    lif_params.tau_m = parse_value_with_default(table, "tau_m", parse_f64, lif_params.tau_m)?;
    lif_params.tref = parse_value_with_default(table, "tref", parse_f64, lif_params.tref)?;
    lif_params.a = parse_value_with_default(table, "a", parse_f64, lif_params.a)?;
    lif_params.b = parse_value_with_default(table, "b", parse_f64, lif_params.b)?;
    lif_params.bayesian_mean = parse_value_with_default(table, "bayesian_mean", parse_f64, lif_params.bayesian_mean)?;
    lif_params.bayesian_std = parse_value_with_default(table, "bayesian_std", parse_f64, lif_params.bayesian_std)?;
    lif_params.bayesian_max = parse_value_with_default(table, "bayesian_max", parse_f64, lif_params.bayesian_max)?;
    lif_params.bayesian_min = parse_value_with_default(table, "bayesian_min", parse_f64, lif_params.bayesian_min)?;

    println!("{:#?}", lif_params);

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
        (None, Some(total_time_value)) => { (total_time_value as f64 / lif_params.dt) as usize },
        (None, None) => { return Err(Error::new(ErrorKind::InvalidInput, "Missing 'iterations' or 'total_time' argument")); },
    };
    println!("iterations: {}\n", iterations);

    return Ok(SimulationParameters {
        num_rows: num_rows, 
        num_cols: num_cols, 
        iterations: iterations, 
        radius: radius, 
        random_volt_initialization: random_volt_initialization,
        lif_params: lif_params,
        lif_type: lif_type,
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

    let output_value = run_simulation(
        sim_params.num_rows, 
        sim_params.num_cols, 
        sim_params.iterations, 
        sim_params.radius, 
        sim_params.random_volt_initialization,
        sim_params.lif_type,
        &sim_params.lif_params,
        &sim_params.default_cell_values,
        &mut input_func,
        Output::Averaged(vec![]),
    )?;

    let x: Vec<f64> = match output_value {
        Output::Averaged(value) => { value },
        _ => { unreachable!() },
    };

    let total_time: f64 = sim_params.iterations as f64 * sim_params.lif_params.dt;
    // let (_faxis, sxx) = get_power_density(x, sim_params.lif_params.dt, total_time);
    let (_faxis, sxx) = get_power_density(x, power_density_dt, total_time);
    let score = power_density_comparison(eeg, &sxx)?;

    return Ok(score);
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Requires .toml argument file");
        return Err(Error::new(ErrorKind::InvalidInput, "Requires .toml argument file"));
    }

    let toml_content = read_to_string(&args[1]).expect("Cannot read file");
    let config: Value = from_str(&toml_content).expect("Cannot read config");

    if let Some(simulation_table) = config.get("simulation") {
        let output_type: String = parse_value_with_default(
            &simulation_table, 
            "output_type", 
            parse_string, 
            String::from("averaged")
        )?;
        println!("output_type: {}", output_type);

        let output_type = Output::from_str(&output_type)?;

        let equation: String = parse_value_with_default(
            &simulation_table, 
            "input_equation", 
            parse_string, 
            String::from("sign * mp + 100 + rd * (nc^2 * 200)")
        )?;
        let equation: &str = equation.trim();
        println!("equation: {}", equation);
    
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

        println!("here");

        let sim_params = get_parameters(&simulation_table)?;

        let output_value = run_simulation(
            sim_params.num_rows, 
            sim_params.num_cols, 
            sim_params.iterations, 
            sim_params.radius, 
            sim_params.random_volt_initialization,
            sim_params.lif_type,
            &sim_params.lif_params,
            &sim_params.default_cell_values,
            &mut input_func,
            output_type,
        )?;

        match output_value {
            Output::Grid(grid_vec) => {
                let voltage_matrix = grid_vec.last().expect("Cannot get last value");

                for row in voltage_matrix {
                    for neuron in row {
                        print!("{:.3} ", neuron.current_voltage);
                    }
                    println!();
                }
            }
            Output::Averaged(averaged_vec) => {
                // println!("{:?}", averaged_vec.last().expect("Cannot get last value"));
                println!("{:#?}", averaged_vec);
            }
        }
    } else if let Some(ga_table) = config.get("ga") {
        let n_bits: usize = parse_value_with_default(&ga_table, "n_bits", parse_usize, 10)?;
        println!("n_bits: {}", n_bits);

        let n_iter: usize = parse_value_with_default(&ga_table, "n_iter", parse_usize, 100)?;
        println!("n_iter: {}", n_iter);

        let n_pop: usize = parse_value_with_default(&ga_table, "n_pop", parse_usize, 100)?;
        println!("n_pop: {}", n_pop);

        let r_cross: f64 = parse_value_with_default(&ga_table, "r_cross", parse_f64, 0.4)?;
        println!("r_cross: {}", r_cross);

        let r_mut: f64 = parse_value_with_default(&ga_table, "r_mut", parse_f64, 0.4)?;
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
        // option to use different dts 
        // (option to use eeg dt in power density calculation and nowhere else)
    } else {
        return Err(Error::new(ErrorKind::InvalidInput, "Simulation config not found"));
    }

    Ok(())
}
