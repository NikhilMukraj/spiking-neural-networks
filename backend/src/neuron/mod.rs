// use std::collections::HashMap;
pub mod integrate_and_fire;
pub mod hodgkin_huxley;
pub mod attractors;
pub mod spike_train;
use spike_train::{SpikeTrain, NeuralRefractoriness};
pub mod iterate_and_spike;
use iterate_and_spike::{ 
    CurrentVoltage, GapConductance, Potentiation, BayesianFactor, LastFiringTime, STDP,
    IterateAndSpike, BayesianParameters, STDPParameters, PotentiationType,
    Neurotransmitters, NeurotransmitterType, NeurotransmitterKinetics, // NeurotransmitterConcentrations,
    ApproximateNeurotransmitter, weight_neurotransmitter_concentration,
    LigandGatedChannels,
    impl_current_voltage_with_kinetics,
    impl_gap_conductance_with_kinetics,
    impl_potentiation_with_kinetics,
    impl_bayesian_factor_with_kinetics,
    impl_last_firing_time_with_kinetics,
    impl_stdp_with_kinetics,
    impl_necessary_iterate_and_spike_traits,
};
// use crate::graph::GraphFunctionality;


pub fn gap_junction<T: CurrentVoltage, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    postsynaptic_neuron.get_gap_conductance() * 
    (presynaptic_neuron.get_current_voltage() - postsynaptic_neuron.get_current_voltage())
}

pub fn signed_gap_junction<T: CurrentVoltage + Potentiation, U: CurrentVoltage + GapConductance>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    sign * gap_junction(presynaptic_neuron, postsynaptic_neuron)
}

pub fn iterate_coupled_spiking_neurons<T: IterateAndSpike>(
    presynaptic_neuron: &mut T, 
    postsynaptic_neuron: &mut T,
    do_receptor_kinetics: bool,
    bayesian: bool,
    input_current: f64,
) {
    let (t_total, post_current, input_current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let input_current = input_current * pre_bayesian_factor;

        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    } else {
        let post_current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (t_total, post_current, input_current)
    };

    let _pre_spiking = presynaptic_neuron.iterate_and_spike(input_current);

    let _post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        post_current,
        t_total.as_ref(),
    );
}

pub fn spike_train_gap_juncton<T: SpikeTrain + Potentiation, U: GapConductance>(
    presynaptic_neuron: &T,
    postsynaptic_neuron: &U,
    timestep: usize,
) -> f64 {
    let (v_max, v_resting) = presynaptic_neuron.get_height();

    if let None = presynaptic_neuron.get_last_firing_time() {
        return v_resting;
    }

    let sign = match presynaptic_neuron.get_potentiation_type() {
        PotentiationType::Excitatory => 1.,
        PotentiationType::Inhibitory => -1.,
    };

    let last_firing_time = presynaptic_neuron.get_last_firing_time().unwrap();
    let refractoriness_function = presynaptic_neuron.get_refractoriness_function();
    let dt = presynaptic_neuron.get_refractoriness_timestep();
    let conductance = postsynaptic_neuron.get_gap_conductance();

    sign * conductance * refractoriness_function.get_effect(timestep, last_firing_time, v_max, v_resting, dt)
}

pub fn iterate_coupled_spiking_neurons_and_spike_train<T: SpikeTrain, U: IterateAndSpike>(
    spike_train: &mut T,
    presynaptic_neuron: &mut U, 
    postsynaptic_neuron: &mut U,
    timestep: usize,
    do_receptor_kinetics: bool,
    bayesian: bool,
) {
    let input_current = spike_train_gap_juncton(spike_train, presynaptic_neuron, timestep);

    let (pre_t_total, post_t_total, current) = if bayesian {
        let pre_bayesian_factor = presynaptic_neuron.get_bayesian_factor();
        let post_bayesian_factor = postsynaptic_neuron.get_bayesian_factor();

        let pre_t_total = if do_receptor_kinetics {
            let mut t = spike_train.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, pre_bayesian_factor);

            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let mut t = presynaptic_neuron.get_neurotransmitter_concentrations();
            weight_neurotransmitter_concentration(&mut t, post_bayesian_factor);

            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    } else {
        let pre_t_total = if do_receptor_kinetics {
            let t = spike_train.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        let current = signed_gap_junction(
            &*presynaptic_neuron,
            &*postsynaptic_neuron,
        );

        let post_t_total = if do_receptor_kinetics {
            let t = presynaptic_neuron.get_neurotransmitter_concentrations();
            Some(t)
        } else {
            None
        };

        (pre_t_total, post_t_total, current)
    };

    let spike_train_spiking = spike_train.iterate();   
    if spike_train_spiking {
        spike_train.set_last_firing_time(Some(timestep));
    }
    
    let pre_spiking = presynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        input_current,
        pre_t_total.as_ref(),
    );
    if pre_spiking {
        presynaptic_neuron.set_last_firing_time(Some(timestep));
    }

    let post_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
        current,
        post_t_total.as_ref(),
    ); 
    if post_spiking {
        postsynaptic_neuron.set_last_firing_time(Some(timestep));
    }
}

pub fn update_weight<T: LastFiringTime, U: IterateAndSpike>(
    presynaptic_neuron: &T, 
    postsynaptic_neuron: &U
) -> f64 {
    let mut delta_w: f64 = 0.;

    match (presynaptic_neuron.get_last_firing_time(), postsynaptic_neuron.get_last_firing_time()) {
        (Some(t_pre), Some(t_post)) => {
            let (t_pre, t_post): (f64, f64) = (t_pre as f64, t_post as f64);

            if t_pre < t_post {
                delta_w = postsynaptic_neuron.get_stdp_params().a_plus * 
                    (-1. * (t_pre - t_post).abs() / postsynaptic_neuron.get_stdp_params().tau_plus).exp();
            } else if t_pre > t_post {
                delta_w = -1. * postsynaptic_neuron.get_stdp_params().a_minus * 
                    (-1. * (t_post - t_pre).abs() / postsynaptic_neuron.get_stdp_params().tau_minus).exp();
            }
        },
        _ => {}
    };

    return delta_w;
}

// fn update_isolated_presynaptic_neuron_weights<T: IterateAndSpike>(
//     neurons: &mut Vec<T>,
//     neuron: &T,
//     weights: &mut Vec<f64>,
//     delta_ws: &mut Vec<f64>,
//     timestep: usize,
//     is_spikings: Vec<bool>,
// ) {
//     for (n, i) in is_spikings.iter().enumerate() {
//         if *i {
//             neurons[n].set_last_firing_time(Some(timestep));
//             delta_ws[n] = update_weight(&neurons[n], &*neuron);
//             weights[n] += delta_ws[n];
//         }
//     }
// }

// let calculated_voltage: f64 = (0..n)
//     .map(
//         |i| {
//             let output = weights[i] * signed_gap_junction(&presynaptic_neurons[i], &*postsynaptic_neuron);

//             if averaged {
//                 output / (n as f64)
//             } else {
//                 output
//             }
//         }
//     ) 
//     .collect::<Vec<f64>>()
//     .iter()
//     .sum();
// let presynaptic_neurotransmitters: Option<NeurotransmitterConcentrations> = match do_receptor_kinetics {
//     true => Some({
//         let neurotransmitters_vec = (0..n) 
//             .map(|i| {
//                 let mut presynaptic_neurotransmitter = presynaptic_neurons[i].get_neurotransmitter_concentrations();
//                 weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, weights[i]);

//                 if averaged {
//                     weight_neurotransmitter_concentration(&mut presynaptic_neurotransmitter, (1 / n) as f64);
//                 } 

//                 presynaptic_neurotransmitter
//             }
//         ).collect::<Vec<NeurotransmitterConcentrations>>();

//         let neurotransmitters = aggregate_neurotransmitter_concentrations(&neurotransmitters_vec);

//         neurotransmitters
//     }),
//     false => None
// };

// let noise_factor = postsynaptic_neuron.get_bayesian_factor();
// let presynaptic_inputs: Vec<f64> = (0..n)
//     .map(|i| input_currents[i] * presynaptic_neurons[i].get_bayesian_factor())
//     .collect();
// let is_spikings: Vec<bool> = presynaptic_neurons.iter_mut().zip(presynaptic_inputs.iter())
//     .map(|(presynaptic_neuron, input_value)| {
//         presynaptic_neuron.iterate_and_spike(*input_value)
//     })
//     .collect();
// let is_spiking = postsynaptic_neuron.iterate_with_neurotransmitter_and_spike(
//     noise_factor * calculated_voltage,
//     presynaptic_neurotransmitters.as_ref(),
// );

// update_isolated_presynaptic_neuron_weights(
//     presynaptic_neurons, 
//     &postsynaptic_neuron,
//     &mut weights, 
//     &mut delta_ws, 
//     timestep, 
//     is_spikings,
// );

// if is_spiking {
//     postsynaptic_neuron.set_last_firing_time(Some(timestep));
//     for (n_neuron, i) in presynaptic_neurons.iter().enumerate() {
//         delta_ws[n_neuron] = update_weight(i, postsynaptic_neuron);
//         weights[n_neuron] += delta_ws[n_neuron];
//     }
// }

// pub trait GridHistory {
//     fn update<T: IterateAndSpike>(&mut self, state: &Vec<Vec<T>>);
// }

// #[derive(Debug, Clone)]
// pub struct EEGHistory {

// }

// #[derive(Debug, Clone)]
// pub struct GridVoltageHistory {

// }

// pub trait SpikeTrainHistory {
//     fn update<T: SpikeTrain>(&mut self, state: &Vec<Vec<T>>);
// }

// #[derive(Debug, Clone)]
// pub struct SpikeTrainGridHistory {

// }

// macro_rules! impl_reset_timing  {
//     () => {
//         pub fn reset_timing(&mut self) {
//             self.internal_clock = 0;
//             self.cell_grid.iter_mut()
//                 .for_each(|i| {
//                     i.iter_mut()
//                         .for_each(|j| {
//                             j.set_last_firing_time(None)
//                     })
//                 });
//         } 
//     };
// }

// should be a private method of lattice
// fn get_input_from_positions<T: IterateAndSpike, U: GraphFunctionality>(
//     cell_grid: &CellGrid<T>, 
//     graph: &U,
//     position: &Position,
//     input_positions: &HashSet<Position>, 
//     bayesian: bool,
// ) -> f64 {
//     let (x, y) = position;
//     let postsynaptic_neuron = &cell_grid[*x][*y];

//     let mut input_val = input_positions
//         .iter()
//         .map(|input_position| {
//             let (pos_x, pos_y) = input_position;
//             let input_cell = &cell_grid[*pos_x][*pos_y];

//             let final_input = signed_gap_junction(input_cell, postsynaptic_neuron);
            
//             final_input * graph.lookup_weight(&input_position, position).unwrap().unwrap()
//         })
//         .sum();

//     if bayesian {
//         input_val *= cell_grid[*x][*y].get_bayesian_factor();
//     }

//     input_val /= input_positions.len() as f64;

//     return input_val;
// }

// should be a private method of lattice
// fn get_neurotransmitter_input_from_positions<T: IterateAndSpike, U: GraphFunctionality>(
//     cell_grid: &CellGrid<T>, 
//     graph: &U,
//     position: &Position,
//     input_positions: &HashSet<Position>, 
//     bayesian: bool,
// ) -> NeurotransmitterConcentrations {
//     let input_vals = input_positions
//         .iter()
//         .map(|input_position| {
//             let (pos_x, pos_y) = input_position;
//             let input_cell = &cell_grid[*pos_x][*pos_y];

//             let mut final_input = input_cell.get_neurotransmitter_concentrations();
//             let weight = graph.lookup_weight(&input_position, position).unwrap().unwrap();
            
//             weight_neurotransmitter_concentration(&mut final_input, weight);

//             final_input
//         })
//         .collect::<Vec<NeurotransmitterConcentrations>>();

//     let mut input_val = aggregate_neurotransmitter_concentrations(&input_vals);

//     if bayesian {
//         let (x, y) = position;
//         weight_neurotransmitter_concentration(&mut input_val, cell_grid[*x][*y].get_bayesian_factor());
//     }

//     weight_neurotransmitter_concentration(&mut input_val, (1 / input_positions.len()) as f64);

//     return input_val;
// }

// // grid history should either be eeg or grid voltage for now
// // later can move to looking at neurotransmitter/receptor tracking
// // inputtable lattice
// #[derive(Debug, Clone)]
// pub struct Lattice<T: IterateAndSpike, U: GraphFunctionality, V: GridHistory> {
//     cell_grid: Vec<Vec<T>>,
//     graph: U,
//     grid_history: V,
//     update_graph_history: bool,
//     update_grid_history: bool,
//     do_stdp: bool,
//     do_receptor_kinetics: bool,
//     bayesian: bool,
//     internal_clock: usize,
// }

// updating graph history boolean should be moved to graph itself

// impl<T: IterateAndSpike, U: GraphFunctionality, V: GridHistory> Lattice<T, U, V> {
//     impl_reset_timing!();

//     // should always be averaged
//     // external inputs should be passed to this function
//     // external inputs relate desired neuron to neurons in other grid
//     fn calculate_inputs(&self) -> 
//     (HashMap<Position, f64>, Option<HashMap<Pos, NeurotransmitterConcentrations>>) {
//         let neurotransmitter_inputs = match self.do_receptor_kinetics {
//             true => {
//                 let neurotransmitters: HashMap<Position, NeurotransmitterConcentrations> = graph.get_every_node()
//                     .iter()
//                     .map(|&pos| {
//                         let input_positions = graph.get_incoming_connections(&pos).expect("Cannot find position");

//                         let neurotransmitter_input = get_neurotransmitter_input_from_positions(
//                             &cell_grid,
//                             &*graph,
//                             &pos,
//                             &input_positions,
//                             self.bayesian,
//                         );

//                         (pos, neurotransmitter_input)
//                     })
//                     .collect();
                    
//                 Some(neurotransmitters)
//             },
//             false => None,
//         };

//         // eventually convert to this
//         // let inputs: HashMap<Position, f64> = graph
//         //     .get_every_node()
//         //     .par_iter()
//         //     .map(|&pos| {
//         //     // .. calculating input
//         //     (pos, change)
//         //     });
//         //     .collect();

//         let inputs = self.graph.get_every_node()
//             .iter()
//             .map(|pos| {
//                 let input_positions = graph.get_incoming_connections(&pos).expect("Cannot find position");

//                 let input = get_input_from_positions(
//                     &cell_grid,
//                     &*graph,
//                     &pos,
//                     &input_positions,
//                     bayesian,
//                 );

//                 (pos, input)
//             });

//         (internal_inputs, neurotransmitter_inputs)
//     }

    // should have private run lattice function should just take inputs as an argument
    // one version would be just calculating the interal inputs
    // one would take the external inputs
    // and two more versions would do the same but with only electrical synapses
    // stdp weight update functionality should be abstracted to a function
//     pub fn run_lattice(
//         &mut self, 
//         iterations: usize,
//         external_inputs: HashMap<Position, Vec<f64>>, 
//         external_neurotransmitter_inputs: Option<HashMap<Position, Vec<NeurotransmitterConcentrations>>>,
//     ) -> Result<()> {
//         match (self.do_receptor_kinetics, external_neurotransmitter_inputs) {
//             (true, Some(_)) => {},
//             (true, None) => {
//                 let external_neurotransmitter_inputs: HashMap<Position, NeurotransmitterType> = HashMap::new()
//             }
//             (false, Some(_)) => {
//                 return Err(Error::new(
//                     ErrorKind::InvalidInput,
//                     "Cannot use neurotransmitter input when receptor kinetics is false"
//                 ))
//             },
//             (false, None) => {}
//         };

//         for loop_timestep in 0..iterations {
//             let timestep = loop_timestep + self.internal_clock;         
    
//             // loop through every cell
//             // modify the voltage and handle stdp
//             // end loop

//             let (inputs, neurotransmitter_inputs) = self.calculate_internal_inputs();
    
//             // could be changed to graph.get_every_node()
//             for pos in graph.get_every_node() {
//                 let (x, y) = *pos;
//                 let input_value = *inputs.get(&pos).unwrap();
    
//                 // if cloning becomes performance bottleneck
//                 // calculate bayesian factor within iterate function
//                 // apply bayesian there and to each part of neurotransmitter in update receptor kinetics
//                 // necessary to keep input hashmaps immutable for the sake of simplicity and interfacing
//                 let input_neurotransmitter = match neurotransmitter_inputs {
//                     Some(ref neurotransmitter_hashmap) => Some(neurotransmitter_hashmap.get(&pos).unwrap()),
//                     None => None,
//                 };
    
//                 // takes neurotransmitter input since input is not needed after this statement
//                 // if neurotransmitter needs to be read for some reason other than this
//                 // read neurotransmitter concentration directly from the given neuron
//                 let is_spiking = cell_grid[x][y].iterate_with_neurotransmitter_and_spike(
//                     input_value, input_neurotransmitter,
//                 );
    
//                 if is_spiking {
//                     cell_grid[x][y].set_last_firing_time(Some(timestep));
//                 }
    
//                 if do_stdp && is_spiking {
//                     let input_positions = graph.get_incoming_connections(&pos)?;
//                     for i in input_positions {
//                         let (x_in, y_in) = i;
//                         let current_weight = graph.lookup_weight(&(x_in, y_in), &pos)?.unwrap();
                                                    
//                         graph.edit_weight(
//                             &(x_in, y_in), 
//                             &pos, 
//                             Some(current_weight + update_weight(&cell_grid[x_in][y_in], &cell_grid[x][y]))
//                         )?;
//                     }
    
//                     let out_going_connections = graph.get_outgoing_connections(&pos)?;
    
//                     for i in out_going_connections {
//                         let (x_out, y_out) = i;
//                         let current_weight = graph.lookup_weight(&pos, &(x_out, y_out))?.unwrap();
    
//                         graph.edit_weight(
//                             &pos, 
//                             &(x_out, y_out), 
//                             Some(current_weight + update_weight(&cell_grid[x][y], &cell_grid[x_out][y_out]))
//                         )?; 
//                     }
//                 } 
//             }

//             if self.update_graph_history {
//                 self.graph.update_history();
//             }
//             if self.update_grid_history {
//                 self.grid_history.update(&self.cell_grid);
//             }
//         }

//         Ok(())
//     }
// }

// #[derive(Debug, Clone)]
// pub struct SpikeTrainLattice<T: SpikeTrain, U: SpikeTrainHistory> {
//     cell_grid: Vec<Vec<T>>,
//     grid_history: U,
//     update_grid_history: bool,
//     internal_clock: usize,
// }

// impl<T: SpikeTrain, U: SpikeTrainHistory> SpikeTrainLattice<T, U> {
//     impl_reset_timing!();

//     fn iterate(&mut self) {
//         self.cell_grid.iter_mut()
//             .for_each(|i| {
//                 i.iter_mut()
//                     .for_each(|j| {
//                         let is_spiking = j.iterate();
//                         if is_spiking {
//                             j.set_last_firing_time(Some(self.internal_clock))
//                         }
//                 })
//             });

//         if self.update_grid_history {
//             self.grid_history.update(&self.cell_grid);
//         }
//     }

//     pub fn run_lattice(&mut self, iterations: Option<usize>) {
//         let iterations = match iterations {
//             Some(value) => value,
//             None => 1,
//         };

//         for _ in 0..iterations {
//             self.iterate();
//         }
//     }
// }
