use std::fs::File;
use std::io::{BufWriter, Write};

use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::error::AgentError;
use spiking_neural_networks::neuron::create_agent_type_for_network;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError, 
    neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        iterate_and_spike::{IonotropicNeurotransmitterType, ApproximateNeurotransmitter, ApproximateReceptor}, 
        plasticity::{RewardModulatedSTDP, TraceRSTDP, STDP}, 
        spike_train::{DeltaDiracRefractoriness, PoissonNeuron}, 
        Lattice, RewardModulatedConnection, RewardModulatedLattice, 
        RewardModulatedLatticeNetwork, SpikeTrainLattice
    }, 
    reinforcement::{Environment, State},
};


// connect neurons within a radius of 2 with an 80% chance of connection
fn sparse_connect(x: (usize, usize), y: (usize, usize)) -> bool {
    (((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)) as f64).sqrt() <= 4. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.4 &&
    x != y
}

fn feedforward_connect(x: (usize, usize), y: (usize, usize)) -> bool {
    y.0 - x.0 == 1 
}

create_agent_type_for_network!(
    AgentType,
    STDP, 
    RewardModulatedSTDP, 
    TraceRSTDP,
    IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
    PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>,
    IonotropicNeurotransmitterType,
);

struct TestState {
    dopamine_history: Vec<f32>,
    timestep: usize,
}

impl State for TestState {
    type A = AgentType;
    fn update_state(&mut self, network: &Self::A) -> Result<(), AgentError> {
        self.timestep = network.internal_clock;
        self.dopamine_history.push(
            network.reward_modulated_lattices_values()
                .map(|i| i.reward_modulator.dopamine)
                .sum()
        );

        Ok(())
    }
}

fn reward_function(state: &TestState, _: &AgentType) -> Result<f32, AgentError> {
    let reward = if state.timestep % 2000 == 0 && state.timestep != 0 {
        1.
    } else {
        0.
    };

    Ok(reward)
}

fn state_encoder(state: &TestState, network: &mut AgentType) -> Result<(), AgentError> {
    if state.timestep % 2000 == 0 && state.timestep != 0 {
        network.spike_trains_values_mut()
            .for_each(|i| {
                i.apply(|n| {
                    n.chance_of_firing = 0.025
                })
            })
    } else if state.timestep % 2000 == 500 || state.timestep == 0 {
        network.spike_trains_values_mut()
            .for_each(|i| {
                i.apply(|n| {
                    n.chance_of_firing = 0.
                })
            })
    }

    Ok(())
}

/// Creates a skeleton for a liquid state machine and writes the history
/// of weights and voltages for the readout layer
fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = IzhikevichNeuron::default_impl();
    let base_spike_train = PoissonNeuron::default_impl();

    let mut poisson_input = SpikeTrainLattice::default_impl();
    poisson_input.populate(&base_spike_train, 1, 10);
    poisson_input.update_grid_history = true;
    poisson_input.set_id(0);

    let mut liquid = Lattice::default_impl();
    liquid.populate(&base_neuron, 10, 10);
    // could edit so weight matrix should have a spectral radius of 1
    liquid.connect(&sparse_connect, None);
    liquid.apply(|n| {
        n.current_voltage = rand::thread_rng().gen_range(n.v_init..=n.v_th);
    });
    liquid.set_id(1);

    let mut readout = RewardModulatedLattice::default_impl();
    readout.populate(&base_neuron, 4, 2);
    readout.connect(&feedforward_connect, &(
        |_, _| TraceRSTDP {
            counter: 0,
            dw: 0.,
            c: 0.,
            weight: rand::thread_rng().gen_range(0.1..=0.5),
        }
    ));
    readout.apply(|n| {
        n.current_voltage = rand::thread_rng().gen_range(n.v_init..=n.v_th);
    });
    readout.do_modulation = true;
    readout.update_graph_history = true;
    readout.update_grid_history = true;
    readout.set_id(2);

    let lattices = vec![liquid];
    let reward_modulated_lattices = vec![readout];
    let spike_trains = vec![poisson_input];

    let mut lsm: AgentType = RewardModulatedLatticeNetwork::generate_network(
        lattices, 
        reward_modulated_lattices, 
        spike_trains,
    )?;

    lsm.connect(0, 1, &(|_, _| rand::thread_rng().gen_range(0 as f32..=1.) < 0.05), None)?;
    lsm.connect_with_reward_modulation(
        1, 
        2, 
        &(|_, y| y.0 == 0 && rand::thread_rng().gen_range(0 as f32..=1.) < 0.05), 
        &(|_, _| RewardModulatedConnection::Weight(2.0))
    )?;

    let state = TestState { timestep: 0, dopamine_history: vec![] };
    let mut env = Environment { 
        state, 
        agent: lsm, 
        state_encoder: &state_encoder,
        reward_function: &reward_function,
    };

    env.run_with_reward(10000)?;

    let mut weights_file = BufWriter::new(File::create("weights.txt").expect("Could not create file"));

    for matrix in &env.agent.get_reward_modulated_lattice(&2).unwrap().graph.history {
        for row in matrix {
            for value in row {
                match value {
                    Some(trace) => write!(weights_file, "{},", trace.weight).expect("Could not write to file"),
                    None => write!(weights_file, "0,").expect("Could not write to file"),
                }
            }
            
            writeln!(weights_file).expect("Could not write to file");
        }

        writeln!(weights_file, "-----").expect("Could not write to file");
    }

    let mut voltage_file = BufWriter::new(File::create("voltage.txt").expect("Could not create file"));

    for matrix in &env.agent.get_reward_modulated_lattice(&2).unwrap().grid_history.history {
        for row in matrix {
            for value in row {
                write!(voltage_file, "{},", value).expect("Could not write to file");
            }
            
            writeln!(voltage_file).expect("Could not write to file");
        }

        writeln!(voltage_file, "-----").expect("Could not write to file");
    }

    Ok(())
}
