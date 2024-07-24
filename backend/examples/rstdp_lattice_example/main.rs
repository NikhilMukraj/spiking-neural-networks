use std::{fs::File, io::Write};
use rand::Rng;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError, graph::AdjacencyMatrix, neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor}, 
        GridVoltageHistory, Optimizer, RewardModulatedLattice,
        RewardModulatedSTDP, State, TraceRSTDP
    }
};


// connect neurons within a radius of 2 with an 80% chance of connection
fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    (((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)) as f64).sqrt() <= 2. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
    x != y
}

// randomly initialize weights
fn weight_logic(_: (usize, usize), _: (usize, usize)) -> TraceRSTDP {
    TraceRSTDP {
        weight: rand::thread_rng().gen_range(0.7..=1.5),
        ..TraceRSTDP::default()
    }
}

pub struct TestState {
    pub timestep: usize,
}

type AgentType = RewardModulatedLattice<
    TraceRSTDP,
    IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
    AdjacencyMatrix<(usize, usize), TraceRSTDP>,
    GridVoltageHistory,
    RewardModulatedSTDP,
>;

impl State for TestState {
    type A = AgentType;
    fn update_state(&mut self, lattice: &Self::A) {
        self.timestep = lattice.internal_clock;
    }
}

fn reward_function(state: &TestState, _: &AgentType) -> f32 {
    if state.timestep % 1000 == 0 {
        1.
    } else {
        0.
    }
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let izhikevich_neuron = IzhikevichNeuron::default_impl();
    let mut reward_modulated_lattice = RewardModulatedLattice::default_impl();
    reward_modulated_lattice.update_graph_history = true;

    reward_modulated_lattice.populate(&izhikevich_neuron, 5, 5);
    reward_modulated_lattice.connect(&connection_conditional, &weight_logic);

    let state = TestState { timestep: 0 };
    let mut optimizer = Optimizer { 
        state: state, 
        agent: reward_modulated_lattice, 
        reward_function: &reward_function,
    };

    optimizer.run(10000)?;

    let mut file = File::create("weights.csv").expect("Could not create file");

    for matrix in optimizer.agent.graph.history {
        for row in matrix {
            for value in row {
                match value {
                    Some(trace) => write!(file, "{},", trace.weight).expect("Could not write to file"),
                    None => write!(file, "0,").expect("Could not write to file"),
                }
            }
            
            writeln!(file).expect("Could not write to file");
        }

        writeln!(file, "-----").expect("Could not write to file");
    }

    Ok(())
}
