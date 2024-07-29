use std::{fs::File, io::{BufWriter, Write}};
use rand::Rng;
use spiking_neural_networks::{
    neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor}, 
        GridVoltageHistory, RewardModulatedLattice, RewardModulatedSTDP, TraceRSTDP
    },
    error::SpikingNeuralNetworksError, 
    graph::AdjacencyMatrix, 
    reinforcement::{Environment, State, Agent},
};


// connect neurons within a radius of 2 with an 80% chance of connection
fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
    ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
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
    pub dopamine_history: Vec<f32>,
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
        self.dopamine_history.push(lattice.reward_modulator.dopamine);
    }
}

fn reward_function(state: &TestState, _: &AgentType) -> f32 {
    if state.timestep % 2000 == 0 && state.timestep != 0 {
        1.
    } else {
        0.
    }
}

fn state_encoder<T: State, U: Agent>(_: &T, _: &mut U) {}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let izhikevich_neuron = IzhikevichNeuron::default_impl();
    let mut reward_modulated_lattice = RewardModulatedLattice::default_impl();
    reward_modulated_lattice.update_graph_history = true;

    reward_modulated_lattice.populate(&izhikevich_neuron, 5, 5);
    reward_modulated_lattice.connect(&connection_conditional, &weight_logic);
    reward_modulated_lattice.apply(|neuron: &mut _| {
        let mut rng = rand::thread_rng();
        neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
    });

    let state = TestState { timestep: 0, dopamine_history: vec![] };
    let mut env = Environment { 
        state, 
        agent: reward_modulated_lattice, 
        state_encoder: &state_encoder,
        reward_function: &reward_function,
    };

    env.run(10000)?;

    let mut file = BufWriter::new(File::create("weights.txt").expect("Could not create file"));

    for matrix in env.agent.graph.history {
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

    let mut file = BufWriter::new(File::create("dopamine.txt").expect("Could not create file"));

    for i in env.state.dopamine_history {
        writeln!(file, "{}", i).expect("Could not write to file");
    }

    Ok(())
}
