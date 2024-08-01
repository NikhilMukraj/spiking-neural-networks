use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError, 
    graph::{AdjacencyMatrix, GraphPosition}, 
    neuron::{
        integrate_and_fire::IzhikevichNeuron, 
        iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor}, 
        plasticity::{RewardModulatedSTDP, TraceRSTDP, STDP}, 
        spike_train::{DeltaDiracRefractoriness, PoissonNeuron}, 
        GridVoltageHistory, Lattice, RewardModulatedConnection, 
        RewardModulatedLattice, RewardModulatedLatticeNetwork, SpikeTrainGridHistory, 
        SpikeTrainLattice
    }, 
    reinforcement::{Environment, State} 
};


// connect neurons within a radius of 2 with an 80% chance of connection
fn sparse_connect(x: (usize, usize), y: (usize, usize)) -> bool {
    (((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)) as f64).sqrt() <= 4. && 
    rand::thread_rng().gen_range(0.0..=1.0) <= 0.4 &&
    x != y
}

fn feedforward_connect(x: (usize, usize), y: (usize, usize)) -> bool {
    y.1 - x.1 == 1 
}

type AgentType = RewardModulatedLatticeNetwork<
    TraceRSTDP, IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
    AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, 
    PoissonNeuron<ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
    SpikeTrainGridHistory, AdjacencyMatrix<GraphPosition, RewardModulatedConnection<TraceRSTDP>>, 
    STDP, RewardModulatedSTDP, AdjacencyMatrix<(usize, usize), TraceRSTDP>
>;

pub struct TestState {
    pub dopamine_history: Vec<f32>,
    pub timestep: usize,
}

impl State for TestState {
    type A = AgentType;
    fn update_state(&mut self, network: &Self::A) {
        self.timestep = network.internal_clock;
        self.dopamine_history.push(
            network.reward_modulated_lattices_values()
                .map(|i| i.reward_modulator.dopamine)
                .sum()
        );
    }
}

fn reward_function(state: &TestState, _: &AgentType) -> f32 {
    if state.timestep % 2000 == 0 && state.timestep != 0 {
        1.
    } else {
        0.
    }
}

fn state_encoder(state: &TestState, network: &mut AgentType) {
    if state.timestep % 2000 == 0 && state.timestep != 0 {
        network.spike_trains_values_mut()
            .for_each(|i| {
                i.apply(|n| {
                    n.chance_of_firing = 0.01
                })
            })
    } else if state.timestep % 2000 == 1 || state.timestep == 0 {
        network.spike_trains_values_mut()
            .for_each(|i| {
                i.apply(|n| {
                    n.chance_of_firing = 0.
                })
            })
    }
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let base_neuron = IzhikevichNeuron::default_impl();
    let base_spike_train = PoissonNeuron::default_impl();

    let mut poisson_input = SpikeTrainLattice::default_impl();
    poisson_input.populate(&base_spike_train, 1, 10);

    let mut liquid = Lattice::default_impl();
    liquid.populate(&base_neuron, 10, 10);
    // weight matrix should have a spectral radius of 1
    liquid.connect(&sparse_connect, None);
    liquid.apply(|n| {
        n.current_voltage = rand::thread_rng().gen_range(n.v_init..=n.v_th);
    });
    liquid.set_id(1);

    let mut readout = RewardModulatedLattice::default_impl();
    readout.populate(&base_neuron, 4, 3);
    readout.connect(&feedforward_connect, &(
        |_, _| TraceRSTDP {
            counter: 0,
            dw: 0.,
            c: 0.,
            weight: rand::thread_rng().gen_range(0.5..=2.0),
        }
    ));
    readout.apply(|n| {
        n.current_voltage = rand::thread_rng().gen_range(n.v_init..=n.v_th);
    });
    readout.do_modulation = true;
    readout.set_id(2);

    let lattices = vec![liquid];
    let reward_modulated_lattices = vec![readout];
    let spike_trains = vec![poisson_input];

    let mut lsm = RewardModulatedLatticeNetwork::generate_network(
        lattices, 
        reward_modulated_lattices, 
        spike_trains,
    )?;

    lsm.connect(0, 1, &(|_, _| rand::thread_rng().gen_range(0 as f32..=1.) < 0.05), None)?;
    lsm.connect(1, 2, &(|_, y| y.0 == 1 && rand::thread_rng().gen_range(0 as f32..=1.) < 0.05), None)?;

    let state = TestState { timestep: 0, dopamine_history: vec![] };
    let mut env = Environment { 
        state, 
        agent: lsm, 
        state_encoder: &state_encoder,
        reward_function: &reward_function,
    };

    env.run(10000)?;

    Ok(())
}
