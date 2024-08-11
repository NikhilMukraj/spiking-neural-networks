//! A set of tools for reward based learning.

use crate::error::AgentError;


/// Agent struct to be used in environment
pub trait Agent {
    /// Applies reward and updates the agent's state
    fn update_and_apply_reward(&mut self, reward: f32) -> Result<(), AgentError>;
    /// Updates agent's state without applying a reward
    fn update(&mut self) -> Result<(), AgentError>;
}

/// Updates self based on the agent's state
pub trait State {
    type A: Agent;
    fn update_state(&mut self, agent: &Self::A) -> Result<(), AgentError>;
}

/// An encapsulation of the state and the agent
pub struct Environment<'a, T: Agent, U: State<A=T>> {
    /// Agent to do actions and respond to state
    pub agent: T,
    /// State for agent to interact with
    pub state: U,
    /// Functon that encodes the current state into the agent
    pub state_encoder: &'a dyn Fn(&U, &mut T) -> Result<(), AgentError>,
    /// Function that takes in the state and the agent to return a reward
    pub reward_function: &'a dyn Fn(&U, &T) -> Result<f32, AgentError>,
}

impl<'a, T: Agent, U: State<A=T>> Environment<'a, T, U> {
    pub fn run_with_reward(&mut self, iterations: usize) -> Result<(), AgentError> {
        for _ in 0..iterations {            
            // get reward
            let reward = (self.reward_function)(&self.state, &self.agent)?;
            // update agent (taking action) and apply reward for the action
            self.agent.update_and_apply_reward(reward)?;
            // update state based on agent
            self.state.update_state(&self.agent)?;
            // encodes state into agent
            (self.state_encoder)(&self.state, &mut self.agent)?;
        }

        Ok(())
    }

    pub fn run(&mut self, iterations: usize) -> Result<(), AgentError> {
        for _ in 0..iterations {
            // take action
            self.agent.update()?;
            // update state
            self.state.update_state(&self.agent)?;
            // encodes state into agent
            (self.state_encoder)(&self.state, &mut self.agent)?;
        }

        Ok(())
    }
}

// /// Agent to be used in unsupervised environment
// pub trait UnsupervisedAgent {
//     /// Updates agent
//     fn update(&mut self) -> Result<(), AgentError>;
// }

// /// Environment without a reward signal
// pub struct UnsupervisedEnvironment<'a, T: UnsupervisedAgent, U: State> {
//     /// Unsupervised agent
//     pub agent: T,
//     /// State of the environment
//     pub state: U,
//     /// Function to encode state into agent
//     pub state_encoder: &'a dyn Fn(&U, &mut T) -> Result<(), AgentError>,
// }
