//! A set of tools for reward based learning.

use crate::error::AgentError;


/// Agent struct to be used in environment
pub trait Agent {
    /// Applies reward and updates the agent's state
    fn update_and_apply_reward(&mut self, reward: f32) -> Result<(), AgentError>;
}

/// Updates self based on the agent's state
pub trait State {
    type A: Agent;
    fn update_state(&mut self, agent: &Self::A);
}

/// An encapsulation of the state and the agent
pub struct Environment<'a, T: Agent, U: State<A=T>> {
    /// Agent to do actions and respond to state
    pub agent: T,
    /// State for agent to interact with
    pub state: U,
    /// Function that takes in the state and the agent to return a reward
    pub reward_function: &'a dyn Fn(&U, &T) -> f32,
}

impl<'a, T: Agent, U: State<A=T>> Environment<'a, T, U> {
    pub fn run(&mut self, iterations: usize) -> Result<(), AgentError> {
        for _ in 0..iterations {
            // get reward
            let reward = (self.reward_function)(&self.state, &self.agent);
            // update agent (taking action) and apply reward for the action
            self.agent.update_and_apply_reward(reward)?;
            // update state based on agent
            self.state.update_state(&self.agent);
        }

        Ok(())
    }
}
