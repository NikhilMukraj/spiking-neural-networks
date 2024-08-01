use super::iterate_and_spike::{LastFiringTime, IterateAndSpike};


/// Handles plasticity rules given the two neurons and whether to 
/// update weights based on the given neuron
pub trait Plasticity<T, U, V>: Default + Send + Sync {
    /// Modifies the weight between given two neurons
    fn update_weight(&self, weight: &mut V, presynaptic: &T, postsynaptic: &U);
    // Determines whether to update weights given the neuron
    fn do_update(&self, neuron: &U) -> bool;
}

/// Spike time dependent plasticity rule
#[derive(Debug, Clone, Copy)]
pub struct STDP {
    /// Postitive STDP modifier 
    pub a_plus: f32,
    /// Negative STDP modifier  
    pub a_minus: f32,
    /// Postitive STDP decay modifier  
    pub tau_plus: f32, 
    /// Negative STDP decay modifier 
    pub tau_minus: f32, 
    /// Timestep
    pub dt: f32,
}

impl Default for STDP {
    fn default() -> Self {
        STDP { 
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 4.5, 
            tau_minus: 4.5, 
            dt: 0.1,
        }
    }
}

impl<T, U> Plasticity<T, U, f32> for STDP
where
    T: LastFiringTime,
    U: IterateAndSpike,
{
    fn update_weight(&self, weight: &mut f32, presynaptic: &T, postsynaptic: &U) {
        let mut delta_w: f32 = 0.;

        match (presynaptic.get_last_firing_time(), postsynaptic.get_last_firing_time()) {
            (Some(t_pre), Some(t_post)) => {
                let (t_pre, t_post): (f32, f32) = (t_pre as f32, t_post as f32);

                if t_pre < t_post {
                    delta_w = self.a_plus * (-1. * ((t_pre - t_post) * self.dt).abs() / self.tau_plus).exp();
                } else if t_pre > t_post {
                    delta_w = -1. * self.a_minus * (-1. * ((t_post - t_pre) * self.dt).abs() / self.tau_minus).exp();
                }
            },
            (None, None) => {},
            (None, Some(_)) =>  {},
            (Some(_), None) =>  {},
        };

        *weight += delta_w;
    }

    fn do_update(&self, neuron: &U) -> bool {
        neuron.is_spiking()
    }
}

/// A weight that can be reward modulated
pub trait RewardModulatedWeight: std::fmt::Debug + Clone + Copy {
    /// Get synaptic coupling factor
    fn get_weight(&self) -> f32;
}

/// Weight trace for RSTDP
#[derive(Debug, Clone, Copy)]
pub struct TraceRSTDP {
    /// Counter to find when second rolling update is applied
    pub counter: usize,
    /// Spike time dependent weight change
    pub dw: f32,
    /// Synaptic coupling factor
    pub weight: f32,
    /// Trace value
    pub c: f32,
}

impl Default for TraceRSTDP {
    fn default() -> Self {
        TraceRSTDP { counter: 0, dw: 0., weight: 0., c: 0. }
    }
}

impl TraceRSTDP {
    /// Updates trace based on weight change and current state
    pub fn update_trace(&mut self, dt: f32, tau_c: f32) {
        self.c = self.c * (-dt / tau_c).exp() + tau_c * self.dw;
    }
}

impl RewardModulatedWeight for TraceRSTDP {
    fn get_weight(&self) -> f32 {
        self.weight
    }
}

/// Handles modulation of neurons using reward
pub trait RewardModulator<T, U, V>: Default + Clone + Send + Sync {
    /// Update parameters based on reward
    fn update(&mut self, reward: f32);
    /// Update weight given two neurons and the weight itself
    fn update_weight(&self, weight: &mut V, presynaptic_neuron: &T, postsynaptic_neuron: &U);
    /// Whether to update the given neuron
    fn do_update(&self, neuron: &U) -> bool;
}

/// An implementation of spike time dependent plasticity that is reward modulated with 
/// dopamine based synapses
#[derive(Debug, Clone, Copy)]
pub struct RewardModulatedSTDP {
    // Dopamine concentration
    pub dopamine: f32,
    // Dopamine decay factor
    pub tau_d: f32,
    // Trace decay factor
    pub tau_c: f32,
    /// Postitive STDP modifier 
    pub a_plus: f32,
    /// Negative STDP modifier  
    pub a_minus: f32,
    /// Postitive STDP decay modifier  
    pub tau_plus: f32, 
    /// Negative STDP decay modifier 
    pub tau_minus: f32, 
    /// Timestep for calculating weight changes
    pub dt: f32,
}

impl Default for RewardModulatedSTDP {
    fn default() -> Self {
        RewardModulatedSTDP { 
            dopamine: 0., 
            tau_d: 20., 
            tau_c: 0.0001,
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 4.5, 
            tau_minus: 4.5, 
            dt: 0.1 
        }
    }
}

impl<T, U> RewardModulator<T, U, TraceRSTDP> for RewardModulatedSTDP 
where 
    T: LastFiringTime,
    U: IterateAndSpike,
{
    fn update(&mut self, reward: f32) {
        self.dopamine = self.dopamine * (-self.dt / self.tau_d).exp() + self.tau_d * reward;
    }

    fn update_weight(&self, weight: &mut TraceRSTDP, presynaptic: &T, postsynaptic: &U) {
        let mut delta_w: f32 = 0.;

        match (presynaptic.get_last_firing_time(), postsynaptic.get_last_firing_time()) {
            (Some(t_pre), Some(t_post)) => {
                let (t_pre, t_post): (f32, f32) = (t_pre as f32, t_post as f32);

                if t_pre < t_post {
                    delta_w = self.a_plus * (-1. * ((t_pre - t_post) * self.dt).abs() / self.tau_plus).exp();
                } else if t_pre > t_post {
                    delta_w = -1. * self.a_minus * (-1. * ((t_post - t_pre) * self.dt).abs() / self.tau_minus).exp();
                }
            },
            (None, None) => {},
            (None, Some(_)) => {},
            (Some(_), None) => {},
        };

        weight.dw += delta_w;

        if weight.counter == 0 {
            weight.counter = 1;
        } else {
            weight.update_trace(self.dt, self.tau_c);
            weight.counter = 0;
            weight.dw = 0.;
        }

        weight.weight += weight.c * self.dopamine;
    }

    fn do_update(&self, _: &U) -> bool {
        true
    }
}
