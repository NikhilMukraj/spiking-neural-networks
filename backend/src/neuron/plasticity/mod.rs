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
            _ => {}
        };

        *weight += delta_w;
    }

    fn do_update(&self, neuron: &U) -> bool {
        neuron.is_spiking()
    }
}
