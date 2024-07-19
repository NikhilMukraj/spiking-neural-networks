use super::iterate_and_spike::{LastFiringTime, IterateAndSpike};


/// Handles plasticity rules given the two neurons and whether to 
/// update weights based on the given neuron
pub trait Plasticity<T, U, V>: Default + Send + Sync {
    /// Calculates the change in weight given two neurons
    fn update_weight(&self, presynaptic: &T, postsynaptic: &U) -> f32;
    // Determines whether to update weights given the neuron
    fn do_update(&self, neuron: &V) -> bool;
}

/// Spike time dependent plasticity
#[derive(Debug, Clone, Copy)]
pub struct STDPlasticity {
    /// Postitive STDP modifier 
    pub a_plus: f32,
    /// Negative STDP modifier  
    pub a_minus: f32,
    /// Postitive STDP decay modifier  
    pub tau_plus: f32, 
    /// Negative STDP decay modifier 
    pub tau_minus: f32, 
}

impl Default for STDPlasticity {
    fn default() -> Self {
        STDPlasticity { 
            a_plus: 2., 
            a_minus: 2., 
            tau_plus: 45., 
            tau_minus: 45., 
        }
    }
}

impl<T, U, V> Plasticity<T, U, V> for STDPlasticity
where
    T: LastFiringTime,
    U: LastFiringTime,
    V: IterateAndSpike,
{
    fn update_weight(&self, presynaptic: &T, postsynaptic: &U) -> f32 {
        let mut delta_w: f32 = 0.;

        match (presynaptic.get_last_firing_time(), postsynaptic.get_last_firing_time()) {
            (Some(t_pre), Some(t_post)) => {
                let (t_pre, t_post): (f32, f32) = (t_pre as f32, t_post as f32);

                if t_pre < t_post {
                    delta_w = self.a_plus * (-1. * (t_pre - t_post).abs() / self.tau_plus).exp();
                } else if t_pre > t_post {
                    delta_w = -1. * self.a_minus * (-1. * (t_post - t_pre).abs() / self.tau_minus).exp();
                }
            },
            _ => {}
        };

        return delta_w;
    }

    fn do_update(&self, neuron: &V) -> bool {
        neuron.is_spiking()
    }
}
