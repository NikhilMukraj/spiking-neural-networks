//! A series of ion channels and ion channel traits to implement for usage in neuron models

#[cfg(feature = "gpu")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "gpu")]
use super::iterate_and_spike::{BufferGPU, BufferType, AvailableBufferType};
#[cfg(feature = "gpu")]
use crate::error::GPUError;
#[cfg(feature = "gpu")]
use opencl3::{command_queue::CommandQueue, context::Context};

/// A gating variable for necessary ion channels
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicGatingVariable {
    /// Gating variable
    pub alpha: f32,
    /// Gating variable
    pub beta: f32,
    /// Current state of the gate
    pub state: f32,
}

impl Default for BasicGatingVariable {
    fn default() -> Self {
        BasicGatingVariable {
            alpha: 0.,
            beta: 0.,
            state: 0.,
        }
    }
}

impl BasicGatingVariable {
    /// Initializes the gating variable state
    pub fn init_state(&mut self) {
        self.state = self.alpha / (self.alpha + self.beta);
    }

    // Updates the gating variable based on a given timestep (ms)
    pub fn update(&mut self, dt: f32) {
        let alpha_state: f32 = self.alpha * (1. - self.state);
        let beta_state: f32 = self.beta * self.state;
        self.state += dt * (alpha_state - beta_state);
    }
}

// CHECK THIS PAPER TO CREATE MORE ION CHANNELS WHEN REFACTORING
// https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/25282547/

// https://webpages.uidaho.edu/rwells/techdocs/Biological%20Signal%20Processing/Chapter%2004%20The%20Biological%20Neuron.pdf

// https://www.nature.com/articles/356441a0.pdf : calcium currents paper
// https://github.com/ModelDBRepository/151460/blob/master/CaT.mod // low threshold calcium current
// https://modeldb.science/279?tab=1 // low threshold calcium current (thalamic)
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod // high threshold calcium current (l type)
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9373714/ // assume [Ca2+]in,inf is initial [Ca2+] value

/// Handles dynamics of an ion channel that does not need timestep information
/// 
/// Example implementation:
/// ```rust
/// # use spiking_neural_networks::neuron::ion_channels::TimestepIndependentIonChannel;
/// #
/// //// A potassium leak channel
/// #[derive(Debug, Clone, Copy)]
/// pub struct KLeakChannel {
///     /// Maximal conductance (nS)
///     pub g_k_leak: f32,
///     /// Reversal potential (mV)
///     pub e_k_leak: f32,
///     /// Current output
///     pub current: f32,
/// }
/// 
/// impl TimestepIndependentIonChannel for KLeakChannel {
///     fn update_current(&mut self, voltage: f32) {
///         self.current = self.g_k_leak * (voltage - self.e_k_leak);
///     }
/// 
///     fn get_current(&self) -> f32 {
///         self.current
///     }
/// }
/// ```
pub trait TimestepIndependentIonChannel: Clone + Sync + Send {
    /// Updates current based on the current voltage (mV)
    fn update_current(&mut self, voltage: f32);
    /// Returns the current
    fn get_current(&self) -> f32;
}

/// Handles dynamics of an ion channel
/// 
/// Example implementation:
/// ```rust
/// # use spiking_neural_networks::neuron::ion_channels::{BasicGatingVariable, IonChannel};
/// #
/// /// An implementation of a calcium channel
/// #[derive(Debug, Clone, Copy)]
/// pub struct CalciumIonChannel {
///     /// Maximal conductance of the channel (nS)
///     pub g_ca: f32,
///     /// Reversal potential (mV)
///     pub e_ca: f32,
///     /// Gating variable
///     pub s: BasicGatingVariable,
///     /// Current generated by channel
///     pub current: f32,
/// }
/// 
/// impl CalciumIonChannel {
///     /// Updates the state of the gating variable based on voltage
///     fn update_gate_states(&mut self, voltage: f32) {
///         self.s.alpha = 1.6 / (1. + (-0.072 * (voltage - 5.)).exp());
///         self.s.beta = (0.02 * (voltage + 8.9)) / (((voltage + 8.9).exp() / 5.) - 1.);
///     }
/// }
/// 
/// impl IonChannel for CalciumIonChannel {
///     fn update_current(&mut self, voltage: f32, dt: f32) {
///         self.update_gate_states(voltage);
/// 
///         self.s.update(dt);
/// 
///         self.current = -self.s.state.powf(2.) * self.g_ca * (voltage - self.e_ca);
///     }
/// 
///     fn get_current(&self) -> f32 {
///         self.current
///     }
/// }
/// ```
pub trait IonChannel: Clone + Sync + Send {
    /// Updates current based on the current voltage (mV) and a timestep (ms)
    fn update_current(&mut self, voltage: f32, dt: f32);
    /// Returns the current
    fn get_current(&self) -> f32;
}

/// An implementation of a calcium channel
#[derive(Debug, Clone, Copy)]
pub struct CalciumIonChannel {
    /// Maximal conductance of the channel (nS)
    pub g_ca: f32,
    /// Reversal potential (mV)
    pub e_ca: f32,
    /// Gating variable
    pub s: BasicGatingVariable,
    /// Current generated by channel
    pub current: f32,
}

impl Default for CalciumIonChannel {
    fn default() -> Self {
        CalciumIonChannel {
            g_ca: 0.025, // https://www.ncbi.nlm.nih.gov/books/NBK6181/
            e_ca: 80.,
            s: BasicGatingVariable::default(),
            current: 0.,
        }
    }
}

// https://github.com/ModelDBRepository/121060/blob/master/chan_CaL12.mod
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod
impl CalciumIonChannel {
    /// Updates the state of the gating variable based on voltage
    fn update_gate_states(&mut self, voltage: f32) {
        self.s.alpha = 1.6 / (1. + (-0.072 * (voltage - 5.)).exp());
        self.s.beta = (0.02 * (voltage + 8.9)) / (((voltage + 8.9).exp() / 5.) - 1.);
    }
}

// https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/25282547/
// https://link.springer.com/referenceworkentry/10.1007/978-1-4614-7320-6_230-1
impl IonChannel for CalciumIonChannel {
    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.update_gate_states(voltage);

        self.s.update(dt);

        self.current = -self.s.state.powf(2.) * self.g_ca * (voltage - self.e_ca);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// A sodium ion channel
#[derive(Debug, Clone, Copy)]
pub struct NaIonChannel {
    /// Maximal conductance (nS)
    pub g_na: f32,
    /// Reversal potential (mV)
    pub e_na: f32,
    /// Gating variable that changes based on voltage
    pub m: BasicGatingVariable,
    /// Gating variable that changes based on voltage
    pub h: BasicGatingVariable,
    /// Current output
    pub current: f32,
}

impl Default for NaIonChannel {
    fn default() -> Self {    
        NaIonChannel {
            g_na: 120., 
            e_na: 50., 
            m: BasicGatingVariable::default(),
            h: BasicGatingVariable::default(),
            current: 0.,
        }
    }
}

impl NaIonChannel {
    /// Updates the state of the gating variable based on voltage
    fn update_gate_states(&mut self, voltage: f32) {
        self.m.alpha = 0.1 * ((voltage + 40.) / (1. - (-(voltage + 40.) / 10.).exp()));
        self.m.beta = 4. * (-(voltage + 65.) / 18.).exp();
        self.h.alpha = 0.07 * (-(voltage + 65.) / 20.).exp();
        self.h.beta = 1. / ((-(voltage + 35.) / 10.).exp() + 1.);
    }
}

impl IonChannel for NaIonChannel {
    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.update_gate_states(voltage);

        self.m.update(dt);
        self.h.update(dt);

        self.current = self.m.state.powf(3.) * self.h.state * self.g_na * (voltage - self.e_na);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// A potassium ion channel
#[derive(Debug, Clone, Copy)]
pub struct KIonChannel {
    /// Maximal conductance (nS)
    pub g_k: f32,
    /// Reversal potential (mV)
    pub e_k: f32,
    /// Gating variable that changes based on voltage
    pub n: BasicGatingVariable,
    /// Current output
    pub current: f32,
}

impl Default for KIonChannel {
    fn default() -> Self {
        KIonChannel {
            g_k: 36., 
            e_k: -77., 
            n: BasicGatingVariable::default(),
            current: 0.,
        }
    }
}

impl KIonChannel {
    /// Updates the state of the gating variable based on voltage
    fn update_gate_states(&mut self, voltage: f32) {
        self.n.alpha = 0.01 * (voltage + 55.) / (1. - (-(voltage + 55.) / 10.).exp());
        self.n.beta = 0.125 * (-(voltage + 65.) / 80.).exp();
    }
}

impl IonChannel for KIonChannel {
    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.update_gate_states(voltage);

        self.n.update(dt);

        self.current = self.n.state.powf(4.) * self.g_k * (voltage - self.e_k);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// A potassium leak channel
#[derive(Debug, Clone, Copy)]
pub struct KLeakChannel {
    /// Maximal conductance (nS)
    pub g_k_leak: f32,
    /// Reversal potential (mV)
    pub e_k_leak: f32,
    /// Current output
    pub current: f32,
}

impl Default for KLeakChannel {
    fn default() -> Self {
        KLeakChannel {
            g_k_leak: 0.3, 
            e_k_leak: -55., 
            current: 0.,
        }
    }
}

impl TimestepIndependentIonChannel for KLeakChannel {
    fn update_current(&mut self, voltage: f32) {
        self.current = self.g_k_leak * (voltage - self.e_k_leak);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// A calcium channel with reduced dimensionality
#[derive(Debug, Clone, Copy)]
pub struct ReducedCalciumChannel {
    /// Conductance of calcium channel (nS)
    pub g_ca: f32,
    /// Reversal potential (mV)
    pub v_ca: f32,
    /// Gating variable steady state
    pub m_ss: f32,
    /// Tuning parameter
    pub v_1: f32,
    /// Tuning parameter
    pub v_2: f32,
    /// Current output
    pub current: f32,
}

impl Default for ReducedCalciumChannel {
    fn default() -> Self {
        ReducedCalciumChannel {
            g_ca: 4.,
            v_ca: 120.,
            m_ss: 0.,
            v_1: -1.2,
            v_2: 18.,
            current: 0.,
        }
    }
}

impl TimestepIndependentIonChannel for ReducedCalciumChannel {
    fn update_current(&mut self, voltage: f32) {
        self.m_ss = 0.5 * (1. + ((voltage - self.v_1) / self.v_2).tanh());

        self.current = self.g_ca * self.m_ss * (voltage - self.v_ca);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// A potassium channel based on steady state calculations
#[derive(Debug, Clone, Copy)]
pub struct KSteadyStateChannel {
    /// Conductance of potassium channel (nS)
    pub g_k: f32,
    /// Reversal potential (mV)
    pub v_k: f32,
    /// Gating variable
    pub n: f32,
    /// Gating variable steady state
    pub n_ss: f32,
    /// Gating decay
    pub t_n: f32,
    /// Reference frequency
    pub phi: f32,
    /// Tuning parameter
    pub v_3: f32,
    /// Tuning parameter
    pub v_4: f32,
    /// Current output
    pub current: f32
}

impl Default for KSteadyStateChannel {
    fn default() -> Self {
        KSteadyStateChannel { 
            g_k: 8., 
            v_k: -84., 
            n: 0., 
            n_ss: 0., 
            t_n: 0.,
            phi: 0.067, 
            v_3: 12., 
            v_4: 17.4, 
            current: 0.,
        }
    }
}

impl KSteadyStateChannel {
    fn update_gating_variables(&mut self, voltage: f32) {
        self.n_ss = 0.5 * (1. + ((voltage - self.v_3) / self.v_4).tanh());
        self.t_n = 1. / (self.phi * ((voltage - self.v_3) / (2. * self.v_4)).cosh());
    }
}

impl IonChannel for KSteadyStateChannel {
    fn update_current(&mut self, voltage: f32, dt: f32) {
        self.update_gating_variables(voltage);

        let n_change = ((self.n_ss - self.n) / self.t_n) * dt;

        self.n += n_change;

        self.current = self.g_k * self.n * (voltage - self.v_k);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

/// An implementation of a leak channel
#[derive(Debug, Clone, Copy)]
pub struct LeakChannel {
    /// Conductance of leak channel (nS)
    pub g_l: f32,
    /// Reversal potential (mV)
    pub v_l: f32,
    /// Current output
    pub current: f32
}

impl Default for LeakChannel {
    fn default() -> Self {
        LeakChannel { 
            g_l: 2., 
            v_l: -60., 
            current: 0.,
        }
    }
}

impl TimestepIndependentIonChannel for LeakChannel {
    fn update_current(&mut self, voltage: f32) {
        self.current = self.g_l * (voltage - self.v_l);
    }

    fn get_current(&self) -> f32 {
        self.current
    }
}

#[cfg(feature = "gpu")]
pub trait IonChannelGPU: IonChannel {
    /// Gets a given attribute from the ion channel
    fn get_attribute(&self, attribute: &str) -> Option<BufferType>;
    /// Gets a sets attribute in the ion channel
    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error>;
    /// Gets all possible attributes
    fn get_all_attributes() -> HashSet<(String, AvailableBufferType)>;
    /// Retrieves all attribute names as a vector
    fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)>;
    /// Gets update function with the associated argument names
    fn get_update_function() -> (Vec<String>, String);
    /// Converts the representation to one that can be used on the GPU
    fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts the GPU representation to a CPU representation
    fn convert_to_cpu(
        prefix: &str,
        grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}

#[cfg(feature = "gpu")]
pub trait TimestepIndependentIonChannelGPU: TimestepIndependentIonChannel {
    /// Gets a given attribute from the ion channel
    fn get_attribute(&self, attribute: &str) -> Option<BufferType>;
    /// Gets a sets attribute in the ion channel
    fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error>;
    /// Gets all possible attributes
    fn get_all_attributes() -> HashSet<(String, AvailableBufferType)>;
    /// Retrieves all attribute names as a vector
    fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)>;
    /// Gets update function with the associated argument names
    fn get_update_function() -> (Vec<String>, String);
    /// Converts the representation to one that can be used on the GPU
    fn convert_to_gpu(
        grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts the GPU representation to a CPU representation
    fn convert_to_cpu(
        prefix: &str,
        grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}
