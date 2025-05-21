//! A few implementations of different spike trains that can be coupled with `IterateAndSpike`
//! based neurons.

use rand::Rng;
use super::iterate_and_spike::{
    ApproximateNeurotransmitter, CurrentVoltage, IonotropicNeurotransmitterType, IsSpiking, LastFiringTime, NeurotransmitterConcentrations, NeurotransmitterKinetics, NeurotransmitterType, Neurotransmitters, Timestep
};
use super::iterate_and_spike_traits::{SpikeTrainBase, Timestep};
use super::plasticity::BCMActivity;
use super::intermediate_delegate::NeurotransmittersIntermediate;
#[cfg(feature = "gpu")]
use opencl3::{
    context::Context, command_queue::CommandQueue,
    types::{cl_float, cl_uint, cl_int, CL_BLOCKING, CL_NON_BLOCKING},
    memory::{Buffer, CL_MEM_READ_WRITE}, 
    kernel::Kernel, program::Program,
};
#[cfg(feature = "gpu")]
use std::ptr;
#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use super::iterate_and_spike::{
    KernelFunction, BufferGPU, NeurotransmitterKineticsGPU, NeurotransmitterTypeGPU,
    generate_unique_prefix, AvailableBufferType,
    create_float_buffer, create_optional_uint_buffer, create_uint_buffer,
    read_and_set_buffer, flatten_and_retrieve_field, write_buffer,
};
#[cfg(feature = "gpu")]
use crate::error::GPUError;


/// Handles dynamics of spike train effect on another neuron given the current timestep
/// of the simulation (neural refractoriness function), when the spike train spikes
/// the total effect also spikes while every subsequent iteration after that spike
/// results in the effect decaying back to a resting point mimicking an action potential
pub trait NeuralRefractoriness: Default + Clone + Send + Sync {
    /// Sets decay value
    fn set_decay(&mut self, decay_factor: f32);
    /// Gets decay value
    fn get_decay(&self) -> f32;
    /// Calculates neural refractoriness based on the current time of the simulation, the 
    /// last spiking time, the maximum and minimum voltage (mV)
    /// for scaling, and the simulation timestep (ms)
    fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32;
}

macro_rules! impl_default_neural_refractoriness {
    ($name:ident, $effect:expr) => {
        impl Default for $name {
            fn default() -> Self {
                $name {
                    k: 10000.,
                }
            }
        }

        impl NeuralRefractoriness for $name {
            fn set_decay(&mut self, decay_factor: f32) {
                self.k = decay_factor;
            }

            fn get_decay(&self) -> f32 {
                self.k
            }

            fn get_effect(&self, timestep: usize, last_firing_time: usize, v_max: f32, v_resting: f32, dt: f32) -> f32 {
                let a = v_max - v_resting;
                let time_difference = (timestep - last_firing_time) as f32;

                $effect(self.k, a, time_difference, v_resting, dt)
            }
        }
    };
}

/// Calculates refractoriness based on the delta dirac function
#[derive(Debug, Clone, Copy)]
pub struct DeltaDiracRefractoriness {
    /// Decay value
    pub k: f32,
}

fn delta_dirac_effect(k: f32, a: f32, time_difference: f32, v_resting: f32, dt: f32) -> f32 {
    a * ((-1. / (k / dt)) * time_difference.powf(2.)).exp() + v_resting
}

impl_default_neural_refractoriness!(DeltaDiracRefractoriness, delta_dirac_effect);

#[cfg(feature = "gpu")]
impl NeuralRefractorinessGPU for DeltaDiracRefractoriness {
    fn get_refractoriness_gpu_function() -> Result<(Vec<(String, Option<AvailableBufferType>)>, String), GPUError> {
        let args = vec![
            (String::from("timestep"), None), (String::from("last_firing_time"), Some(AvailableBufferType::OptionalUInt)),
            (String::from("v_th"), Some(AvailableBufferType::Float)), (String::from("v_resting"), Some(AvailableBufferType::Float)), 
            (String::from("neural_refractoriness$k"), Some(AvailableBufferType::Float)), (String::from("dt"), Some(AvailableBufferType::Float)),
        ];

        let program_source = String::from(r#"
            float get_effect(
                int timestep,
                int last_firing_time,
                float v_th,
                float v_resting,
                float k, 
                float dt
            ) {
                float a = v_th - v_resting;
                float time_difference = timestep - last_firing_time;

                return a * exp((-1.0f / (k / dt)) * time_difference * time_difference) + v_resting;
            }
        "#);

        Ok((args, program_source))
    }
    
    fn convert_to_gpu(
        grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if grid.is_empty() {
            return Ok(HashMap::new());
        }

        let mut buffers = HashMap::new();

        create_float_buffer!(k_buffer, context, queue, grid, k, last);

        buffers.insert(String::from("neural_refractoriness$k"), BufferGPU::Float(k_buffer));

        Ok(buffers)
    }
    
    #[allow(clippy::needless_range_loop)]
    fn convert_to_cpu(
        grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            grid.clear();

            return Ok(());
        }

        let mut k: Vec<f32> = vec![0.0; rows * cols];

        read_and_set_buffer!(buffers, queue, "neural_refractoriness$k", &mut k, Float);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let value = &mut grid[i][j];

                value.k = k[idx];
            }
        }

        Ok(())
    }
}

/// Calculates refactoriness based on exponential decay
#[derive(Debug, Clone, Copy)]
pub struct ExponentialDecayRefractoriness {
    /// Decay value
    pub k: f32
}

fn exponential_decay_effect(k: f32, a: f32, time_difference: f32, v_resting: f32, dt: f32) -> f32 {
    a * ((-1. / (k / dt)) * time_difference).exp() + v_resting
}

impl_default_neural_refractoriness!(ExponentialDecayRefractoriness, exponential_decay_effect);

/// Handles spike train dynamics
pub trait SpikeTrain: CurrentVoltage + IsSpiking + LastFiringTime + Timestep + Clone + Send + Sync {
    type U: NeuralRefractoriness;
    type N: NeurotransmitterType;
    /// Updates spike train
    fn iterate(&mut self) -> bool;
    /// Gets maximum and minimum voltage values
    fn get_height(&self) -> (f32, f32);
    /// Returns neurotransmitter concentrations
    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N>;
    /// Returns refractoriness dynamics
    fn get_refractoriness_function(&self) -> &Self::U;
}

#[cfg(feature = "gpu")]
type NameAndOptionalType = (String, Option<AvailableBufferType>);

#[cfg(feature = "gpu")]
pub trait NeuralRefractorinessGPU: NeuralRefractoriness {
    fn get_refractoriness_gpu_function() -> Result<(Vec<NameAndOptionalType>, String), GPUError>;
    fn convert_to_gpu(
        grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    fn convert_to_cpu(
        grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}

#[cfg(feature = "gpu")]
/// Handles spike train dyanmics on the GPU
pub trait SpikeTrainGPU: SpikeTrain 
where 
    Self::U: NeuralRefractorinessGPU,
    Self::N: NeurotransmitterTypeGPU
{
    /// Returns the compiled kernel for electrical outputs
    fn spike_train_electrical_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
    /// Returns the compiled kernel for chemical outputs
    fn spike_train_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError>;
    // gets the kernel to calculate the neural refractoriness connections
    // fn refractoriness_kernel(context: &Context) -> Result<?>;
    /// Converts a grid of the spike train type to a vector of buffers
    fn convert_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts buffers back to a grid of spike trains
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
    /// Converts a grid of the spike train type to a vector of buffers with necessary chemical data
    fn convert_electrochemical_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError>;
    /// Converts buffers back to a grid of spike trains with necessary chemical data
    fn convert_electrochemical_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError>;
}

/// A Poisson neuron
#[derive(Debug, Clone, SpikeTrainBase)]
pub struct PoissonNeuron<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f32,
    /// Maximum voltage (mV)
    pub v_th: f32,
    /// Minimum voltage (mV)
    pub v_resting: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last firing time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Chance of neuron firing at a given timestep
    pub chance_of_firing: f32,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

macro_rules! impl_default_spike_train_methods {
    () => {
        type U = U;
        type N = N;

        fn get_height(&self) -> (f32, f32) {
            (self.v_th, self.v_resting)
        }
    
        fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    
        fn get_refractoriness_function(&self) -> &Self::U {
            &self.neural_refractoriness
        }
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PoissonNeuron<N, T, U> {
    fn default() -> Self {
        PoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            chance_of_firing: 0.,
            dt: 0.1,
        }
    }
}

impl PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f32, dt: f32) -> Self {
        PoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> PoissonNeuron<N, T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f32, dt: f32) -> Self {
        let mut poisson_neuron = PoissonNeuron::<N, T, U>::default();

        poisson_neuron.dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.dt) / hertz);

        poisson_neuron
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Timestep for PoissonNeuron<N, T, U> {
    fn get_dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        let scalar = dt / self.dt;
        self.chance_of_firing *= scalar;
        self.dt = dt;
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PoissonNeuron<N, T, U> {
    fn iterate(&mut self) -> bool {
        let is_spiking = if rand::thread_rng().gen_range(0.0..=1.0) <= self.chance_of_firing {
            self.current_voltage = self.v_th;

            true
        } else {
            self.current_voltage = self.v_resting;

            false
        };
        self.is_spiking = is_spiking;

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        is_spiking
    }

    impl_default_spike_train_methods!();
}

// gpu implementation of spike train
// randomly emit spike in kernel
// modify neurotransmitter based on is_spiking
// have associated neural refractoriness function

// scale the rand
#[cfg(feature = "gpu")]
const RAND_FUNCTION: &str = r#"
    uint rand(uint seed) {
        uint x = seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }
"#;

#[cfg(feature = "gpu")]
impl<N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU, U: NeuralRefractorinessGPU> SpikeTrainGPU for PoissonNeuron<N, T, U> {
    fn spike_train_electrical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let kernel_name = String::from("poisson_neuron_electrical_kernel");
        let mut argument_names = vec![
            String::from("index_to_position"), String::from("seed"), String::from("current_voltage"), String::from("dt"), 
            String::from("v_resting"), String::from("v_th"), String::from("chance_of_firing"), String::from("is_spiking")
        ];

        let uint_args = [String::from("is_spiking"), String::from("seed"), String::from("index_to_position")];

        let mut processed_argument_names: Vec<String> = argument_names.iter()
            .map(|i| {
                if uint_args.contains(i) {
                    format!("__global uint *{}", i)
                } else {
                    format!("__global float *{}", i)
                }
            })
            .collect();

        processed_argument_names.insert(0, String::from("uint skip_index"));

        let program_source = format!(r#"
            {}

            __kernel void poisson_neuron_electrical_kernel(
                {}
            ) {{
                int gid = get_global_id(0);
                int index = index_to_position[gid + skip_index] - skip_index;

                uint new_seed = rand(seed[index]);
                seed[index] = new_seed;
                float random_number = ((float) new_seed / 0xFFFFFFFF);
                if (random_number < chance_of_firing[index]) {{
                    is_spiking[index] = 1;
                }} else {{
                    is_spiking[index] = 0;
                }}

                if (is_spiking[index] == 1) {{
                    current_voltage[index] = v_th[index];
                }} else {{
                    current_voltage[index] = v_resting[index];
                }}
            }}
            "#,
            RAND_FUNCTION,
            processed_argument_names.join(",\n")
        );

        let spike_train_program = match Program::create_and_build_from_source(context, &program_source, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let kernel = match Kernel::create(&spike_train_program, &kernel_name) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        argument_names.insert(0, String::from("skip_index"));

        Ok(
            KernelFunction { 
                kernel, 
                program_source, 
                kernel_name, 
                argument_names, 
            }
        )
    }

    fn spike_train_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let kernel_name = String::from("poisson_neuron_electrochemical_kernel");
        let mut argument_names = vec![
            String::from("number_of_types"), String::from("index_to_position"),  String::from("neuro_flags"),
            String::from("seed"), String::from("current_voltage"), String::from("dt"), 
            String::from("v_resting"), String::from("v_th"), String::from("chance_of_firing"), 
            String::from("is_spiking")
        ];

        let neuro_prefix = generate_unique_prefix(&argument_names, "neuro");
        let neurotransmitter_args = T::get_attribute_names_as_vector()
            .iter()
            .map(|i| (
                i.1, 
                format!(
                    "{}{}", neuro_prefix,
                    i.0.split("$").collect::<Vec<&str>>()[1],
                )
            ))
            .collect::<Vec<(AvailableBufferType, String)>>();
        let neurotransmitter_arg_names = neurotransmitter_args.iter()
            .map(|i| i.1.clone())
            .collect::<Vec<String>>();

        let parsed_neurotransmitter_args = neurotransmitter_args.iter()
            .map(|i| format!("__global {}* {}", i.0.to_str(), i.1))
            .collect::<Vec<String>>();

        let uint_args = [
            String::from("neuro_flags"), String::from("index_to_position"), 
            String::from("is_spiking"), String::from("seed"),
        ];

        let mut parsed_argument_names: Vec<String> = argument_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let qualifier = if i < 3 { "__global const " } else { "__global " };
                let type_decl = if uint_args.contains(name) { "uint" } else { "float" };
                format!("{}{}* {}", qualifier, type_decl, name)
            })
            .collect::<Vec<_>>();

        parsed_argument_names[0] = String::from("uint number_of_types");

        parsed_argument_names.extend(parsed_neurotransmitter_args);

        parsed_argument_names.insert(0, String::from("uint skip_index"));

        argument_names.insert(0, String::from("skip_index"));

        let program_source = format!(r#"
            {}
            {}
            {}

            __kernel void poisson_neuron_electrochemical_kernel(
                {}
            ) {{
                int gid = get_global_id(0);
                int index = index_to_position[gid + skip_index] - skip_index;

                float new_seed = rand(seed[index]);
                seed[index] = new_seed;
                float random_number = ((float) new_seed / 0xFFFFFFFF);
                if (random_number < chance_of_firing[index]) {{
                    is_spiking[index] = 1;
                }} else {{
                    is_spiking[index] = 0;
                }}

                if (is_spiking[index] == 1) {{
                    current_voltage[index] = v_th[index];
                }} else {{
                    current_voltage[index] = v_resting[index];
                }}

                neurotransmitters_update(
                    index, 
                    number_of_types,
                    neuro_flags,
                    current_voltage,
                    is_spiking,
                    dt,
                    {}
                );
            }}
            "#,
            RAND_FUNCTION,
            T::get_update_function().1,
            Neurotransmitters::<IonotropicNeurotransmitterType, T>::get_neurotransmitter_update_kernel_code(),
            parsed_argument_names.join(",\n"),
            neurotransmitter_arg_names.join(",\n"),
        );

        let spike_train_program = match Program::create_and_build_from_source(context, &program_source, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let kernel = match Kernel::create(&spike_train_program, &kernel_name) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let mut full_argument_names = argument_names.clone();
        full_argument_names.extend(
            T::get_attribute_names_as_vector().iter()
                .map(|i| i.0.clone())
                .collect::<Vec<_>>()
        );

        Ok(
            KernelFunction { 
                kernel, 
                program_source, 
                kernel_name, 
                argument_names: full_argument_names, 
            }
        )
    }

    fn convert_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if cell_grid.is_empty() || cell_grid.iter().all(|i| i.is_empty()) {
            return Ok(HashMap::new());
        }

        let mut buffers = HashMap::new();

        create_float_buffer!(current_voltage_buffer, context, queue, cell_grid, current_voltage);
        create_float_buffer!(dt_buffer, context, queue, cell_grid, dt);
        create_float_buffer!(v_th_buffer, context, queue, cell_grid, v_th);
        create_float_buffer!(v_resting, context, queue, cell_grid, v_resting);
        create_float_buffer!(chance_of_firing_buffer, context, queue, cell_grid, chance_of_firing);

        create_optional_uint_buffer!(last_firing_time_buffer, context, queue, cell_grid, last_firing_time);

        create_uint_buffer!(is_spiking_buffer, context, queue, cell_grid, is_spiking, last);

        let size = cell_grid.iter().flat_map(|inner| inner.iter()).collect::<Vec<_>>().len();

        let mut seed_buffer = unsafe {
            Buffer::<cl_uint>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())
                .map_err(|_| GPUError::BufferCreateError)?
        };

        let initial_data: Vec<u32> = (0..size).map(|_| rand::thread_rng().gen_range(0..0xFFFFFFFF))
            .collect();

        let write_event = unsafe {
            queue
                .enqueue_write_buffer(&mut seed_buffer, CL_NON_BLOCKING, 0, &initial_data, &[])
                .map_err(|_| GPUError::BufferWriteError)?
        };
    
        write_event.wait().map_err(|_| GPUError::WaitError)?;

        buffers.insert(String::from("current_voltage"), BufferGPU::Float(current_voltage_buffer));
        buffers.insert(String::from("dt"), BufferGPU::Float(dt_buffer));
        buffers.insert(String::from("v_th"), BufferGPU::Float(v_th_buffer));
        buffers.insert(String::from("v_resting"), BufferGPU::Float(v_resting));
        buffers.insert(String::from("chance_of_firing"), BufferGPU::Float(chance_of_firing_buffer));
        buffers.insert(String::from("last_firing_time"), BufferGPU::OptionalUInt(last_firing_time_buffer));
        buffers.insert(String::from("is_spiking"), BufferGPU::UInt(is_spiking_buffer));
        buffers.insert(String::from("seed"), BufferGPU::UInt(seed_buffer));

        let refractoriness: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.neural_refractoriness.clone()).collect())
            .collect();

        let refractoriness_buffers = U::convert_to_gpu(
            &refractoriness, context, queue
        )?;

        buffers.extend(refractoriness_buffers);

        Ok(buffers)
    }

    #[allow(clippy::needless_range_loop)]
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            cell_grid.clear();

            return Ok(());
        }

        let mut current_voltage: Vec<f32> = vec![0.0; rows * cols];
        let mut dt: Vec<f32> = vec![0.0; rows * cols];
        let mut v_th: Vec<f32> = vec![0.0; rows * cols];
        let mut v_resting: Vec<f32> = vec![0.0; rows * cols];
        let mut chance_of_firing: Vec<f32> = vec![0.0; rows * cols];
        let mut last_firing_time: Vec<i32> = vec![0; rows * cols];
        let mut is_spiking: Vec<u32> = vec![0; rows * cols];

        read_and_set_buffer!(buffers, queue, "current_voltage", &mut current_voltage, Float);
        read_and_set_buffer!(buffers, queue, "dt", &mut dt, Float);
        read_and_set_buffer!(buffers, queue, "v_th", &mut v_th, Float);
        read_and_set_buffer!(buffers, queue, "v_resting", &mut v_resting, Float);
        read_and_set_buffer!(buffers, queue, "chance_of_firing", &mut chance_of_firing, Float);
        read_and_set_buffer!(buffers, queue, "last_firing_time", &mut last_firing_time, OptionalUInt);
        read_and_set_buffer!(buffers, queue, "is_spiking", &mut is_spiking, UInt);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let cell = &mut cell_grid[i][j];

                cell.current_voltage = current_voltage[idx];
                cell.dt = dt[idx];
                cell.v_th = v_th[idx];
                cell.v_resting = v_resting[idx];
                cell.chance_of_firing = chance_of_firing[idx];

                cell.last_firing_time = if last_firing_time[idx] == -1 {
                    None
                } else {
                    Some(last_firing_time[idx] as usize)
                };
                cell.is_spiking = is_spiking[idx] == 1;
            }
        }

        Ok(())
    }

    fn convert_electrochemical_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if cell_grid.is_empty() {
            return Ok(HashMap::new());
        }

        let mut buffers = Self::convert_to_gpu(cell_grid, context, queue)?;

        let neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
            .collect();

        let neurotransmitter_buffers = Neurotransmitters::<N, T>::convert_to_gpu(
            &neurotransmitters, context, queue
        )?;

        buffers.extend(neurotransmitter_buffers);

        Ok(buffers)
    }

    fn convert_electrochemical_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            cell_grid.clear();

            return Ok(());
        }

        let mut neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
            .collect();

        Self::convert_to_cpu(cell_grid, buffers, rows, cols, queue)?;
        
        Neurotransmitters::<N, T>::convert_to_cpu(
            &mut neurotransmitters, buffers, queue, rows, cols
        )?;

        for (i, row) in cell_grid.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                cell.synaptic_neurotransmitters = neurotransmitters[i][j].clone();
            }
        }

        Ok(())
    }
}

/// A preset spike train that has a set of designated firing times and an internal clock,
/// the internal clock is updated every iteration by `dt` and once the internal clock reaches one of the 
/// firing times the neuron fires, the internal clock is reset and the clock will iterate until the
/// next firing time is reached, this will cycle until the last firing time is reached and the 
/// next firing time becomes the first firing time
#[derive(Debug, Clone, SpikeTrainBase, Timestep)]
pub struct PresetSpikeTrain<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f32,
    /// Maximum voltage (mV)
    pub v_th: f32,
    /// Minimum voltage (mV)
    pub v_resting: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last spiking time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Set of times to fire at
    pub firing_times: Vec<f32>,
    /// Internal clock to track when to fire
    pub internal_clock: f32,
    /// Pointer to which firing time is next
    pub counter: usize,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for PresetSpikeTrain<N, T, U> {
    fn default() -> Self {
        PresetSpikeTrain {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            firing_times: Vec::new(),
            internal_clock: 0.,
            counter: 0,
            dt: 0.1,
        }
    }
}

impl PresetSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        PresetSpikeTrain::default()
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for PresetSpikeTrain<N, T, U> {
    fn iterate(&mut self) -> bool {
        self.internal_clock += self.dt;

        let is_spiking = if self.internal_clock > self.firing_times[self.counter] {
            self.current_voltage = self.v_th;

            self.internal_clock = 0.;

            self.counter += 1;
            if self.counter == self.firing_times.len() {
                self.counter = 0;
            }

            true
        } else {
            self.current_voltage = self.v_resting;

            false
        };
        self.is_spiking = is_spiking;

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        is_spiking
    }

    impl_default_spike_train_methods!();
}

/// A BCM compatible Poisson neuron
#[derive(Debug, Clone, SpikeTrainBase)]
pub struct BCMPoissonNeuron<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Membrane potential (mV)
    pub current_voltage: f32,
    /// Maximum voltage (mV)
    pub v_th: f32,
    /// Minimum voltage (mV)
    pub v_resting: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last firing time
    pub last_firing_time: Option<usize>,
    /// Average activity
    pub average_activity: f32,
    /// Current activity
    pub current_activity: f32,
    /// Period for calculating average activity
    pub period: usize,
    /// Current number of spikes in the firing window
    pub num_spikes: usize,
    /// Clock for firing rate calculation
    pub firing_rate_clock: f32,
    /// Current window for firing rate
    pub firing_rate_window: f32,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Chance of neuron firing at a given timestep
    pub chance_of_firing: f32,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for BCMPoissonNeuron<N, T, U> {
    fn default() -> Self {
        BCMPoissonNeuron {
            current_voltage: 0.,
            v_th: 30.,
            v_resting: 0.,
            is_spiking: false,
            last_firing_time: None,
            average_activity: 0.,
            current_activity: 0.,
            period: 3,
            num_spikes: 0,
            firing_rate_clock: 0.,
            firing_rate_window: 500.,
            synaptic_neurotransmitters: Neurotransmitters::<N, T>::default(),
            neural_refractoriness: U::default(),
            chance_of_firing: 0.,
            dt: 0.1,
        }
    }
}

impl BCMPoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> {
    /// Returns the default implementation of the spike train
    pub fn default_impl() -> Self {
        BCMPoissonNeuron::default()
    }

    /// Returns the default implementation of the spike train given a firing rate
    pub fn default_impl_from_firing_rate(hertz: f32, dt: f32) -> Self {
        BCMPoissonNeuron::from_firing_rate(hertz, dt)
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> BCMPoissonNeuron<N, T, U> {
    /// Generates Poisson neuron with appropriate chance of firing based
    /// on the given hertz (Hz) and a given refractoriness timestep (ms)
    pub fn from_firing_rate(hertz: f32, dt: f32) -> Self {
        let mut poisson_neuron = BCMPoissonNeuron::<N, T, U>::default();

        poisson_neuron.dt = dt;
        poisson_neuron.chance_of_firing = 1. / ((1000. / poisson_neuron.dt) / hertz);

        poisson_neuron
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Timestep for BCMPoissonNeuron<N, T, U> {
    fn get_dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        let scalar = dt / self.dt;
        self.chance_of_firing *= scalar;
        self.dt = dt;
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for BCMPoissonNeuron<N, T, U> {
    // activity measured as current voltage - last voltage
    
    fn iterate(&mut self) -> bool {
        let is_spiking = if rand::thread_rng().gen_range(0.0..=1.0) <= self.chance_of_firing {
            self.current_activity = self.v_th - self.current_voltage;
            self.current_voltage = self.v_th;

            true
        } else {
            self.current_activity = self.v_resting - self.current_voltage;
            self.current_voltage = self.v_resting;

            false
        };

        if is_spiking {
            self.num_spikes += 1;
        }
        self.firing_rate_clock += self.dt;
        if self.firing_rate_clock >= self.firing_rate_window {
            self.firing_rate_clock = 0.;
            self.current_activity = self.num_spikes as f32 / (self.firing_rate_window * self.dt);
            self.average_activity -= self.average_activity / self.period as f32;
            self.average_activity += self.current_activity / self.period as f32;
        }

        self.is_spiking = is_spiking;

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        is_spiking
    }

    impl_default_spike_train_methods!();
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> BCMActivity for BCMPoissonNeuron<N, T, U> {
    fn get_activity(&self) -> f32 {
        self.current_activity
    }
    
    fn get_averaged_activity(&self) -> f32 {
        self.average_activity
    }
}

#[derive(Debug, Clone, Timestep, SpikeTrainBase)]
pub struct RateSpikeTrain<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> {
    /// Current voltage (mV)
    pub current_voltage: f32,
    /// Maximal Voltage (mV)
    pub v_th: f32,
    /// Resting voltage (mV)
    pub v_resting: f32,
    /// Rate of firing
    pub rate: f32,
    /// Current step
    pub step: f32,
    /// Whether the spike train is currently spiking
    pub is_spiking: bool,
    /// Last firing time
    pub last_firing_time: Option<usize>,
    /// Postsynaptic eurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<N, T>,
    /// Neural refactoriness dynamics
    pub neural_refractoriness: U,
    /// Timestep for refractoriness (ms)
    pub dt: f32,
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> Default for RateSpikeTrain<N, T, U> {
    fn default() -> Self {
        RateSpikeTrain { 
            current_voltage: 0., 
            v_th: 30., 
            v_resting: 0., 
            rate: 0., 
            step: 0., 
            is_spiking: false, 
            last_firing_time: None, 
            synaptic_neurotransmitters: Neurotransmitters::default(), 
            neural_refractoriness: U::default(), 
            dt: 0.1,
        }
    }
}

impl<N: NeurotransmitterType, T: NeurotransmitterKinetics, U: NeuralRefractoriness> SpikeTrain for RateSpikeTrain<N, T, U> {
    fn iterate(&mut self) -> bool {
        self.step += self.dt;
        if self.rate != 0. && self.step > self.rate {
            self.step = 0.;
            self.current_voltage = self.v_th;
            self.is_spiking = true;
        } else {
            self.current_voltage = self.v_resting;
            self.is_spiking = false;
        }

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        self.is_spiking
    }

    impl_default_spike_train_methods!();
}

impl<N: NeurotransmitterTypeGPU, T: NeurotransmitterKineticsGPU, U: NeuralRefractorinessGPU> SpikeTrainGPU for RateSpikeTrain<N, T, U> {
    fn spike_train_electrical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let kernel_name = String::from("rate_spike_train_electrical_kernel");
        let mut argument_names = vec![
            String::from("index_to_position"), String::from("seed"), String::from("current_voltage"), String::from("dt"), 
            String::from("v_resting"), String::from("v_th"), String::from("rate"), String::from("step"), String::from("is_spiking")
        ];

        let uint_args = [String::from("is_spiking"), String::from("index_to_position")];

        let mut processed_argument_names: Vec<String> = argument_names.iter()
            .map(|i| {
                if uint_args.contains(i) {
                    format!("__global uint *{}", i)
                } else {
                    format!("__global float *{}", i)
                }
            })
            .collect();

        processed_argument_names.insert(0, String::from("uint skip_index"));

        let program_source = format!(r#"
            {}

            __kernel void rate_spike_train_electrical_kernel(
                {}
            ) {{
                int gid = get_global_id(0);
                int index = index_to_position[gid + skip_index] - skip_index;

                step[index] += dt[index];
                if (rate[index] != 0.0f && step[index] > rate[index]) {{
                    step[index] = 0.0f;
                    current_voltage[index] = v_th[index];
                    is_spiking[index] = true;
                }} else {{
                    current_voltage[index] = v_resting[index];
                    is_spiking[index] = false;
                }}
            }}
            "#,
            RAND_FUNCTION,
            processed_argument_names.join(",\n")
        );

        let spike_train_program = match Program::create_and_build_from_source(context, &program_source, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let kernel = match Kernel::create(&spike_train_program, &kernel_name) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        argument_names.insert(0, String::from("skip_index"));

        Ok(
            KernelFunction { 
                kernel, 
                program_source, 
                kernel_name, 
                argument_names, 
            }
        )
    }

    fn spike_train_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {
        let kernel_name = String::from("rate_spike_train_electrochemical_kernel");
        let mut argument_names = vec![
            String::from("number_of_types"), String::from("index_to_position"),  String::from("neuro_flags"),
            String::from("seed"), String::from("current_voltage"), String::from("dt"), 
            String::from("v_resting"), String::from("v_th"), String::from("rate"), 
            String::from("step"), String::from("is_spiking")
        ];

        let neuro_prefix = generate_unique_prefix(&argument_names, "neuro");
        let neurotransmitter_args = T::get_attribute_names_as_vector()
            .iter()
            .map(|i| (
                i.1, 
                format!(
                    "{}{}", neuro_prefix,
                    i.0.split("$").collect::<Vec<&str>>()[1],
                )
            ))
            .collect::<Vec<(AvailableBufferType, String)>>();
        let neurotransmitter_arg_names = neurotransmitter_args.iter()
            .map(|i| i.1.clone())
            .collect::<Vec<String>>();

        let parsed_neurotransmitter_args = neurotransmitter_args.iter()
            .map(|i| format!("__global {}* {}", i.0.to_str(), i.1))
            .collect::<Vec<String>>();

        let uint_args = [
            String::from("neuro_flags"), String::from("index_to_position"), String::from("is_spiking"), 
        ];

        let mut parsed_argument_names: Vec<String> = argument_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let qualifier = if i < 3 { "__global const " } else { "__global " };
                let type_decl = if uint_args.contains(name) { "uint" } else { "float" };
                format!("{}{}* {}", qualifier, type_decl, name)
            })
            .collect::<Vec<_>>();

        parsed_argument_names[0] = String::from("uint number_of_types");

        parsed_argument_names.extend(parsed_neurotransmitter_args);

        parsed_argument_names.insert(0, String::from("uint skip_index"));

        argument_names.insert(0, String::from("skip_index"));

        let program_source = format!(r#"
            {}
            {}
            {}

            __kernel void rate_spike_train_electrochemical_kernel(
                {}
            ) {{
                int gid = get_global_id(0);
                int index = index_to_position[gid + skip_index] - skip_index;

                step[index] += dt[index];
                if (rate[index] != 0.0f && step[index] > rate[index]) {{
                    step[index] = 0.0f;
                    current_voltage[index] = v_th[index];
                    is_spiking[index] = true;
                }} else {{
                    current_voltage[index] = v_resting[index];
                    is_spiking[index] = false;
                }}

                neurotransmitters_update(
                    index, 
                    number_of_types,
                    neuro_flags,
                    current_voltage,
                    is_spiking,
                    dt,
                    {}
                );
            }}
            "#,
            RAND_FUNCTION,
            T::get_update_function().1,
            Neurotransmitters::<IonotropicNeurotransmitterType, T>::get_neurotransmitter_update_kernel_code(),
            parsed_argument_names.join(",\n"),
            neurotransmitter_arg_names.join(",\n"),
        );

        let spike_train_program = match Program::create_and_build_from_source(context, &program_source, "") {
            Ok(value) => value,
            Err(_) => return Err(GPUError::ProgramCompileFailure),
        };
        let kernel = match Kernel::create(&spike_train_program, &kernel_name) {
            Ok(value) => value,
            Err(_) => return Err(GPUError::KernelCompileFailure),
        };

        let mut full_argument_names = argument_names.clone();
        full_argument_names.extend(
            T::get_attribute_names_as_vector().iter()
                .map(|i| i.0.clone())
                .collect::<Vec<_>>()
        );

        Ok(
            KernelFunction { 
                kernel, 
                program_source, 
                kernel_name, 
                argument_names: full_argument_names, 
            }
        )
    }

    fn convert_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if cell_grid.is_empty() || cell_grid.iter().all(|i| i.is_empty()) {
            return Ok(HashMap::new());
        }

        let mut buffers = HashMap::new();

        create_float_buffer!(current_voltage_buffer, context, queue, cell_grid, current_voltage);
        create_float_buffer!(dt_buffer, context, queue, cell_grid, dt);
        create_float_buffer!(v_th_buffer, context, queue, cell_grid, v_th);
        create_float_buffer!(v_resting, context, queue, cell_grid, v_resting);
        create_float_buffer!(rate_buffer, context, queue, cell_grid, rate);
        create_float_buffer!(step_buffer, context, queue, cell_grid, step);

        create_optional_uint_buffer!(last_firing_time_buffer, context, queue, cell_grid, last_firing_time);

        create_uint_buffer!(is_spiking_buffer, context, queue, cell_grid, is_spiking, last);

        buffers.insert(String::from("current_voltage"), BufferGPU::Float(current_voltage_buffer));
        buffers.insert(String::from("dt"), BufferGPU::Float(dt_buffer));
        buffers.insert(String::from("v_th"), BufferGPU::Float(v_th_buffer));
        buffers.insert(String::from("v_resting"), BufferGPU::Float(v_resting));
        buffers.insert(String::from("rate"), BufferGPU::Float(rate_buffer));
        buffers.insert(String::from("step"), BufferGPU::Float(step_buffer));
        buffers.insert(String::from("last_firing_time"), BufferGPU::OptionalUInt(last_firing_time_buffer));
        buffers.insert(String::from("is_spiking"), BufferGPU::UInt(is_spiking_buffer));

        let refractoriness: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.neural_refractoriness.clone()).collect())
            .collect();

        let refractoriness_buffers = U::convert_to_gpu(
            &refractoriness, context, queue
        )?;

        buffers.extend(refractoriness_buffers);

        Ok(buffers)
    }

    #[allow(clippy::needless_range_loop)]
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            cell_grid.clear();

            return Ok(());
        }

        let mut current_voltage: Vec<f32> = vec![0.0; rows * cols];
        let mut dt: Vec<f32> = vec![0.0; rows * cols];
        let mut v_th: Vec<f32> = vec![0.0; rows * cols];
        let mut v_resting: Vec<f32> = vec![0.0; rows * cols];
        let mut rate: Vec<f32> = vec![0.0; rows * cols];
        let mut step: Vec<f32> = vec![0.0; rows * cols];
        let mut last_firing_time: Vec<i32> = vec![0; rows * cols];
        let mut is_spiking: Vec<u32> = vec![0; rows * cols];

        read_and_set_buffer!(buffers, queue, "current_voltage", &mut current_voltage, Float);
        read_and_set_buffer!(buffers, queue, "dt", &mut dt, Float);
        read_and_set_buffer!(buffers, queue, "v_th", &mut v_th, Float);
        read_and_set_buffer!(buffers, queue, "v_resting", &mut v_resting, Float);
        read_and_set_buffer!(buffers, queue, "rate", &mut rate, Float);
        read_and_set_buffer!(buffers, queue, "step", &mut step, Float);
        read_and_set_buffer!(buffers, queue, "last_firing_time", &mut last_firing_time, OptionalUInt);
        read_and_set_buffer!(buffers, queue, "is_spiking", &mut is_spiking, UInt);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let cell = &mut cell_grid[i][j];

                cell.current_voltage = current_voltage[idx];
                cell.dt = dt[idx];
                cell.v_th = v_th[idx];
                cell.v_resting = v_resting[idx];
                cell.rate = rate[idx];
                cell.step = step[idx];

                cell.last_firing_time = if last_firing_time[idx] == -1 {
                    None
                } else {
                    Some(last_firing_time[idx] as usize)
                };
                cell.is_spiking = is_spiking[idx] == 1;
            }
        }

        Ok(())
    }

    fn convert_electrochemical_to_gpu(
        cell_grid: &[Vec<Self>], 
        context: &Context,
        queue: &CommandQueue,
    ) -> Result<HashMap<String, BufferGPU>, GPUError> {
        if cell_grid.is_empty() {
            return Ok(HashMap::new());
        }

        let mut buffers = Self::convert_to_gpu(cell_grid, context, queue)?;

        let neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
            .collect();

        let neurotransmitter_buffers = Neurotransmitters::<N, T>::convert_to_gpu(
            &neurotransmitters, context, queue
        )?;

        buffers.extend(neurotransmitter_buffers);

        Ok(buffers)
    }

    fn convert_electrochemical_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: &HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) -> Result<(), GPUError> {
        if rows == 0 || cols == 0 {
            cell_grid.clear();

            return Ok(());
        }

        let mut neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
            .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
            .collect();

        Self::convert_to_cpu(cell_grid, buffers, rows, cols, queue)?;
        
        Neurotransmitters::<N, T>::convert_to_cpu(
            &mut neurotransmitters, buffers, queue, rows, cols
        )?;

        for (i, row) in cell_grid.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                cell.synaptic_neurotransmitters = neurotransmitters[i][j].clone();
            }
        }

        Ok(())
    }
}