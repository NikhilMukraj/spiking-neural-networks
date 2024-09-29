//! Various integrate and fire models that implement [`IterateAndSpike`] 
//! as well as neurotransmitter and receptor dynamics through
//! [`NeurotransmitterKinetics`] and [`ReceptorKinetics`].

use iterate_and_spike_traits::IterateAndSpikeBase;
use super::iterate_and_spike::{
    ApproximateNeurotransmitter, ApproximateReceptor, CurrentVoltage, 
    GapConductance, GaussianParameters, IonotropicNeurotransmitterType, 
    IsSpiking, IterateAndSpike, LastFiringTime, LigandGatedChannels, NeurotransmitterConcentrations, 
    NeurotransmitterKinetics, Neurotransmitters, ReceptorKinetics, Timestep
};
#[cfg(feature = "gpu")]
use super::iterate_and_spike::{
    IterateAndSpikeGPU, BufferGPU, KernelFunction,
};
#[cfg(feature = "gpu")]
use opencl3::{
    context::Context, kernel::Kernel, program::Program, command_queue::CommandQueue,
    memory::{Buffer, CL_MEM_READ_WRITE}, 
    types::{cl_float, cl_uint, CL_NON_BLOCKING, CL_BLOCKING},
};
#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use std::ptr;
use super::plasticity::BCMActivity;


/// Takes in a static current as an input and iterates the given
/// neuron for a given duration, set `gaussian` to true to add 
/// normally distributed noise to the input as it iterates,
/// returns the voltages from the neuron over time
pub fn run_static_input_integrate_and_fire<T: IterateAndSpike>(
    cell: &mut T, 
    input_current: f32, 
    gaussian: Option<GaussianParameters>, 
    iterations: usize,
) -> Vec<f32> {
    let mut voltages: Vec<f32> = vec![];

    for _ in 0..iterations {
        let _is_spiking = match gaussian {
            Some(ref params) => cell.iterate_and_spike(params.get_random_number() * input_current),
            None => cell.iterate_and_spike(input_current),
        };

        voltages.push(cell.get_current_voltage());
    }

    voltages
}

macro_rules! impl_default_neurotransmitter_methods {
    () => {  
        type N = IonotropicNeurotransmitterType;

        fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
            self.synaptic_neurotransmitters.get_concentrations()
        }
    }
}

pub(crate) use impl_default_neurotransmitter_methods;

macro_rules! impl_default_impl_integrate_and_fire {
    ($name:ident) => {
        impl $name<ApproximateNeurotransmitter, ApproximateReceptor> {
            /// Returns the default implementation of the neuron
            pub fn default_impl() -> Self {
                $name::default()
            }
        }
    }
}

macro_rules! impl_default_handle_spiking {
    () => {
        /// Determines whether the neuron is spiking and resets the voltage
        /// if so, also handles refractory period
        pub fn handle_spiking(&mut self) -> bool {
            let mut is_spiking = false;

            if self.refractory_count > 0. {
                self.current_voltage = self.v_reset;
                self.refractory_count -= 1.;
            } else if self.current_voltage >= self.v_th {
                is_spiking = !is_spiking;
                self.current_voltage = self.v_reset;
                self.refractory_count = self.tref / self.dt;
            }

            self.is_spiking = is_spiking;

            is_spiking
        }
    }
}

/// A leaky integrate and fire neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct LeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Voltage reset value (mV)
    pub v_reset: f32, 
    /// Voltage initialization value (mV)
    pub v_init: f32, 
    /// Counter for refractory period
    pub refractory_count: f32, 
    /// Total refractory period (ms)
    pub tref: f32,
    /// Leak constant 
    pub leak_constant: f32, 
    /// Input value modifier
    pub integration_constant: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Leak reversal potential (mV)
    pub e_l: f32, 
    /// Leak conductance (nS)
    pub g_l: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(LeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for LeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        LeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> LeakyIntegrateAndFireNeuron<T, R> {
    /// Calculates the change in voltage given an input current
    pub fn leaky_get_dv_change(&self, i: f32) -> f32 {
        (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.integration_constant * (i / self.g_l))
        ) * (self.dt / self.tau_m)
    }

    impl_default_handle_spiking!();
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for LeakyIntegrateAndFireNeuron<T, R> {
    impl_default_neurotransmitter_methods!();

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        let dv = self.leaky_get_dv_change(input_current);
        self.current_voltage += dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

        let dv = self.leaky_get_dv_change(input_current);
        let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }
}

macro_rules! impl_iterate_and_spike {
    ($name:ident, $dv_method:ident, $dw_method:ident, $handle_spiking:ident) => {
        impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for $name<T, R> {
            impl_default_neurotransmitter_methods!();

            fn iterate_and_spike(&mut self, input_current: f32) -> bool {
                let dv = self.$dv_method(input_current);
                let dw = self.$dw_method();

                self.current_voltage += dv;
                self.w_value += dw;

                self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

                self.$handle_spiking()
            }

            fn iterate_with_neurotransmitter_and_spike(
                &mut self, 
                input_current: f32, 
                t_total: &NeurotransmitterConcentrations<Self::N>,
            ) -> bool {
                self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
                self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

                let dv = self.$dv_method(input_current);
                let dw = self.$dw_method();
                let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

                self.current_voltage += dv + neurotransmitter_dv;
                self.w_value += dw;

                self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

                self.$handle_spiking()
            }
        }
    };
}

#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct QuadraticIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32, 
    /// Voltage reset value/resting membrane potential (mV)
    pub v_reset: f32, 
    /// Voltage initialization value (mV)
    pub v_init: f32, 
    /// Counter for refractory period
    pub refractory_count: f32, 
    /// Total refractory period (ms)
    pub tref: f32, 
    /// Steepness of slope
    pub alpha: f32, 
    /// Critical voltage for spike initiation (mV)
    pub v_c: f32,
    /// Input value modifier
    pub integration_constant: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(QuadraticIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for QuadraticIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        QuadraticIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            integration_constant: 1.,
            gap_conductance: 7.,
            alpha: 1.,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // resting potential (mV)
            v_c: -60., // spike initiation threshold (mV)
            tau_m: 100., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -75., // initial potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> QuadraticIntegrateAndFireNeuron<T, R> {
    /// Calculates the change in voltage given an input current
    fn quadratic_get_dv_change(&self, i: f32) -> f32 {
        ((self.alpha * (self.current_voltage - self.v_reset) * (self.current_voltage - self.v_c)) + 
        self.integration_constant * i) * (self.dt / self.tau_m)
    }

    impl_default_handle_spiking!();
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for QuadraticIntegrateAndFireNeuron<T, R> {
    impl_default_neurotransmitter_methods!();

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        let dv = self.quadratic_get_dv_change(input_current);
        self.current_voltage += dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

        let dv = self.quadratic_get_dv_change(input_current);
        let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }
} 

#[cfg(feature = "gpu")]
macro_rules! read_and_set_buffer {
    ($buffers:expr, $queue:expr, $buffer_name:expr, $vec:expr, Float) => {
        if let Some(BufferGPU::Float(buffer)) = $buffers.get($buffer_name) {
            unsafe {
                $queue
                    .enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, $vec, &[])
                    .expect("Could not read buffer");
            }
        }
    };
    
    ($buffers:expr, $queue:expr, $buffer_name:expr, $vec:expr, UInt) => {
        if let Some(BufferGPU::UInt(buffer)) = $buffers.get($buffer_name) {
            unsafe {
                $queue
                    .enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, $vec, &[])
                    .expect("Could not read buffer");
            }
        }
    };
}

#[cfg(feature = "gpu")]
macro_rules! write_buffer {
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, Float) => {
        let mut $name = unsafe {
            Buffer::<cl_float>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let _ = unsafe { 
            $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[])
                .expect("Could not write to buffer") 
        };
    };
    
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, UInt) => {
        let mut $name = unsafe {
            Buffer::<cl_uint>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let _ = unsafe { 
            $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[])
                .expect("Could not write to buffer") 
        };
    };

    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, Float, last) => {
        let mut $name = unsafe {
            Buffer::<cl_float>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let last_event = unsafe { 
            $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[])
                .expect("Could not write to buffer") 
        };

        last_event.wait().expect("Could not wait");
    };
    
    ($name:ident, $context:expr, $queue:expr, $num:ident, $array:expr, UInt, last) => {
        let mut $name = unsafe {
            Buffer::<cl_uint>::create($context, CL_MEM_READ_WRITE, $num, ptr::null_mut())
                .expect("Could not create buffer")
        };

        let last_event = unsafe { 
            $queue.enqueue_write_buffer(&mut $name, CL_BLOCKING, 0, $array, &[])
                .expect("Could not write to buffer") 
        };

        last_event.wait().expect("Could not wait");
    };
}

#[cfg(feature = "gpu")]
macro_rules! flatten_and_retrieve_field {
    ($grid:expr, $field:ident, f32) => {
        $grid.iter()
            .flat_map(|inner| inner.iter())
            .map(|neuron| neuron.$field)
            .collect::<Vec<f32>>()
    };

    ($grid:expr, $field:ident, u32) => {
        $grid.iter()
            .flat_map(|inner| inner.iter())
            .map(|neuron| if neuron.$field { 1 } else { 0 })
            .collect::<Vec<u32>>()
    };
}

#[cfg(feature = "gpu")]
macro_rules! create_float_buffer {
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, f32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, Float);
    };

    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident, last) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, f32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, Float, last);
    };
}

#[cfg(feature = "gpu")]
macro_rules! create_uint_buffer {
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, u32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, UInt);
    };
    
    ($name:ident, $context:expr, $queue:expr, $grid:expr, $field:ident, last) => {
        let flattened_field = flatten_and_retrieve_field!($grid, $field, u32);
        let cell_grid_size = flattened_field.len();
        write_buffer!($name, $context, $queue, cell_grid_size, &flattened_field, UInt, last);
    };
}

#[cfg(feature = "gpu")]
impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpikeGPU for QuadraticIntegrateAndFireNeuron<T, R> {
    fn iterate_and_spike_electrical_kernel(context: &Context) -> KernelFunction {
        let kernel_name = String::from("quadratic_integrate_and_fire_iterate_and_spike");
        let argument_names = vec![
            String::from("inputs"), String::from("index_to_position"), String::from("current_voltage"), 
            String::from("alpha"), String::from("v_reset"), String::from("v_c"), 
            String::from("integration_constant"), String::from("dt"), String::from("tau_m"),
            String::from("v_th"), String::from("refractory_count"), String::from("tref"),
            String::from("is_spiking"),
        ];

        let program_source = String::from(r#"
            __kernel void quadratic_integrate_and_fire_iterate_and_spike(
                __global const float *inputs,
                __global const uint *index_to_position,
                __global float *current_voltage,
                __global float *alpha,
                __global float *v_reset,
                __global float *v_c,
                __global float *integration_constant,
                __global float *dt,
                __global float *tau_m,
                __global float *v_th,
                __global float *refractory_count,
                __global float *tref,
                __global uint *is_spiking
            ) {
                int gid = get_global_id(0);
                int index = index_to_position[gid];

                current_voltage[index] += (
                    alpha[index] * (current_voltage[index] - v_reset[index]) * 
                    (current_voltage[index] - v_c[index]) + integration_constant[index] * inputs[index]
                    ) 
                    * (dt[index] / tau_m[index]);

                if (refractory_count[index] > 0.0f) {
                    current_voltage[index] = v_reset[index];
                    refractory_count[index] -= 1.0f; 
                    is_spiking[index] = 0;
                } else if (current_voltage[index] >= v_th[index]) {
                    current_voltage[index] = v_reset[index];
                    is_spiking[index] = 1;
                    refractory_count[index] = tref[index] / dt[index];
                } else {
                    is_spiking[index] = 0;
                }
            }
        "#);

        let iterate_and_spike_program = Program::create_and_build_from_source(context, &program_source, "")
            .expect("Program::create_and_build_from_source failed");
        let kernel = Kernel::create(&iterate_and_spike_program, &kernel_name)
            .expect("Kernel::create failed");

        KernelFunction { 
            kernel, 
            program_source, 
            kernel_name, 
            argument_names, 
        }
    }
    
    fn convert_to_gpu(cell_grid: &[Vec<Self>], context: &Context, queue: &CommandQueue) -> HashMap<String, BufferGPU> {
        let mut buffers = HashMap::new();

        create_float_buffer!(current_voltage_buffer, context, queue, cell_grid, current_voltage);
        create_float_buffer!(gap_conductance_buffer, context, queue, cell_grid, gap_conductance);
        create_float_buffer!(alpha_buffer, context, queue, cell_grid, alpha);
        create_float_buffer!(v_reset_buffer, context, queue, cell_grid, v_reset);
        create_float_buffer!(v_c_buffer, context, queue, cell_grid, v_c);
        create_float_buffer!(integration_constant_buffer, context, queue, cell_grid, integration_constant);
        create_float_buffer!(dt_buffer, context, queue, cell_grid, dt);
        create_float_buffer!(tau_m_buffer, context, queue, cell_grid, tau_m);
        create_float_buffer!(v_th_buffer, context, queue, cell_grid, v_th);
        create_float_buffer!(refractory_count_buffer, context, queue, cell_grid, refractory_count);
        create_float_buffer!(tref_buffer, context, queue, cell_grid, tref);

        create_uint_buffer!(is_spiking_buffer, context, queue, cell_grid, is_spiking, last);

        buffers.insert(String::from("current_voltage"), BufferGPU::Float(current_voltage_buffer));
        buffers.insert(String::from("gap_conductance"), BufferGPU::Float(gap_conductance_buffer));
        buffers.insert(String::from("alpha"), BufferGPU::Float(alpha_buffer));
        buffers.insert(String::from("v_reset"), BufferGPU::Float(v_reset_buffer));
        buffers.insert(String::from("v_c"), BufferGPU::Float(v_c_buffer));
        buffers.insert(String::from("integration_constant"), BufferGPU::Float(integration_constant_buffer));
        buffers.insert(String::from("dt"), BufferGPU::Float(dt_buffer));
        buffers.insert(String::from("tau_m"), BufferGPU::Float(tau_m_buffer));
        buffers.insert(String::from("v_th"), BufferGPU::Float(v_th_buffer));
        buffers.insert(String::from("refractory_count"), BufferGPU::Float(refractory_count_buffer));
        buffers.insert(String::from("tref"), BufferGPU::Float(tref_buffer));

        buffers.insert(String::from("is_spiking"), BufferGPU::UInt(is_spiking_buffer));

        buffers
    }

    #[allow(clippy::needless_range_loop)]
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) {
        let mut current_voltage: Vec<f32> = vec![0.0; rows * cols];
        let mut gap_conductance: Vec<f32> = vec![0.0; rows * cols];
        let mut alpha: Vec<f32> = vec![0.0; rows * cols];
        let mut v_reset: Vec<f32> = vec![0.0; rows * cols];
        let mut v_c: Vec<f32> = vec![0.0; rows * cols];
        let mut integration_constant: Vec<f32> = vec![0.0; rows * cols];
        let mut dt: Vec<f32> = vec![0.0; rows * cols];
        let mut tau_m: Vec<f32> = vec![0.0; rows * cols];
        let mut v_th: Vec<f32> = vec![0.0; rows * cols];
        let mut refractory_count: Vec<f32> = vec![0.0; rows * cols];
        let mut tref: Vec<f32> = vec![0.0; rows * cols];
        let mut is_spiking: Vec<u32> = vec![0; rows * cols];

        read_and_set_buffer!(buffers, queue, "current_voltage", &mut current_voltage, Float);
        read_and_set_buffer!(buffers, queue, "gap_conductance", &mut gap_conductance, Float);
        read_and_set_buffer!(buffers, queue, "alpha", &mut alpha, Float);
        read_and_set_buffer!(buffers, queue, "v_reset", &mut v_reset, Float);
        read_and_set_buffer!(buffers, queue, "v_c", &mut v_c, Float);
        read_and_set_buffer!(buffers, queue, "integration_constant", &mut integration_constant, Float);
        read_and_set_buffer!(buffers, queue, "dt", &mut dt, Float);
        read_and_set_buffer!(buffers, queue, "tau_m", &mut tau_m, Float);
        read_and_set_buffer!(buffers, queue, "v_th", &mut v_th, Float);
        read_and_set_buffer!(buffers, queue, "refractory_count", &mut refractory_count, Float);
        read_and_set_buffer!(buffers, queue, "tref", &mut tref, Float);
        read_and_set_buffer!(buffers, queue, "is_spiking", &mut is_spiking, UInt);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let cell = &mut cell_grid[i][j];
                
                cell.current_voltage = current_voltage[idx];
                cell.gap_conductance = gap_conductance[idx];
                cell.alpha = alpha[idx];
                cell.v_reset = v_reset[idx];
                cell.v_c = v_c[idx];
                cell.integration_constant = integration_constant[idx];
                cell.dt = dt[idx];
                cell.tau_m = tau_m[idx];
                cell.v_th = v_th[idx];
                cell.refractory_count = refractory_count[idx];
                cell.tref = tref[idx];
                cell.is_spiking = is_spiking[idx] == 1;
            }
        }
    }
}


/// An adaptive leaky integrate and fire neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct AdaptiveLeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32, 
    /// Voltage reset value (mV)
    pub v_reset: f32, 
    /// Voltage initialization value (mV)
    pub v_init: f32, 
    /// Counter for refractory period
    pub refractory_count: f32, 
    /// Total refractory period (ms)
    pub tref: f32, 
    /// Controls effect of leak reverse potential on adaptive value update
    pub alpha: f32, 
    /// Controls how adaptive value is changed in spiking
    pub beta: f32, 
    /// Adaptive value
    pub w_value: f32, 
    /// Adaptive value initialization
    pub w_init: f32, 
    /// Leak constant
    pub leak_constant: f32, 
    /// Input value modifier
    pub integration_constant: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Leak reversal potential (mV)
    pub e_l: f32, 
    /// Leak conductance (nS)
    pub g_l: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(AdaptiveLeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for AdaptiveLeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        AdaptiveLeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 6.0,
            beta: 10.0,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

macro_rules! impl_adaptive_default_methods {
    () => {
        /// Calculates how adaptive value changes
        pub fn adaptive_get_dw_change(&self) -> f32 {
            let dw = (
                self.alpha * (self.current_voltage - self.e_l) -
                self.w_value
            ) * (self.dt / self.tau_m);
    
            dw
        }
    
        /// Determines whether the neuron is spiking, resets the voltage and 
        /// updates the adaptive value if spiking, also handles refractory period
        pub fn adaptive_handle_spiking(&mut self) -> bool {
            let mut is_spiking = false;
    
            if self.refractory_count > 0. {
                self.current_voltage = self.v_reset;
                self.refractory_count -= 1.;
            } else if self.current_voltage >= self.v_th {
                is_spiking = !is_spiking;
                self.current_voltage = self.v_reset;
                self.w_value += self.beta;
                self.refractory_count = self.tref / self.dt
            }

            self.is_spiking = is_spiking;
    
            is_spiking
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> AdaptiveLeakyIntegrateAndFireNeuron<T, R> {
    /// Calculates the change in voltage given an input current
    pub fn adaptive_get_dv_change(&self, i: f32) -> f32 {
        (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m)
    }

    impl_adaptive_default_methods!();
}

impl_iterate_and_spike!(
    AdaptiveLeakyIntegrateAndFireNeuron, 
    adaptive_get_dv_change, 
    adaptive_get_dw_change,
    adaptive_handle_spiking
);

/// An adaptive exponential leaky integrate and fire neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct AdaptiveExpLeakyIntegrateAndFireNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32, 
    /// Voltage reset value (mV)
    pub v_reset: f32, 
    /// Voltage initialization value (mV)
    pub v_init: f32, 
    /// Counter for refractory period
    pub refractory_count: f32, 
    /// Total refractory period (ms)
    pub tref: f32, 
    /// Controls effect of leak reverse potential on adaptive value update
    pub alpha: f32, 
    /// Controls how adaptive value is changed in spiking
    pub beta: f32, 
    /// Controls steepness
    pub slope_factor: f32, 
    /// Adaptive value
    pub w_value: f32, 
    /// Adaptive value initialization
    pub w_init: f32, 
    /// Leak constant
    pub leak_constant: f32, 
    /// Input value modifier
    pub integration_constant: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Leak reversal potential (mV)
    pub e_l: f32, 
    /// Leak conductance (nS)
    pub g_l: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(AdaptiveExpLeakyIntegrateAndFireNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for AdaptiveExpLeakyIntegrateAndFireNeuron<T, R> {
    fn default() -> Self {
        AdaptiveExpLeakyIntegrateAndFireNeuron {
            current_voltage: -75., 
            refractory_count: 0.0,
            leak_constant: -1.,
            integration_constant: 1.,
            gap_conductance: 7.,
            w_value: 0.,
            alpha: 6.0,
            beta: 10.0,
            slope_factor: 1.,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            g_l: 10., // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e_l: -75., // leak reversal potential (mV)
            tref: 10., // refractory time (ms), could rename to refract_time
            w_init: 0., // initial w value
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> AdaptiveExpLeakyIntegrateAndFireNeuron<T, R> {
    /// Calculates the change in voltage given an input current
    pub fn exp_adaptive_get_dv_change(&self, i: f32) -> f32 {
        (
            (self.leak_constant * (self.current_voltage - self.e_l)) +
            (self.slope_factor * ((self.current_voltage - self.v_th) / self.slope_factor).exp()) +
            (self.integration_constant * (i / self.g_l)) - 
            (self.w_value / self.g_l)
        ) * (self.dt / self.c_m)
    }

    impl_adaptive_default_methods!();
}

impl_iterate_and_spike!(
    AdaptiveExpLeakyIntegrateAndFireNeuron, 
    exp_adaptive_get_dv_change, 
    adaptive_get_dw_change,
    adaptive_handle_spiking
);

/// An Izhikevich neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct IzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Voltage initialization value (mV) 
    pub v_init: f32, 
    /// Controls speed
    pub a: f32, 
    /// Controls sensitivity to adaptive value
    pub b: f32,
    /// After spike reset value for voltage 
    pub c: f32,
    /// After spike reset value for adaptive value 
    pub d: f32, 
    /// Adaptive value
    pub w_value: f32, 
    /// Adaptive value initialization
    pub w_init: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(IzhikevichNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for IzhikevichNeuron<T, R> {
    fn default() -> Self {
        IzhikevichNeuron {
            current_voltage: -65., 
            gap_conductance: 7.,
            w_value: 30.,
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            tau_m: 1., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -65., // initial potential (mV)
            w_init: 30., // initial w value
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

macro_rules! impl_izhikevich_default_methods {
    () => {
        // Calculates how adaptive value changes
        pub fn izhikevich_get_dw_change(&self) -> f32 {
            let dw = (
                self.a * (self.b * self.current_voltage - self.w_value)
            ) * (self.dt / self.tau_m);
    
            dw
        }
    
        /// Determines whether the neuron is spiking, updates the voltage and 
        /// updates the adaptive value if spiking
        pub fn izhikevich_handle_spiking(&mut self) -> bool {
            let mut is_spiking = false;
    
            if self.current_voltage >= self.v_th {
                is_spiking = !is_spiking;
                self.current_voltage = self.c;
                self.w_value += self.d;
            }

            self.is_spiking = is_spiking;
    
            is_spiking
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IzhikevichNeuron<T, R> {
    impl_izhikevich_default_methods!();

    /// Calculates the change in voltage given an input current
    pub fn izhikevich_get_dv_change(&self, i: f32) -> f32 {
        (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (self.dt / self.c_m)
    }
}

impl_iterate_and_spike!(
    IzhikevichNeuron, 
    izhikevich_get_dv_change, 
    izhikevich_get_dw_change,
    izhikevich_handle_spiking
);

/// A leaky Izhikevich neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct LeakyIzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Voltage initialization value (mV) 
    pub v_init: f32, 
    /// Controls speed
    pub a: f32, 
    /// Controls sensitivity to adaptive value
    pub b: f32,
    /// After spike reset value for voltage 
    pub c: f32,
    /// After spike reset value for adaptive value 
    pub d: f32, 
    /// Adaptive value
    pub w_value: f32, 
    /// Adaptive value initialization
    pub w_init: f32, 
    /// Leak reversal potential (mV)
    pub e_l: f32,
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(LeakyIzhikevichNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for LeakyIzhikevichNeuron<T, R> {
    fn default() -> Self {
        LeakyIzhikevichNeuron {
            current_voltage: -65., 
            gap_conductance: 7.,
            w_value: 30.,
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            tau_m: 10., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -65., // initial potential (mV)
            e_l: -65., // leak reversal potential (mV)
            w_init: 30., // initial w value
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> LeakyIzhikevichNeuron<T, R> {
    impl_izhikevich_default_methods!();

    /// Calculates the change in voltage given an input current
    pub fn izhikevich_leaky_get_dv_change(&self, i: f32) -> f32 {
        (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - 
            self.w_value * (self.current_voltage - self.e_l) + i
        ) * (self.dt / self.c_m)
    }
}

impl_iterate_and_spike!(
    LeakyIzhikevichNeuron, 
    izhikevich_leaky_get_dv_change, 
    izhikevich_get_dw_change,
    izhikevich_handle_spiking
);

/// A BCM compatible Izhikevich neuron
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct BCMIzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Voltage initialization value (mV) 
    pub v_init: f32, 
    /// Controls speed
    pub a: f32, 
    /// Controls sensitivity to adaptive value
    pub b: f32,
    /// After spike reset value for voltage 
    pub c: f32,
    /// After spike reset value for adaptive value 
    pub d: f32, 
    /// Adaptive value
    pub w_value: f32, 
    /// Adaptive value initialization
    pub w_init: f32, 
    /// Controls conductance of input gap junctions
    pub gap_conductance: f32, 
    /// Membrane time constant (ms)
    pub tau_m: f32, 
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is spiking
    pub is_spiking: bool,
    /// Last timestep the neuron has spiked
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
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl_default_impl_integrate_and_fire!(BCMIzhikevichNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for BCMIzhikevichNeuron<T, R> {
    fn default() -> Self {
        BCMIzhikevichNeuron {
            current_voltage: -65., 
            gap_conductance: 7.,
            w_value: 30.,
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 8.0,
            v_th: 30., // spike threshold (mV)
            tau_m: 1., // membrane time constant (ms)
            c_m: 100., // membrane capacitance (nF)
            v_init: -65., // initial potential (mV)
            w_init: 30., // initial w value
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            average_activity: 0.,
            current_activity: 0.,
            period: 3,
            num_spikes: 0,
            firing_rate_clock: 0.,
            firing_rate_window: 500.,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> BCMIzhikevichNeuron<T, R> {
    impl_izhikevich_default_methods!();

    /// Calculates the change in voltage given an input current
    pub fn izhikevich_get_dv_change(&self, i: f32) -> f32 {
        (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (self.dt / self.c_m)
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for BCMIzhikevichNeuron<T, R> {
    impl_default_neurotransmitter_methods!();

    // activity measured as current voltage - last voltage

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        if self.is_spiking {
            self.num_spikes += 1;
        }
        self.firing_rate_clock += self.dt;
        if self.firing_rate_clock >= self.firing_rate_window {
            self.firing_rate_clock = 0.;
            self.current_activity = self.num_spikes as f32 / (self.firing_rate_window * self.dt);
            self.average_activity -= self.average_activity / self.period as f32;
            self.average_activity += self.current_activity / self.period as f32;
        }

        let dv = self.izhikevich_get_dv_change(input_current);
        let dw = self.izhikevich_get_dw_change();
        self.current_voltage += dv;
        self.w_value += dw;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.izhikevich_handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        if self.is_spiking {
            self.num_spikes += 1;
        }
        self.firing_rate_clock += self.dt;
        if self.firing_rate_clock >= self.firing_rate_window {
            self.firing_rate_clock = 0.;
            self.current_activity = self.num_spikes as f32 / self.firing_rate_window;
            self.average_activity -= self.average_activity / self.period as f32;
            self.average_activity += self.current_activity / self.period as f32;
        }

        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

        let dv = self.izhikevich_get_dv_change(input_current);
        let dw = self.izhikevich_get_dw_change();
        let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;
        self.w_value += dw;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.izhikevich_handle_spiking()
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> BCMActivity for BCMIzhikevichNeuron<T, R> {
    fn get_activity(&self) -> f32 {
        self.current_activity
    }
    
    fn get_averaged_activity(&self) -> f32 {
        self.average_activity
    }
}

#[derive(Clone, IterateAndSpikeBase)]
pub struct SimpleLeakyIntegrateAndFire<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Current voltage (mV)
    pub current_voltage: f32,
    /// Leaky channel conductance (nS)
    pub g: f32,
    /// Reversal potential (mV)
    pub e: f32,
    /// Voltage threshold (mV)
    pub v_th: f32,
    /// Voltage reset value (mV)
    pub v_reset: f32,
    /// Initial voltage value (mV)
    pub v_init: f32,
    /// Gap conductance of input gap junctions
    pub gap_conductance: f32,
    /// Membrane capacitance (nF)
    pub c_m: f32, 
    /// Time step (ms)
    pub dt: f32, 
    /// Whether the neuron is currently spiking
    pub is_spiking: bool,
    /// Last timestep the neuron fired
    pub last_firing_time: Option<usize>,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<IonotropicNeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: LigandGatedChannels<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for SimpleLeakyIntegrateAndFire<T, R> {
    fn default() -> Self {
        SimpleLeakyIntegrateAndFire {
            current_voltage: -75., 
            gap_conductance: 10.,
            v_th: -55., // spike threshold (mV)
            v_reset: -75., // reset potential (mV)
            c_m: 100., // membrane capacitance (nF)
            g: -0.1, // leak conductance (nS)
            v_init: -75., // initial potential (mV)
            e: 0., // leak reversal potential (mV)
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            synaptic_neurotransmitters: Neurotransmitters::<IonotropicNeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl SimpleLeakyIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> {
    pub fn default_impl() -> Self {
        SimpleLeakyIntegrateAndFire::default()
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> SimpleLeakyIntegrateAndFire<T, R> {
    fn handle_spiking(&mut self) -> bool {
        let mut is_spiking = false;

        if self.current_voltage >= self.v_th {
            is_spiking = !is_spiking;
            self.current_voltage = self.v_reset;
        }

        self.is_spiking = is_spiking;

        self.is_spiking
    }

    fn get_dv_change(&self, i: f32) -> f32 {
        (self.g * (self.current_voltage - self.e) + i) * self.dt
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for SimpleLeakyIntegrateAndFire<T, R> {
    impl_default_neurotransmitter_methods!();

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
        let dv = self.get_dv_change(input_current);
        self.current_voltage += dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

        let dv = self.get_dv_change(input_current);
        let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.handle_spiking()
    }
}

#[cfg(feature = "gpu")]
impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpikeGPU for SimpleLeakyIntegrateAndFire<T, R> {
    fn iterate_and_spike_electrical_kernel(context: &Context) -> KernelFunction {
        let kernel_name = String::from("simple_leaky_integrate_and_fire_iterate_and_spike");
        let argument_names = vec![
            String::from("inputs"), String::from("index_to_position"), String::from("current_voltage"), 
            String::from("g"), String::from("e"), String::from("v_th"), 
            String::from("v_reset"), String::from("dt"), String::from("is_spiking"),
        ];

        let program_source = String::from(r#"
            __kernel void simple_leaky_integrate_and_fire_iterate_and_spike(
                __global const float *inputs,
                __global const uint *index_to_position,
                __global float *v,
                __global float *g,
                __global float *e,
                __global float *v_th,
                __global float *v_reset,
                __global uint *is_spiking,
                __global float *dt
            ) {
                int gid = get_global_id(0);
                int index = index_to_position[gid];

                v[index] += (g[index] * (v[index] - e[index]) + inputs[index]) * dt[index];
                if (v[index] >= v_th[index]) {
                    v[index] = v_reset[index];
                    is_spiking[index] = 1;
                } else {
                    is_spiking[index] = 0;
                }
            }
        "#);

        let iterate_and_spike_program = Program::create_and_build_from_source(context, &program_source, "")
            .expect("Program::create_and_build_from_source failed");
        let kernel = Kernel::create(&iterate_and_spike_program, &kernel_name)
            .expect("Kernel::create failed");

        KernelFunction { 
            kernel, 
            program_source, 
            kernel_name, 
            argument_names, 
        }
    }
    
    fn convert_to_gpu(cell_grid: &[Vec<Self>], context: &Context, queue: &CommandQueue) -> HashMap<String, BufferGPU> {
        let mut buffers = HashMap::new();

        create_float_buffer!(current_voltage_buffer, context, queue, cell_grid, current_voltage);
        create_float_buffer!(gap_conductance_buffer, context, queue, cell_grid, gap_conductance);
        create_float_buffer!(g_buffer, context, queue, cell_grid, g);
        create_float_buffer!(e_buffer, context, queue, cell_grid, e);
        create_float_buffer!(v_reset_buffer, context, queue, cell_grid, v_reset);
        create_float_buffer!(dt_buffer, context, queue, cell_grid, dt);
        create_float_buffer!(v_th_buffer, context, queue, cell_grid, v_th);

        create_uint_buffer!(is_spiking_buffer, context, queue, cell_grid, is_spiking, last);

        buffers.insert(String::from("current_voltage"), BufferGPU::Float(current_voltage_buffer));
        buffers.insert(String::from("gap_conductance"), BufferGPU::Float(gap_conductance_buffer));
        buffers.insert(String::from("g"), BufferGPU::Float(g_buffer));
        buffers.insert(String::from("e"), BufferGPU::Float(e_buffer));
        buffers.insert(String::from("v_reset"), BufferGPU::Float(v_reset_buffer));
        buffers.insert(String::from("dt"), BufferGPU::Float(dt_buffer));
        buffers.insert(String::from("v_th"), BufferGPU::Float(v_th_buffer));

        buffers.insert(String::from("is_spiking"), BufferGPU::UInt(is_spiking_buffer));

        buffers
    }

    #[allow(clippy::needless_range_loop)]
    fn convert_to_cpu(
        cell_grid: &mut Vec<Vec<Self>>,
        buffers: HashMap<String, BufferGPU>,
        rows: usize,
        cols: usize,
        queue: &CommandQueue,
    ) {
        let mut current_voltage: Vec<f32> = vec![0.0; rows * cols];
        let mut gap_conductance: Vec<f32> = vec![0.0; rows * cols];
        let mut g: Vec<f32> = vec![0.0; rows * cols];
        let mut e: Vec<f32> = vec![0.0; rows * cols];
        let mut v_reset: Vec<f32> = vec![0.0; rows * cols];
        let mut dt: Vec<f32> = vec![0.0; rows * cols];
        let mut v_th: Vec<f32> = vec![0.0; rows * cols];
        let mut is_spiking: Vec<u32> = vec![0; rows * cols];

        read_and_set_buffer!(buffers, queue, "current_voltage", &mut current_voltage, Float);
        read_and_set_buffer!(buffers, queue, "gap_conductance", &mut gap_conductance, Float);
        read_and_set_buffer!(buffers, queue, "g", &mut g, Float);
        read_and_set_buffer!(buffers, queue, "e", &mut e, Float);
        read_and_set_buffer!(buffers, queue, "dt", &mut dt, Float);
        read_and_set_buffer!(buffers, queue, "v_reset", &mut v_reset, Float);
        read_and_set_buffer!(buffers, queue, "v_th", &mut v_th, Float);
        read_and_set_buffer!(buffers, queue, "is_spiking", &mut is_spiking, UInt);

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let cell = &mut cell_grid[i][j];
                
                cell.current_voltage = current_voltage[idx];
                cell.gap_conductance = gap_conductance[idx];
                cell.g = g[idx];
                cell.e = e[idx];
                cell.v_reset = v_reset[idx];
                cell.dt = dt[idx];
                cell.v_th = v_th[idx];
                cell.is_spiking = is_spiking[idx] == 1;
            }
        }
    }
}
