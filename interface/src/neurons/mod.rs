use spiking_neural_networks::neuron::{
    iterate_and_spike::{
        CurrentVoltage, GapConductance, GaussianFactor, GaussianParameters, IsSpiking, IterateAndSpike, LastFiringTime, NeurotransmitterConcentrations, NeurotransmitterKinetics, NeurotransmitterType, Neurotransmitters, ReceptorKinetics, Timestep
    }, 
    iterate_and_spike_traits::IterateAndSpikeBase
};


#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
#[allow(clippy::upper_case_acronyms)]
pub enum DopaGluGABANeurotransmitterType {
    Dopamine,
    Glutamate,
    GABA,
}

impl NeurotransmitterType for DopaGluGABANeurotransmitterType {}

pub trait GlutamateGABAChannel {
    fn calculate_current(&mut self, voltage: f32) -> f32;
}

#[derive(Clone, Copy, Debug)]
pub struct GlutamateReceptor<T: ReceptorKinetics> {
    pub ampa_g: f32,
    pub inh_modifier: f32,
    pub ampa_receptor: T,
    pub ampa_reversal: f32,
    pub nmda_g: f32,
    pub nmda_modifier: f32,
    pub mg: f32,
    pub nmda_receptor: T,
    pub nmda_reversal: f32,
    pub current: f32,
}

impl<T: ReceptorKinetics> GlutamateGABAChannel for GlutamateReceptor<T> {
    fn calculate_current(&mut self, voltage: f32) -> f32 {
        let ampa_current = self.ampa_g * self.ampa_receptor.get_r() * self.inh_modifier * (voltage - self.ampa_reversal);
        let mg_modifier = 1. / (1. + ((-0.062 * voltage).exp() * self.mg / 3.57));
        let nmda_current = mg_modifier * self.nmda_g * self.nmda_receptor.get_r().powf(self.nmda_modifier) 
            * self.inh_modifier * (voltage - self.nmda_reversal);

        self.current = ampa_current + nmda_current;

        self.current
    }
}

impl<T: ReceptorKinetics> Default for GlutamateReceptor<T> {
    fn default() -> Self {
        GlutamateReceptor {
            ampa_g: 1.,
            inh_modifier: 1.,
            ampa_receptor: T::default(),
            ampa_reversal: 0.,
            nmda_g: 0.6,
            mg: 0.3,
            nmda_modifier: 1.,
            nmda_receptor: T::default(),
            nmda_reversal: 0.,
            current: 0.,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GABAReceptor<T: ReceptorKinetics> {
    pub g: f32,
    pub r: T,
    pub reversal: f32,
    pub current: f32,
}

impl<T: ReceptorKinetics> GlutamateGABAChannel for GABAReceptor<T> {
    fn calculate_current(&mut self, voltage: f32) -> f32 {
        self.current = self.g * self.r.get_r() * (voltage - self.reversal);

        self.current
    }
}

impl<T: ReceptorKinetics> Default for GABAReceptor<T> {
    fn default() -> Self {
        GABAReceptor {
            g: 1.2,
            r: T::default(),
            reversal: -80.,
            current: 0.,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DopamineReceptor<T: ReceptorKinetics> {
    pub d1_r: T,
    pub d1_enabled: bool,
    pub s_d1: f32,
    pub d2_r: T,
    pub d2_enabled: bool,
    pub s_d2: f32,
}

impl<T: ReceptorKinetics> DopamineReceptor<T> {
    pub fn apply_r_changes(&mut self, t: f32, dt: f32) {
        if self.d1_enabled {
            self.d1_r.apply_r_change(t, dt);
        }
        if self.d2_enabled {
            self.d2_r.apply_r_change(t, dt);
        }
    }

    pub fn get_modifiers(&self, inh_modifier: &mut f32, nmda_modifier: &mut f32) {
        let mut d1_modifier = 0.;
        if self.d2_enabled {
            *inh_modifier = 1. - (self.d2_r.get_r().max(0.).min(1.) * self.s_d2);
        }
        if self.d1_enabled {
            d1_modifier = self.d1_r.get_r().max(0.).min(1.) * self.s_d1;
        }
        *nmda_modifier = 1. - d1_modifier;
    }
}

impl<T: ReceptorKinetics> Default for DopamineReceptor<T> {
    fn default() -> Self {
        DopamineReceptor {
            d1_r: T::default(),
            d1_enabled: false,
            s_d1: 1.,
            d2_r: T::default(),
            d2_enabled: false,
            s_d2: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DopaGluGABAReceptors<T: ReceptorKinetics> {
    pub dopamine_receptor: DopamineReceptor<T>,
    pub inh_modifier: f32,
    pub nmda_modifier: f32,
    pub glu_receptor: Option<GlutamateReceptor<T>>,
    pub gaba_receptor: Option<GABAReceptor<T>>,
}

impl<T: ReceptorKinetics> DopaGluGABAReceptors<T> {
    fn update_receptor_kinetics(
        &mut self,
        t_total: &NeurotransmitterConcentrations<DopaGluGABANeurotransmitterType>,
        dt: f32,
    ) {
        if let Some(glutamate_concentration) = t_total.get(&DopaGluGABANeurotransmitterType::Glutamate) {
            if let Some(ref mut glu) = self.glu_receptor {
                glu.ampa_receptor
                    .apply_r_change(*glutamate_concentration, dt);
                glu.nmda_receptor
                    .apply_r_change(*glutamate_concentration, dt);
            }
        }
    
        if let Some(gaba_concentration) = t_total.get(&DopaGluGABANeurotransmitterType::GABA) {
            if let Some(ref mut gaba) = self.gaba_receptor {
                gaba.r.apply_r_change(*gaba_concentration, dt);
            }
        }
    
        if let Some(dopamine_concentration) = t_total.get(&DopaGluGABANeurotransmitterType::Dopamine) {
            self.dopamine_receptor
                .apply_r_changes(*dopamine_concentration, dt);
        }
    }
    
    fn set_receptor_currents(&mut self, voltage: f32) {
        self.dopamine_receptor
            .get_modifiers(&mut self.inh_modifier, &mut self.nmda_modifier);
    
        if let Some(ref mut glu) = self.glu_receptor {
            glu.inh_modifier = self.inh_modifier;
            glu.nmda_modifier = self.nmda_modifier;
            let _ = glu.calculate_current(voltage);
        }
    
        if let Some(ref mut gaba) = self.gaba_receptor {
            let _ = gaba.calculate_current(voltage);
        }
    }
    

    fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32 {
        let mut current = 0.;

        match &self.glu_receptor {
            Some(glu) => {
                current += glu.current;
            },
            None => {},
        }

        match &self.gaba_receptor {
            Some(gaba) => {
                current += gaba.current;
            },
            None => {},
        }

        current * (dt / c_m)
    }
}

impl<T: ReceptorKinetics> Default for DopaGluGABAReceptors<T> {
    fn default() -> Self {
        DopaGluGABAReceptors {
            dopamine_receptor: DopamineReceptor::<T>::default(),
            inh_modifier: 0.,
            nmda_modifier: 0.,
            glu_receptor: None,
            gaba_receptor: None,
        }
    }
}

/// An Izhikevich neuron that has D1 and D2 receptors
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct DopaIzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    /// Membrane potential (mV)
    pub current_voltage: f32, 
    /// Voltage threshold (mV)
    pub v_th: f32,
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
    /// Gaussian parameters
    pub gaussian_params: GaussianParameters,
    /// Postsynaptic neurotransmitters in cleft
    pub synaptic_neurotransmitters: Neurotransmitters<DopaGluGABANeurotransmitterType, T>,
    /// Dopamine, glutamate, and GABA receptors
    pub receptors: DopaGluGABAReceptors<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for DopaIzhikevichNeuron<T, R> {
    fn default() -> Self {
        DopaIzhikevichNeuron {
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
            dt: 0.1, // simulation time step (ms)
            is_spiking: false,
            last_firing_time: None,
            gaussian_params: GaussianParameters::default(),
            synaptic_neurotransmitters: Neurotransmitters::<DopaGluGABANeurotransmitterType, T>::default(),
            receptors: DopaGluGABAReceptors::<R>::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> DopaIzhikevichNeuron<T, R> {
    // Calculates how adaptive value changes
    pub fn izhikevich_get_dw_change(&self) -> f32 {
        (
            self.a * (self.b * self.current_voltage - self.w_value)
        ) * (self.dt / self.tau_m)
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

    /// Calculates the change in voltage given an input current
    pub fn izhikevich_get_dv_change(&self, i: f32) -> f32 {
        (
            0.04 * self.current_voltage.powf(2.0) + 
            5. * self.current_voltage + 140. - self.w_value + i
        ) * (self.dt / self.c_m)
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for DopaIzhikevichNeuron<T, R> {
    type N = DopaGluGABANeurotransmitterType;

    fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_and_spike(&mut self, input_current: f32) -> bool {
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
        self.receptors.update_receptor_kinetics(t_total, self.dt);
        self.receptors.set_receptor_currents(self.current_voltage);

        let dv = self.izhikevich_get_dv_change(input_current);
        let dw = self.izhikevich_get_dw_change();
        let neurotransmitter_dv = -self.receptors.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;
        self.w_value += dw;

        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage, self.dt);

        self.izhikevich_handle_spiking()
    }
}

// impl rk4 function too
