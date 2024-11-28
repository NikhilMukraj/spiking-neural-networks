#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum DopaGluGABANeurotransmitterType {
    Dopamine,
    Glutamate,
    GABA,
}

trait GlutamateGabaChannel {
    fn calculate_current(&mut self, voltage: f32) -> f32;
}

pub struct GlutamateReceptor<T: ReceptorKinetics> {
    ampa_g: f32,
    ampa_modifier: f32,
    ampa_receptor: T,
    ampa_reversal: f32,
    nmda_g: f32,
    nmda_modifier: f32,
    nmda_receptor: T,
    nmda_reversal: f32,
    current: f32,
}

impl<T: ReceptorKinetics> GlutamateGabaChannel for GlutamateReceptor<T> {
    fn calculate_current(&mut self, voltage: f32) {
        let ampa_current = self.ampa_g * self.ampa_receptor.get_r().powf(self.ampa_modifier) * (voltage - self.ampa_reversal);
        let nmda_current = self.nmda_g * self.nmda_receptor.get_r().powf(self.nmda_modifier) * (voltage - self.nmda_reversal);

        self.current = ampa_current + nmda_current;

        self.current
    }
}

impl<T: ReceptorKinetics> Default for GlutamateGabaChannel {
    fn default() -> Self {
        GlutamateGabaChannel {
            ampa_g: 1.2,
            ampa_modifier: 1.,
            ampa_receptor: T::default(),
            ampa_reversal: 0.,
            nmda_g: 0.6,
            nmda_modifier: 1.,
            nmda_receptor: T::default(),
            nmda_reversal: 0.,
            current: 0.,
        }
    }
}

pub struct GABAReceptor<T: ReceptorKinetics> {
    g: f32,
    r: T,
    reversal: f32,
    current: f32,
}

impl<T: ReceptorKinetics> GABAReceptor for GlutamateReceptor<T> {
    fn calculate_current(&mut self, voltage: f32) {
        self.current = self.g * self.r.get_r() * (voltage - self.reversal);

        self.current
    }
}

impl<T: ReceptorKinetics> Default for GABAReceptor {
    fn default() -> Self {
        GABAReceptor {
            g: 0.,
            r: T::default(),
            reversal: -80.,
            current: 0.,
        }
    }
}

pub struct DopamineReceptor<T: ReceptorKinetics> {
    d1_r: T,
    d1_enabled: bool,
    d2_r: T,
    d2_enabled: bool,
}

impl<T: ReceptorKinetics> DopamineReceptor<T> {
    fn apply_r_changes(&mut self, t: f32) {
        if self.d1_enabled {
            self.d1_r.apply_r_change(t);
        }
        if self.d2_enabled {
            self.d2_r.apply_r_change(t);
        }
    }

    fn get_modifiers(&self, ampa_modifier: &mut f32, nmda_modifier: &mut f32) {
        let mut d1_modifier = 0.;
        let mut d2_modifier = 0.;
        if self.d2_enabled {
            ampa_modifier = 1. + self.d2_r.get_r();
            d2_modifier = self.d2_r.get_r();
        }
        if self.d1_enabled {
            d1_modifier = self.d1_r.get_r() * 0.5;
        }
        nmda_modifier = 1. - d1_modifier + d2_modifier;
    }
}

pub struct DopaGluGabaReceptors<T: ReceptorKinetics> {
    dopamine_receptor: DopamineReceptor<T>,
    ampa_modifier: f32,
    nmda_modifier: f32,
    glu_receptor: Option<GlutamateReceptor<T>>,
    gaba_receptor: Option<GlutamateReceptor<T>>,
}

impl<T: ReceptorKinetics> DopaGluGabaReceptors<T> {
    fn update_receptor_kinetics(&mut self, t_total: &NeurotransmitterConcentrations<DopaGluGABANeurotransmitterType>) {
        if t_total.contains(&DopaGluGABANeurotransmitterType::Glutamate) {
            match self.glu_receptor {
                Some(glu) => {
                    glu.ampa_receptor.apply_r_change(t_total.get(&DopaGluGABANeurotransmitterType::Glutamate));
                    glu.nmda_receptor.apply_r_change(t_total.get(&DopaGluGABANeurotransmitterType::Glutamate));
                },
                None => {}
            }
        }
        if t_total.contains(&DopaGluGABANeurotransmitterType::GABA) {
            match self.gaba_receptor {
                Some(gaba) => {
                    gaba.r.apply_r_change(t_total.get(&DopaGluGABANeurotransmitterType::GABA))
                },
                None => {}
            }
        }
        if t_total.contains(&DopaGluGABANeurotransmitterType::Dopamine) {
            self.dopamine_receptor.apply_r_changes(t_total.get(&DopaGluGABANeurotransmitterType::Dopamine));
        }
    }

    fn set_receptor_currents(&mut self, voltage: f32) {
        self.dopamine_receptor.get_modifiers(&mut ampa_modifier, &mut nmda_modifier);

        match self.glu_receptor {
            Some(glu) => {
                glu.ampa_modifier = self.ampa_modifier;
                glu.nmda_modifier = self.nmda_modifier;

                let _ = glu.calculate_current(voltage);
            },
            None => {}
        }

        match self.gaba_receptor {
            Some(gaba) => {
                let _ = gaba.calculate_current(voltage);
            }
        }
    }

    fn get_receptor_currents(&self) {
        let mut current = 0.;

        match self.glu_receptor {
            Some(glu) => {
                current += glu.current;
            },
            None => {}
        }

        match self.gaba_receptor {
            Some(gaba) => {
                current += gaba.current;
            }
        }

        current
    }
}

/// An Izhikevich neuron that has D1 and D2 receptors
#[derive(Debug, Clone, IterateAndSpikeBase)]
pub struct DopaIzhikevichNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
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
    pub synaptic_neurotransmitters: Neurotransmitters<DopaGluGABANeurotransmitterType, T>,
    /// Ionotropic receptor ligand gated channels
    pub ligand_gates: DopaGluGabaReceptors<R>,
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for DopaIzhikevichNeuron<T, R> {
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
            synaptic_neurotransmitters: Neurotransmitters::<DopaGluGABANeurotransmitterType, T>::default(),
            ligand_gates: LigandGatedChannels::default(),
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IzhikevichNeuron<T, R> {
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

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        self.izhikevich_handle_spiking()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f32, 
        t_total: &NeurotransmitterConcentrations<Self::N>,
    ) -> bool {
        self.ligand_gates.update_receptor_kinetics(t_total, self.dt);
        self.ligand_gates.set_receptor_currents(self.current_voltage, self.dt);

        let dv = self.izhikevich_get_dv_change(input_current);
        let dw = self.izhikevich_get_dw_change();
        let neurotransmitter_dv = -self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        self.current_voltage += dv + neurotransmitter_dv;
        self.w_value += dw;

        self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

        self.izhikevich_handle_spiking()
    }
}

// impl rk4 function too
