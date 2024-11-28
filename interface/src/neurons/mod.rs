#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum DopaGluGabaNeurotransmitterType {
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
    nmda_modifier: f32
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

pub struct GABAReceptor<T: ReceptorKinetics> {
    g: f32,
    r: T,
    reversal: f32,
    current: f32,
}

impl<T: ReceptorKinetics> GlutamateGabaChannel for GlutamateReceptor<T> {
    fn calculate_current(&mut self, voltage: f32) {
        self.current = self.g * self.r.get_r() * (voltage - self.reversal);

        self.current
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

pub struct DopaGluGabaReceptors<T: ReceptorKinetics {
    dopamine_receptor: DopamineReceptor<T>,
    ampa_modifier: f32,
    nmda_modifier: f32,
    glu_receptor: Option<GlutamateReceptor<T>>,
    gaba_receptor: Option<GlutamateReceptor<T>>,
}

impl<T: ReceptorKinetics> DopaGluGabaReceptors<T> {
    fn update_receptor_kinetics(&mut self, t_total: &NeurotransmitterConcentrations<DopaGluGabaNeurotransmitterType>) {
        if t_total.contains(&DopaGluGabaNeurotransmitterType::Glutamate) {
            match self.glu_receptor {
                Some(glu) => {
                    glu.ampa_receptor.apply_r_change(t_total.get(&DopaGluGabaNeurotransmitterType::Glutamate));
                    glu.nmda_receptor.apply_r_change(t_total.get(&DopaGluGabaNeurotransmitterType::Glutamate));
                },
                None => {}
            }
        }
        if t_total.contains(&DopaGluGabaNeurotransmitterType::GABA) {
            match self.gaba_receptor {
                Some(gaba) => {
                    gaba.r.apply_r_change(t_total.get(&DopaGluGabaNeurotransmitterType::GABA))
                },
                None => {}
            }
        }
        if t_total.contains(&DopaGluGabaNeurotransmitterType::Dopamine) {
            self.dopamine_receptor.apply_r_changes(t_total.get(&DopaGluGabaNeurotransmitterType::Dopamine));
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
