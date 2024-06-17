
use std::{
    f64::consts::E, 
    fs::File, 
    io::{BufWriter, Write},
    collections::HashMap,
    ops::Sub,
};
use super::{ 
    iterate_and_spike::{
        ReceptorKinetics, BayesianFactor, BayesianParameters, 
        CurrentVoltage, GapConductance, IterateAndSpike, LastFiringTime, 
        Potentiation, PotentiationType, STDPParameters, STDP,
        LigandGatedChannels, NeurotransmitterKinetics, NeurotransmitterType, Neurotransmitters
    },
    impl_bayesian_factor_with_kinetics, 
    impl_current_voltage_with_kinetics, 
    impl_gap_conductance_with_kinetics, 
    impl_last_firing_time_with_kinetics, 
    impl_potentiation_with_kinetics, 
    impl_stdp_with_kinetics, 
    impl_necessary_iterate_and_spike_traits, 
};


// CHECK THIS PAPER TO CREATE MORE ION CHANNELS WHEN REFACTORING
// https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/25282547/

// https://webpages.uidaho.edu/rwells/techdocs/Biological%20Signal%20Processing/Chapter%2004%20The%20Biological%20Neuron.pdf

// https://www.nature.com/articles/356441a0.pdf : calcium currents paper
// https://github.com/ModelDBRepository/151460/blob/master/CaT.mod // low threshold calcium current
// https://modeldb.science/279?tab=1 // low threshold calcium current (thalamic)
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod // high threshold calcium current (l type)
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9373714/ // assume [Ca2+]in,inf is initial [Ca2+] value

// l-type ca2+ channel, ca1.2
#[derive(Clone, Copy)]
pub struct HighThresholdCalciumChannel {
    current: f64,
    z: f64,
    f: f64,
    // r: f64,
    // temp: f64,
    ca_in: f64,
    ca_in_equilibrium: f64,
    ca_out: f64,
    // permeability: f64,
    max_permeability: f64,
    d: f64,
    kt: f64,
    kd: f64,
    tr: f64,
    k: f64,
    p: f64,
    v_th: f64,
    s: f64,
    m_ca: f64,
    alpha: f64,
    beta: f64, 
}

impl Default for HighThresholdCalciumChannel {
    fn default() -> Self {
        HighThresholdCalciumChannel {
            current: 0.,
            z: 2.,
            f: 96489., // C/mol
            // r: 8.31, // J/Kmol
            // temp: 35., // degrees c
            ca_in: 0.001, // mM
            ca_in_equilibrium: 0.001, // mM
            ca_out: 5., // mM
            // permeability: 0.,
            max_permeability: 5.36e-6,
            d: 0.1, // um
            kt: 1e-4, // mM / ms
            kd: 1e-4, // mM
            tr: 43., // ms
            k: 1000.,
            p: 0.02,
            v_th: (9. * 297.) / (2. * E),
            s: 1.,
            m_ca: 0.,
            alpha: 0.,
            beta: 0.,
        }
    }
}

impl HighThresholdCalciumChannel {
    // m^x * n^y
    // x and y here probably refer to 3 and 4
    // fn update_permeability(&mut self, m_state: f64, n_state: f64) {
    //     self.permeability = self.max_permeability * m_state * n_state;
    // }

    fn update_ca_in(&mut self, dt: f64) {
        let term1 = self.k * (-self.current / (2. * self.f * self.d));
        let term2 = self.p * ((self.kt * self.ca_in) / (self.ca_in + self.kd));
        let term3 = (self.ca_in_equilibrium - self.ca_in) / self.tr;
        self.ca_in += (term1 + term2 + term3) * dt;
    }

    fn update_m_ca(&mut self, voltage: f64) {
        self.alpha += 1.6 / (1. + (-0.072 * (voltage - 5.)).exp());
        self.beta += (0.02 * (voltage - 1.31)) / (((voltage - 1.31) / 5.36).exp() - 1.);
        self.m_ca = self.alpha / (self.alpha + self.beta);
    }

    fn get_ca_current(&self, voltage: f64) -> f64 {
        let term1 = self.m_ca.powf(2.) * self.max_permeability * self.s;
        let term2 = (self.z * self.f) / self.v_th;
        let term3 = voltage / self.v_th.exp();
        let term4 = self.ca_in_equilibrium * term3 - self.ca_out;
        let term5 = term3 - 1.;

        term1 * term2 * (term4 / term5) * voltage
    }

    fn get_ca_current_and_update(&mut self, voltage: f64, dt: f64) -> f64 {
        self.update_ca_in(dt);
        self.update_m_ca(voltage);
        self.current = self.get_ca_current(voltage);

        self.current
    }
}

#[derive(Clone, Copy)]
pub struct HighVoltageActivatedCalciumChannel {
    m: f64,
    m_a: f64,
    m_b: f64,
    h: f64,
    h_a: f64,
    h_b: f64,
    gca_bar: f64,
    ca_rev: f64,
    current: f64,
}

impl Default for HighVoltageActivatedCalciumChannel {
    fn default() -> Self {
        HighVoltageActivatedCalciumChannel {
            m: 0.,
            m_a: 0.,
            m_b: 0.,
            h: 0.,
            h_a: 0.,
            h_b: 0.,
            gca_bar: 1e-4,
            ca_rev: 80.,
            current: 0.,
        }
    }
}

// https://github.com/ModelDBRepository/121060/blob/master/chan_CaL12.mod
// https://github.com/gpapamak/snl/blob/master/IL_gutnick.mod
impl HighVoltageActivatedCalciumChannel {
    fn update_m(&mut self, voltage: f64) {
        self.m_a = 0.055 * (-27. - voltage) / (((-27. - voltage) / 3.8).exp() - 1.);
        self.m_b = 0.94 * ((-75. - voltage) / 17.).exp();
    }

    fn update_h(&mut self, voltage: f64) {
        self.h_a = 0.000457 * ((-13. - voltage) / 50.).exp();
        self.h_b = 0.0065 / (((-15. - voltage) / 28.).exp() + 1.);
    }

    fn initialize_m_and_h(&mut self, voltage: f64) {
        self.update_m(voltage);
        self.update_h(voltage);

        self.m = self.m_a / (self.m_a + self.m_b);
        self.h = self.h_a / (self.h_a + self.h_b);
    } 

    fn update_m_and_h_states(&mut self, voltage: f64, dt: f64) {
        self.update_m(voltage);
        self.update_h(voltage);

        self.m += (self.m_a * (1. - self.m) - (self.m_b * self.m)) * dt;
        self.h += (self.h_a * (1. - self.h) - (self.h_b * self.h)) * dt;
    }

    fn get_ca_and_update_current(&mut self, voltage: f64, dt: f64) -> f64 {
        self.update_m_and_h_states(voltage, dt);
        self.current = self.gca_bar * self.m.powf(2.) * self.h * (voltage - self.ca_rev);

        self.current
    }
}

// ** REFACTOR THIS INTO A TRAIT **

// can look at this
// https://github.com/JoErNanO/brianmodel/blob/master/brianmodel/neuron/ioniccurrent/ioniccurrentcal.py
#[derive(Clone, Copy)]
pub enum AdditionalGates {
    LTypeCa(HighThresholdCalciumChannel),
    HVACa(HighVoltageActivatedCalciumChannel), // https://neuronaldynamics.epfl.ch/online/Ch2.S3.html // https://sci-hub.se/https://pubmed.ncbi.nlm.nih.gov/8229187/
    // OscillatingCa(OscillatingCalciumChannel),
    // PotassiumRectifying(KRectifierChannel),
}

impl AdditionalGates {
    pub fn initialize(&mut self, voltage: f64) {
        match self {
            AdditionalGates::LTypeCa(_) => {}, // rewrite this such that m_ca state is initialized
            AdditionalGates::HVACa(channel) => channel.initialize_m_and_h(voltage),
        }
    }

    pub fn get_and_update_current(&mut self, voltage: f64, dt: f64) -> f64 {
        match self {
            AdditionalGates::LTypeCa(channel) => channel.get_ca_current_and_update(voltage, dt),
            AdditionalGates::HVACa(channel) => channel.get_ca_and_update_current(voltage, dt),
        }
    }

    pub fn get_current(&self) -> f64 {
        match &self {
            AdditionalGates::LTypeCa(channel) => channel.current,
            AdditionalGates::HVACa(channel) => channel.current,
        }
    }

    pub fn to_str(&self) -> &str {
        match &self {
            AdditionalGates::LTypeCa(_) => "LTypeCa",
            AdditionalGates::HVACa(_) => "HVA LTypeCa",
        }
    }
}

// pub trait AdditionalGate {
//     fn initialize(&mut self, voltage: f64);
//     fn update_current(voltage: f64);
//     fn get_current(&self) -> f64;
// }

// multicomparment stuff, refer to dopamine modeling paper as well
// https://github.com/antgon/msn-model/blob/main/msn/cell.py 
// https://github.com/jrieke/NeuroSim
// MULTICOMPARTMENT EXPLAINED
// https://neuronaldynamics.epfl.ch/online/Ch3.S2.html
// pub struct Soma {

// }

// pub struct Dendrite {

// }

#[derive(Clone, Copy)]
pub struct Gate {
    pub alpha: f64,
    pub beta: f64,
    pub state: f64,
}

impl Gate {
    pub fn init_state(&mut self) {
        self.state = self.alpha / (self.alpha + self.beta);
    }

    pub fn update(&mut self, dt: f64) {
        let alpha_state: f64 = self.alpha * (1. - self.state);
        let beta_state: f64 = self.beta * self.state;
        self.state += dt * (alpha_state - beta_state);
    }
}

#[derive(Clone)]
pub struct HodgkinHuxleyNeuron<T: NeurotransmitterKinetics, R: ReceptorKinetics> {
    pub current_voltage: f64,
    pub gap_conductance: f64,
    pub potentiation_type: PotentiationType,
    pub dt: f64,
    pub c_m: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_k_leak: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_k_leak: f64,
    pub m: Gate,
    pub n: Gate,
    pub h: Gate,
    pub v_th: f64,
    pub last_firing_time: Option<usize>,
    pub was_increasing: bool,
    pub is_spiking: bool,
    pub additional_gates: Vec<AdditionalGates>,
    pub synaptic_neurotransmitters: Neurotransmitters<T>,
    pub ligand_gates: LigandGatedChannels<R>,
    pub bayesian_params: BayesianParameters,
    pub stdp_params: STDPParameters,
}

impl_necessary_iterate_and_spike_traits!(HodgkinHuxleyNeuron);

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for HodgkinHuxleyNeuron<T, R> {
    fn default() -> Self {
        let default_gate = Gate {
            alpha: 0.,
            beta: 0.,
            state: 0.,
        };

        HodgkinHuxleyNeuron { 
            current_voltage: 0.,
            gap_conductance: 7.,
            potentiation_type: PotentiationType::Excitatory,
            dt: 0.1,
            c_m: 1., 
            e_na: 115., 
            e_k: -12., 
            e_k_leak: 10.6, 
            g_na: 120., 
            g_k: 36., 
            g_k_leak: 0.3, 
            m: default_gate.clone(), 
            n: default_gate.clone(), 
            h: default_gate,  
            v_th: 60.,
            last_firing_time: None,
            is_spiking: false,
            was_increasing: false,
            synaptic_neurotransmitters: Neurotransmitters::default(), 
            ligand_gates: LigandGatedChannels::default(),
            additional_gates: vec![],
            bayesian_params: BayesianParameters::default(),
            stdp_params: STDPParameters::default(),
        }
    }
}

// find peaks of hodgkin huxley
// result starts at index 1 of input list
pub fn diff<T: Sub<Output = T> + Copy>(x: &Vec<T>) -> Vec<T> {
    (1..x.len()).map(|i| x[i] - x[i-1])
        .collect()
}

pub fn find_peaks(voltages: &Vec<f64>, tolerance: f64) -> Vec<usize> {
    let first_diff: Vec<f64> = diff(&voltages);
    let second_diff: Vec<f64> = diff(&first_diff);

    let local_optima = first_diff.iter()
        .enumerate()
        .filter(|(_, i)| i.abs() <= tolerance)
        .map(|(n, i)| (n, *i))
        .collect::<Vec<(usize, f64)>>();

    let local_maxima = local_optima.iter()
        .map(|(n, i)| (*n, *i))
        .filter(|(n, _)| *n < second_diff.len() - 1 && second_diff[n+1] < 0.)
        .collect::<Vec<(usize, f64)>>();

    let local_maxima: Vec<usize> = local_maxima.iter()
        .map(|(n, _)| (n + 2))
        .collect();

    let mut peak_spans: Vec<Vec<usize>> = Vec::new();

    let mut index: usize = 0;
    for (n, i) in local_maxima.iter().enumerate() {
        if n > 0 && local_maxima[n] - local_maxima[n-1] != 1 {
            index += 1;
        }

        if peak_spans.len() - 1 != index {
            peak_spans.push(Vec::new());
        }

        peak_spans[index].push(*i);
    }

    peak_spans.iter()
        .map(|i| i[i.len() / 2])
        .collect::<Vec<usize>>()
}

// https://github.com/swharden/pyHH/blob/master/src/pyhh/models.py
impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> HodgkinHuxleyNeuron<T, R> {
    pub fn update_gate_time_constants(&mut self, voltage: f64) {
        self.n.alpha = 0.01 * (10. - voltage) / (((10. - voltage) / 10.).exp() - 1.);
        self.n.beta = 0.125 * (-voltage / 80.).exp();
        self.m.alpha = 0.1 * ((25. - voltage) / (((25. - voltage) / 10.).exp() - 1.));
        self.m.beta = 4. * (-voltage / 18.).exp();
        self.h.alpha = 0.07 * (-voltage / 20.).exp();
        self.h.beta = 1. / (((30. - voltage) / 10.).exp() + 1.);
    }

    pub fn initialize_parameters(&mut self, starting_voltage: f64) {
        self.current_voltage = starting_voltage;
        self.update_gate_time_constants(starting_voltage);
        self.m.init_state();
        self.n.init_state();
        self.h.init_state();

        self.additional_gates.iter_mut()
            .for_each(|i| i.initialize(starting_voltage));
    }

    pub fn update_cell_voltage(&mut self, input_current: f64) {
        let i_na = self.m.state.powf(3.) * self.g_na * self.h.state * (self.current_voltage - self.e_na);
        let i_k = self.n.state.powf(4.) * self.g_k * (self.current_voltage - self.e_k);
        let i_k_leak = self.g_k_leak * (self.current_voltage - self.e_k_leak);

        let i_ligand_gates = self.ligand_gates.get_receptor_currents(self.dt, self.c_m);

        let i_additional_gates = self.additional_gates
            .iter_mut()
            .map(|i| 
                i.get_and_update_current(self.current_voltage, self.dt)
            ) 
            .sum::<f64>();

        let i_sum = input_current - (i_na + i_k + i_k_leak) + i_ligand_gates + i_additional_gates;
        self.current_voltage += self.dt * i_sum / self.c_m;
    }

    pub fn update_neurotransmitters(&mut self) {
        self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);
    }

    pub fn update_receptors(
        &mut self, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>
    ) {
        self.ligand_gates.update_receptor_kinetics(t_total);
        self.ligand_gates.set_receptor_currents(self.current_voltage);
    }

    pub fn update_gate_states(&mut self) {
        self.m.update(self.dt);
        self.n.update(self.dt);
        self.h.update(self.dt);
    }

    pub fn iterate(&mut self, input: f64) {
        self.update_gate_time_constants(self.current_voltage);
        self.update_cell_voltage(input);
        self.update_gate_states();
        self.update_neurotransmitters();
    }

    pub fn iterate_with_neurotransmitter(
        &mut self, 
        input: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>
    ) {
        self.update_receptors(t_total);
        self.iterate(input);
    }

    pub fn run_static_input(
        &mut self, 
        input: f64, 
        bayesian: bool, 
        iterations: usize, 
        filename: &str, 
        full: bool,
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        if !full {
            writeln!(file, "voltage").expect("Unable to write to file");
            writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
        } else {
            write!(file, "voltage,m,n,h").expect("Unable to write to file");
            writeln!(
                file, 
                ",{}",
                self.additional_gates.iter()
                    .map(|x| x.to_str())
                    .collect::<Vec<&str>>()
                    .join(",")
            ).expect("Unable to write to file");
            write!(file, "{}, {}, {}, {}", 
                self.current_voltage, 
                self.m.state, 
                self.n.state, 
                self.h.state,
            ).expect("Unable to write to file");
            writeln!(
                file, 
                ", {}",
                self.additional_gates.iter()
                    .map(|x| x.get_current().to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            ).expect("Unable to write to file");
        }

        self.initialize_parameters(self.current_voltage);
        
        for _ in 0..iterations {
            if bayesian {
                self.iterate(
                    input * self.get_bayesian_factor()
                );
            } else {
                self.iterate(input);
            }

            if !full {
                writeln!(file, "{}", self.current_voltage).expect("Unable to write to file");
            } else {
                write!(file, "{}, {}, {}, {}", 
                    self.current_voltage, 
                    self.m.state, 
                    self.n.state, 
                    self.h.state,
                ).expect("Unable to write to file");
                writeln!(
                    file, 
                    ", {}",
                    self.additional_gates.iter()
                        .map(|x| x.get_current().to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                ).expect("Unable to write to file");
            }
        }
    }

    pub fn peaks_test(
        &mut self, 
        input: f64, 
        bayesian: bool, 
        iterations: usize, 
        tolerance: f64,
        filename: &str, 
    ) {
        let mut file = BufWriter::new(File::create(filename)
            .expect("Unable to create file"));
        
        let mut voltages: Vec<f64> = vec![self.current_voltage];

        for _ in 0..iterations {
            if bayesian {
                let bayesian_factor = self.get_bayesian_factor();
                let bayesian_input = input * bayesian_factor;

                self.iterate(bayesian_input);
            } else {
                self.iterate(input);
            }

            voltages.push(self.current_voltage);
        }

        let peaks = find_peaks(&voltages, tolerance);

        writeln!(file, "voltages,peak").expect("Could not write to file");
        for (n, i) in voltages.iter().enumerate() {
            let is_peak: &str = if peaks.contains(&n) {
                "true"
            } else {
                "false"
            };

            writeln!(file, "{},{}", i, is_peak).expect("Could not write to file");
        }
    }
}

impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for HodgkinHuxleyNeuron<T, R> {
    type T = T;
    type R = R;

    fn iterate_and_spike(&mut self, input_current: f64) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate(input_current);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed  && self.was_increasing && !increasing_right_now;
        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }

    fn get_ligand_gates(&self) -> &LigandGatedChannels<R> {
        &self.ligand_gates
    }

    fn get_neurotransmitters(&self) -> &Neurotransmitters<T> {
        &self.synaptic_neurotransmitters
    }

    fn get_neurotransmitter_concentrations(&self) -> HashMap<NeurotransmitterType, f64> {
        self.synaptic_neurotransmitters.get_concentrations()
    }

    fn iterate_with_neurotransmitter_and_spike(
        &mut self, 
        input_current: f64, 
        t_total: Option<&HashMap<NeurotransmitterType, f64>>,
    ) -> bool {
        let last_voltage = self.current_voltage;
        self.iterate_with_neurotransmitter(input_current, t_total);

        let increasing_right_now = last_voltage < self.current_voltage;
        let threshold_crossed = self.current_voltage > self.v_th;
        let is_spiking = threshold_crossed  && self.was_increasing && !increasing_right_now;

        self.is_spiking = is_spiking;
        self.was_increasing = increasing_right_now;

        is_spiking
    }
}
