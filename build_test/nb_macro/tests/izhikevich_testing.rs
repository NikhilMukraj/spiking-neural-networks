mod izhikevich_dopamine;

#[cfg(test)]
mod test {
    use crate::izhikevich_dopamine::{BoundedNeurotransmitterKinetics, BoundedReceptorKinetics, DopaGluGABA, DopaGluGABANeurotransmitterType, DopaGluGABAType, GlutamateReceptor, IzhikevichNeuron};
    use spiking_neural_networks::neuron::{
        intermediate_delegate::NeurotransmittersIntermediate, 
        iterate_and_spike::{
            CurrentVoltage, GapConductance, IonotropicReception, IsSpiking, IterateAndSpike, 
            LastFiringTime, NeurotransmitterConcentrations, NeurotransmitterKinetics, Neurotransmitters, 
            ReceptorKinetics, Receptors, Timestep
        }, 
        iterate_and_spike_traits::IterateAndSpikeBase, 
        spike_train::{DeltaDiracRefractoriness, PoissonNeuron, SpikeTrain}
    };


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
        /// Postsynaptic neurotransmitters in cleft
        pub synaptic_neurotransmitters: Neurotransmitters<DopaGluGABANeurotransmitterType, T>,
        /// Dopamine, glutamate, and GABA receptors
        pub receptors: DopaGluGABA<R>,
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
                synaptic_neurotransmitters: Neurotransmitters::<DopaGluGABANeurotransmitterType, T>::default(),
                receptors: DopaGluGABA::<R>::default(),
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

            self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

            self.izhikevich_handle_spiking()
        }

        fn iterate_with_neurotransmitter_and_spike(
            &mut self, 
            input_current: f32, 
            t_total: &NeurotransmitterConcentrations<Self::N>,
        ) -> bool {
            self.receptors.update_receptor_kinetics(t_total, self.dt);
            self.receptors.set_receptor_currents(self.current_voltage, self.dt);

            let dv = self.izhikevich_get_dv_change(input_current);
            let dw = self.izhikevich_get_dw_change();
            let neurotransmitter_dv = -self.receptors.get_receptor_currents(self.dt, self.c_m);

            self.current_voltage += dv + neurotransmitter_dv;
            self.w_value += dw;

            self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));

            self.izhikevich_handle_spiking()
        }
    } 

    const ITERATIONS: usize = 10_000;

    #[test]
    fn test_voltages() {
        let inputs: Vec<f32> = (0..100).map(|i| i as f32 / 2.).collect();

        for i in inputs {
            let mut neuron = IzhikevichNeuron {
                c_m: 100.,
                current_voltage: -65.,
                ..IzhikevichNeuron::default_impl()
            };
            let mut ref_neuron: DopaIzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics> = DopaIzhikevichNeuron {
                c_m: 100.,
                current_voltage: -65.,
                ..DopaIzhikevichNeuron::default()
            };

            for _ in 0..ITERATIONS {
                let _ = neuron.iterate_and_spike(i);
                let _ = ref_neuron.iterate_and_spike(i);

                assert!((neuron.current_voltage - ref_neuron.current_voltage).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_is_spiking() {
        let inputs: Vec<f32> = (0..100).map(|i| i as f32 / 2.).collect();

        for i in inputs {
            let mut neuron = IzhikevichNeuron {
                c_m: 100.,
                current_voltage: -65.,
                ..IzhikevichNeuron::default_impl()
            };
            let mut ref_neuron: DopaIzhikevichNeuron<BoundedNeurotransmitterKinetics, BoundedReceptorKinetics> = DopaIzhikevichNeuron {
                c_m: 100.,
                current_voltage: -65.,
                ..DopaIzhikevichNeuron::default()
            };

            for _ in 0..ITERATIONS {
                let is_spiking = neuron.iterate_and_spike(i);
                let ref_is_spiking = ref_neuron.iterate_and_spike(i);

                assert_eq!(is_spiking, ref_is_spiking);
            }
        }
    }

    #[test]
    fn test_coupling() {
        let rates = [0.0001, 0.001, 0.01];

        let mut spike_counts = [0, 0, 0];

        for (n, rate) in rates.iter().enumerate() {
            let mut spike_train: PoissonNeuron<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness> = PoissonNeuron {
                chance_of_firing: *rate,
                ..PoissonNeuron::default()
            };

            spike_train.synaptic_neurotransmitters.insert(
                DopaGluGABANeurotransmitterType::Glutamate,
                BoundedNeurotransmitterKinetics::default(),
            );

            let mut neuron = IzhikevichNeuron {
                c_m: 100.,
                ..IzhikevichNeuron::default_impl()
            };

            neuron.receptors.insert(
                DopaGluGABANeurotransmitterType::Glutamate,
                DopaGluGABAType::Glutamate(GlutamateReceptor::default()),
            ).unwrap();

            let mut spikes = 0;

            for _ in 0..ITERATIONS {
                let _ = spike_train.iterate();
                let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(
                    0., &spike_train.synaptic_neurotransmitters.get_concentrations()
                );

                if is_spiking {
                    spikes += 1;
                }
            }

            spike_counts[n] += spikes;
        }

        assert!(spike_counts.iter().sum::<usize>() > 0);
        assert!(spike_counts[0] <= spike_counts[1]);
        assert!(spike_counts[1] <= spike_counts[2]);
        assert!(spike_counts[0] < spike_counts[2]);
    }
}