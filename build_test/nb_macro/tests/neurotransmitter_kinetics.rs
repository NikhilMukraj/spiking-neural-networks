#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 
    use spiking_neural_networks::neuron::{
        iterate_and_spike::ApproximateNeurotransmitter, 
        iterate_and_spike_traits::{CurrentVoltage, IsSpiking, Timestep}
    };


    #[derive(CurrentVoltage, IsSpiking, Timestep)]
    struct Intermediate {
        current_voltage: f32,
        is_spiking: bool,
        dt: f32,
    }

    neuron_builder!(r#"
    [neurotransmitter_kinetics]
        type: BasicNeurotransmitterKinetics
        vars: t_max = 1, c = 0.001, conc = 0
        on_iteration:
            [if] is_spiking [then]
                conc = t_max
            [else]
                conc = 0
            [end]

            t = t + dt * -c * t + conc

            t = min(max(t, 0), t_max)
    [end]
    "#);

    #[test]
    pub fn test_approximate_neurotransmitter() {
        let mut intermediate = Intermediate { 
            current_voltage: 0., 
            is_spiking: false, 
            dt: 0.1,
        };

        let mut kinetics_to_test = BasicNeurotransmitterKinetics {
            conc: 0.,
            t_max: 1.,
            t: 0.,
            c: 0.001,
        };

        let mut reference_kinetics = ApproximateNeurotransmitter {
            t_max: 1.,
            t: 0.,
            clearance_constant: 0.001,
        };

        for _ in 0..1000 {
            kinetics_to_test.apply_t_change(&intermediate);
            reference_kinetics.apply_t_change(&intermediate);

            assert_eq!(kinetics_to_test.t, reference_kinetics.t);
        }

        intermediate.is_spiking = true;

        for _ in 0..1000 {
            kinetics_to_test.apply_t_change(&intermediate);
            reference_kinetics.apply_t_change(&intermediate);

            assert_eq!(kinetics_to_test.t, reference_kinetics.t);

            intermediate.is_spiking = false;
        }
    }
}