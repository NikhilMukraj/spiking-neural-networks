#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    // check if getting and setting attributes (particularly with kinetics)
    // works as intended

    use nb_macro::neuron_builder;


    neuron_builder!("
    [receptor_kinetics]
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]

    [receptors]
        type: ExampleReceptors
        vars: m = 1
        neurotransmitter: Basic
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = m * g * r * (v - e)
        neurotransmitter: Combined
        receptors: r1, r2
        vars: current = 0, g1 = 2, e1 = 0, g2 = 3, e2 = 0
        on_iteration:
            current = m * g1 * r1 * (v - e1) + m * g2 * r2 * (v - e2)
    [end]
    ");

    #[test]
    fn test_get() {
        let mut receptors = ExampleReceptors::<BoundedReceptorKinetics>::default();

        assert_eq!(receptors.get_attribute("nonsense"), None);

        assert_eq!(receptors.get_attribute("receptors$top_m"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Basic_g"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined_g1"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined_g2"), None);
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r"), None);
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r_max"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r_max"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r"), None);
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r_max"), None);

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Basic, 
            ExampleReceptorsType::Basic(BasicReceptor::default())
        ).unwrap();

        assert_eq!(receptors.get_attribute("receptors$Basic_g"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Basic_e"), Some(BufferType::Float(0.)));

        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r"), Some(BufferType::Float(0.)));
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r_max"), Some(BufferType::Float(1.)));

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Combined, 
            ExampleReceptorsType::Combined(CombinedReceptor::default())
        ).unwrap();

        assert_eq!(receptors.get_attribute("receptors$Combined_g1"), Some(BufferType::Float(2.)));
        assert_eq!(receptors.get_attribute("receptors$Combined_g2"), Some(BufferType::Float(3.)));

        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r"), Some(BufferType::Float(0.)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r_max"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r"), Some(BufferType::Float(0.)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r_max"), Some(BufferType::Float(1.)));

        assert_eq!(receptors.get_attribute("qwerty"), None);
    }

    #[test]
    fn test_set() {
        let mut receptors = ExampleReceptors::<BoundedReceptorKinetics>::default();

        assert!(receptors.set_attribute("nonsense", BufferType::Float(1.)).is_err());

        assert_eq!(receptors.get_attribute("receptors$top_m"), Some(BufferType::Float(1.)));
        assert!(receptors.set_attribute("receptors$top_m", BufferType::UInt(2)).is_err());
        assert!(receptors.set_attribute("receptors$top_m", BufferType::Float(2.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$top_m"), Some(BufferType::Float(2.)));
        assert_eq!(receptors.get_attribute("receptors$Basic_g"), None);
        assert!(receptors.set_attribute("receptors$Basic_g", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined_g1"), None);
        assert!(receptors.set_attribute("receptors$Combined_g1", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined_g2"), None);
        assert!(receptors.set_attribute("receptors$Combined_g2", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r"), None);
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r_max"), None);
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r_max", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r"), None);
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r_max"), None);
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r_max", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r"), None);
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r", BufferType::Float(1.)).is_err());
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r_max"), None);
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r_max", BufferType::Float(1.)).is_err());

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Basic, 
            ExampleReceptorsType::Basic(BasicReceptor::default())
        ).unwrap();

        assert_eq!(receptors.get_attribute("receptors$Basic_g"), Some(BufferType::Float(1.)));
        assert!(receptors.set_attribute("receptors$Basic_g", BufferType::UInt(2)).is_err());
        assert!(receptors.set_attribute("receptors$Basic_g", BufferType::Float(2.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Basic_g"), Some(BufferType::Float(2.)));

        assert_eq!(receptors.get_attribute("receptors$Basic_e"), Some(BufferType::Float(0.)));
        assert!(receptors.set_attribute("receptors$Basic_e", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Basic_e", BufferType::Float(-2.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Basic_e"), Some(BufferType::Float(-2.)));

        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r"), Some(BufferType::Float(0.)));
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r", BufferType::Float(1.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r_max"), Some(BufferType::Float(1.)));
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r_max", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Basic$r$kinetics$r_max", BufferType::Float(0.5)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Basic$r$kinetics$r_max"), Some(BufferType::Float(0.5)));

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Combined, 
            ExampleReceptorsType::Combined(CombinedReceptor::default())
        ).unwrap();

        assert_eq!(receptors.get_attribute("receptors$Combined_g1"), Some(BufferType::Float(2.)));
        assert!(receptors.set_attribute("receptors$Combined_g1", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Combined_g1", BufferType::Float(1.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined_g1"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Combined_g2"), Some(BufferType::Float(3.)));
        assert!(receptors.set_attribute("receptors$Combined_g2", BufferType::UInt(4)).is_err());
        assert!(receptors.set_attribute("receptors$Combined_g2", BufferType::Float(1.5)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined_g2"), Some(BufferType::Float(1.5)));

        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r"), Some(BufferType::Float(0.)));
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r", BufferType::Float(1.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r_max"), Some(BufferType::Float(1.)));
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r_max", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Combined$r1$kinetics$r_max", BufferType::Float(0.5)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined$r1$kinetics$r_max"), Some(BufferType::Float(0.5)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r"), Some(BufferType::Float(0.)));
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r", BufferType::Float(1.)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r"), Some(BufferType::Float(1.)));
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r_max"), Some(BufferType::Float(1.)));
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r_max", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("receptors$Combined$r2$kinetics$r_max", BufferType::Float(0.5)).is_ok());
        assert_eq!(receptors.get_attribute("receptors$Combined$r2$kinetics$r_max"), Some(BufferType::Float(0.5)));

        assert!(receptors.set_attribute("qwerty", BufferType::Float(1.)).is_err());
        assert!(receptors.set_attribute("qwerty", BufferType::UInt(1)).is_err());
        assert!(receptors.set_attribute("qwerty", BufferType::OptionalUInt(1)).is_err());
    }
}
