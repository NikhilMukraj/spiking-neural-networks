#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{
        command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, 
        device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}, 
        program::Program
    };
    use spiking_neural_networks::error::SpikingNeuralNetworksError;


    neuron_builder!("
    [receptor_kinetics]
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]

    [receptor_kinetics]
        type: ModulatedReceptorKinetics
        vars: m = 1
        on_iteration:
            r = m * t
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

    #[test]
    fn test_get_attributes() {
        let attrs = ExampleReceptors::<BoundedReceptorKinetics>::get_all_attributes();

        assert!(attrs.iter().all(|(i, _)| i.contains("$")));
        assert!(attrs.iter().all(|(i, _)| i.contains("receptors")));

        assert!(!attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::OptionalUInt)));
        assert!(attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Basic_g"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Basic_e"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Combined_g1"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined_g2"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Combined_g1"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$Combined_g2"), AvailableBufferType::UInt)));
        assert!(attrs.contains(&(String::from("receptors$Combined_e1"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined_e2"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::UInt)));

        assert!(attrs.contains(&(String::from("receptors$Combined$r1$kinetics$r"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r2$kinetics$r"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r1$kinetics$r_max"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r2$kinetics$r_max"), AvailableBufferType::Float)));

        let attrs = ExampleReceptors::<ModulatedReceptorKinetics>::get_all_attributes();
        
        assert!(attrs.iter().all(|(i, _)| i.contains("$")));
        assert!(attrs.iter().all(|(i, _)| i.contains("receptors")));

        assert!(!attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::OptionalUInt)));
        assert!(attrs.contains(&(String::from("receptors$top_m"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Basic_g"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Basic_e"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Combined_g1"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined_g2"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Combined_g1"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$Combined_g2"), AvailableBufferType::UInt)));
        assert!(attrs.contains(&(String::from("receptors$Combined_e1"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined_e2"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::UInt)));
        assert!(!attrs.contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::UInt)));
        assert!(attrs.contains(&(String::from("receptors$Basic$r$kinetics$m"), AvailableBufferType::Float)));

        assert!(attrs.contains(&(String::from("receptors$Combined$r1$kinetics$r"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r2$kinetics$r"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Combined$r1$kinetics$r_max"), AvailableBufferType::Float)));
        assert!(!attrs.contains(&(String::from("receptors$Combined$r2$kinetics$r_max"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r1$kinetics$m"), AvailableBufferType::Float)));
        assert!(attrs.contains(&(String::from("receptors$Combined$r2$kinetics$m"), AvailableBufferType::Float)));
    }

    #[test]
    fn test_neurotransmitter_gpu_type() {
        assert_eq!(ExampleReceptorsNeurotransmitterType::number_of_types(), 2);
        assert_eq!(
            ExampleReceptorsNeurotransmitterType::get_all_types(), 
            BTreeSet::from([
                ExampleReceptorsNeurotransmitterType::Basic, 
                ExampleReceptorsNeurotransmitterType::Combined
            ])
        );
    }

    #[test]
    fn test_get_attrs_subdivisions() {
                assert_eq!(
            ExampleReceptors::<ModulatedReceptorKinetics>::get_all_top_level_attributes(), 
            HashSet::from([(String::from("receptors$top_m"), AvailableBufferType::Float)])
        );
        assert_eq!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_all_top_level_attributes(), 
            HashSet::from([(String::from("receptors$top_m"), AvailableBufferType::Float)])
        );

        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Basic)
                .iter().all(|i| !i.0.starts_with("receptors$top_"))
        );

        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Basic)
                .contains(&(String::from("receptors$Basic_g"), AvailableBufferType::Float))
        );
        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Basic)
                .contains(&(String::from("receptors$Basic_e"), AvailableBufferType::Float))
        );
        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Basic)
                .contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::Float))
        );
        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Basic)
                .contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::Float))
        );

        assert!(
            !ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Basic_g"), AvailableBufferType::Float))
        );
        assert!(
            !ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Basic_e"), AvailableBufferType::Float))
        );

        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Combined_g1"), AvailableBufferType::Float))
        );
        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Combined_g2"), AvailableBufferType::Float))
        );

        assert!(
            !ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Basic$r$kinetics$r"), AvailableBufferType::Float))
        );
        assert!(
            !ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Basic$r$kinetics$r_max"), AvailableBufferType::Float))
        );

        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Combined$r1$kinetics$r"), AvailableBufferType::Float))
        );
        assert!(
            ExampleReceptors::<BoundedReceptorKinetics>::get_attributes_associated_with(&ExampleReceptorsNeurotransmitterType::Combined)
                .contains(&(String::from("receptors$Combined$r1$kinetics$r_max"), AvailableBufferType::Float))
        );
    }

    #[test]
    fn test_conversion() {
        let mut receptors = ExampleReceptors::<BoundedReceptorKinetics>::default();

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Basic, 
            ExampleReceptorsType::Basic(BasicReceptor {g: 0.1, ..BasicReceptor::default()})
        ).unwrap();
        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Combined, 
            ExampleReceptorsType::Combined(CombinedReceptor {g1: 0.25, ..CombinedReceptor::default()})
        ).unwrap();

        receptors.m = 10.;

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_grid = ExampleReceptors::convert_to_gpu(&[vec![receptors.clone()]], &context, &queue).unwrap();

        assert!(gpu_grid.contains_key("receptors$flags"));

        for (attr, _) in ExampleReceptors::<BoundedReceptorKinetics>::get_all_attributes() {
            assert!(gpu_grid.contains_key(&attr))
        }

        let mut conversion: Vec<Vec<ExampleReceptors<BoundedReceptorKinetics>>> = vec![vec![ExampleReceptors::default()]];

        assert!(conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Basic).is_none());
        assert!(conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Combined).is_none());

        ExampleReceptors::convert_to_cpu(&mut conversion, &gpu_grid, &queue, 1, 1).unwrap();

        assert_eq!(conversion[0][0].m, receptors.m);

        assert!(conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Basic).is_some());
        assert!(conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Combined).is_some());
        assert_eq!(
            conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Basic).unwrap(),
            receptors.get(&ExampleReceptorsNeurotransmitterType::Basic).unwrap(),
        );
        assert_eq!(
            conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Combined).unwrap(),
            receptors.get(&ExampleReceptorsNeurotransmitterType::Combined).unwrap(),
        );
    }

    #[test]
    fn test_conversion_non_square() {
        let mut receptors = ExampleReceptors::<BoundedReceptorKinetics>::default();

        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Basic, 
            ExampleReceptorsType::Basic(
                BasicReceptor {
                    g: 0.1, 
                    r: BoundedReceptorKinetics { r_max: 0.5, r: 0. }, 
                    ..BasicReceptor::default()
                }
            )
        ).unwrap();
        receptors.insert(
            ExampleReceptorsNeurotransmitterType::Combined, 
            ExampleReceptorsType::Combined(CombinedReceptor {g2: 0.25, ..CombinedReceptor::default()})
        ).unwrap();

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_grid = ExampleReceptors::convert_to_gpu(
            &[vec![receptors.clone(), ExampleReceptors::default()]], 
            &context, 
            &queue
        ).unwrap();    

        let mut conversion: Vec<Vec<ExampleReceptors<BoundedReceptorKinetics>>> = vec![vec![ExampleReceptors::default(), ExampleReceptors::default()]];

        ExampleReceptors::convert_to_cpu(&mut conversion, &gpu_grid, &queue, 1, 2).unwrap();

        assert_eq!(
            conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Basic).unwrap(),
            receptors.get(&ExampleReceptorsNeurotransmitterType::Basic).unwrap(),
        );
        assert_eq!(
            conversion[0][0].get(&ExampleReceptorsNeurotransmitterType::Combined).unwrap(),
            receptors.get(&ExampleReceptorsNeurotransmitterType::Combined).unwrap(),
        );

        assert!(conversion[0][1].get(&ExampleReceptorsNeurotransmitterType::Basic).is_none());
        assert!(conversion[0][1].get(&ExampleReceptorsNeurotransmitterType::Combined).is_none());
    }

    #[test]
    fn test_get_updates() -> Result<(), SpikingNeuralNetworksError> {
        let updates = ExampleReceptors::<BoundedReceptorKinetics>::get_updates();

        assert_eq!(updates.len(), ExampleReceptorsNeurotransmitterType::number_of_types());

        for (_, attrs) in updates.iter() {
            for i in attrs {
                if i.0 == "index" {
                    continue;
                }
                if i.0 == "current_voltage" {
                    continue;
                }
                assert!(
                    ExampleReceptors::<BoundedReceptorKinetics>::get_all_attributes().contains(i), 
                    "{:#?}", 
                    i,
                );
            }
        }

        let mut program = String::from("");
        for (function, _) in updates.iter() {
            if !program.is_empty() {
                program = format!("{}\n{}", program, function);
            } else {
                program = function.clone();
            }
        }

        println!("{}", program);

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        match Program::create_and_build_from_source(&context, &program, "") {
            Ok(_) => Ok(()),
            Err(_) => Err(SpikingNeuralNetworksError::from(GPUError::ProgramCompileFailure)),
        }
    }
}
