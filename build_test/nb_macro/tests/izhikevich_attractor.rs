mod izhikevich_dopamine; 

#[cfg(test)]
mod test {
    use crate::izhikevich_dopamine::{
        DopaGluGABANeurotransmitterType, IzhikevichNeuron
    };
    use rand::Rng;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::AdjacencyMatrix, 
        neuron::{
            attractors::{
                distort_pattern, generate_binary_hopfield_network, 
                generate_hopfield_network, generate_random_patterns
            }, 
            plasticity::STDP, 
            Lattice, LatticeNetwork, RunLattice, RunNetwork, SpikeHistory,
        }
    };


    const ITERATIONS: usize = 1000;

    #[test]
    fn test_autoassociative_bipolar() -> Result<(), SpikingNeuralNetworksError> {
        let mut accuracies = vec![];
        let trials = 3;

        for _ in 0..trials {
            let (num_rows, num_cols) = (7, 7);
            let base_neuron = IzhikevichNeuron {
                gap_conductance: 10.,
                c_m: 100.,
                ..IzhikevichNeuron::default_impl()
            };
        
            let mut lattice: Lattice<_, _, SpikeHistory, STDP, DopaGluGABANeurotransmitterType> = 
                Lattice::default();
            lattice.parallel = true;
            lattice.update_grid_history = true;
            lattice.populate(&base_neuron, num_rows, num_cols).unwrap();
            lattice.set_dt(1.);
        
            let random_patterns = generate_random_patterns(num_rows, num_cols, 1, 0.5);
            let bipolar_connections = generate_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(
                0,
                &random_patterns,
            )?;
            lattice.set_graph(bipolar_connections)?;
        
            let pattern_index = 0;
            let input_pattern = distort_pattern(&random_patterns[pattern_index], 0.1);
            lattice.apply_given_position(|pos, neuron| {
                if input_pattern[pos.0][pos.1] {
                    neuron.current_voltage = neuron.v_th;
                } else {
                    neuron.current_voltage = neuron.c;
                }
            });
        
            lattice.run_lattice(ITERATIONS)?;

            let firing_rates = lattice.grid_history.aggregate();
            let firing_threshold: isize = 5;
            let predicted_pattern: Vec<Vec<bool>> = firing_rates.iter()
                .map(|row| {
                    row.iter().map(|i| *i >= firing_threshold).collect::<Vec<bool>>()
                })
                .collect();
            
            let mut accuracy = 0.;
            for (row1, row2) in predicted_pattern.iter().zip(random_patterns[pattern_index].iter()) {
                for (item1, item2) in row1.iter().zip(row2.iter()) {
                    if item1 == item2 {
                        accuracy += 1.;
                    }
                }
            }

            accuracies.push(accuracy / ((num_rows * num_cols) as f32))
        }

        assert!(accuracies.iter().filter(|i| **i > 0.9).collect::<Vec<_>>().len() >= trials / 2);
    
        Ok(())
    }

    #[test]
    fn test_electrical_autoassociative_binary() -> Result<(), SpikingNeuralNetworksError> {
        let mut accuracies = vec![];
        let trials = 3;

        for _ in 0..trials {
            let (num_rows, num_cols) = (5, 5);
            let (inh_num_rows, inh_num_cols) = (3, 3);
            let base_neuron = IzhikevichNeuron {
                gap_conductance: 10.,
                c_m: 100.,
                ..IzhikevichNeuron::default_impl()
            };
        
            let mut inh: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
            inh.populate(&base_neuron, inh_num_rows, inh_num_cols).unwrap();
            inh.apply(|neuron| 
                neuron.current_voltage = rand::thread_rng().gen_range(neuron.c..=neuron.v_th)
            );
            inh.connect(&(|x, y| x != y), Some(&(|_, _| -1.5)));
        
            let mut exc: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
            exc.update_grid_history = true;
            exc.populate(&base_neuron, num_rows, num_cols).unwrap();
        
            let random_patterns = generate_random_patterns(num_rows, num_cols, 1, 0.5);
            let binary_connections = generate_binary_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(
                1,
                &random_patterns,
                1.,
                1.,
                0.5,
            )?;
            exc.set_graph(binary_connections)?;
            exc.set_id(1);
        
            let pattern_index = 0;
            let input_pattern = distort_pattern(&random_patterns[pattern_index], 0.1);
            exc.apply_given_position(|pos, neuron| {
                if input_pattern[pos.0][pos.1] {
                    neuron.current_voltage = neuron.v_th;
                } else {
                    neuron.current_voltage = neuron.c;
                }
            });
        
            let mut network = LatticeNetwork::default_impl();
            network.add_lattice(exc)?;
            network.add_lattice(inh)?;
            network.parallel = true;
            network.connect(
                0, 1, &(|_, _| true), Some(&(|_, _| -2.))
            ).unwrap();
            network.connect(
                1, 0, &(|_, _| true), Some(&(|_, _| 1.))
            ).unwrap();
        
            network.set_dt(1.);
            network.run_lattices(ITERATIONS)?;
        
            let firing_rates = network.get_lattice(&1).expect("Could not retrieve lattice")
                .grid_history.aggregate();
            let firing_threshold: isize = 10;
            let predicted_pattern: Vec<Vec<bool>> = firing_rates.iter()
                .map(|row| {
                    row.iter().map(|i| *i >= firing_threshold).collect::<Vec<bool>>()
                })
                .collect();
            
            let mut accuracy = 0.;
            for (row1, row2) in predicted_pattern.iter().zip(random_patterns[pattern_index].iter()) {
                for (item1, item2) in row1.iter().zip(row2.iter()) {
                    if item1 == item2 {
                        accuracy += 1.;
                    }
                }
            }

            accuracies.push(accuracy / ((num_rows * num_cols) as f32));
        }
        
        assert!(accuracies.iter().sum::<f32>() / trials as f32 >= 0.85);
 
        Ok(())
    }

    // #[test]
    // fn test_chemical_autoassocative_binary() -> Result<(), SpikingNeuralNetworksError> {
    //     let mut accuracies = vec![];
    //     let trials = 3;

    //     for _ in 0..trials {
    //         let (num_rows, num_cols) = (5, 5);
    //         let (inh_num_rows, inh_num_cols) = (3, 3);
    //         let mut base_neuron = IzhikevichNeuron {
    //             c_m: 25.,
    //             ..IzhikevichNeuron::default_impl()
    //         };

    //         base_neuron.receptors.insert(
    //             DopaGluGABANeurotransmitterType::Glutamate,
    //             DopaGluGABAType::Glutamate(GlutamateReceptor::default()),
    //         ).unwrap();
    //         base_neuron.receptors.insert(
    //             DopaGluGABANeurotransmitterType::GABA,
    //             DopaGluGABAType::GABA(GABAReceptor::default()),
    //         ).unwrap();

    //         let mut inh_neuron = base_neuron.clone();

    //         base_neuron.synaptic_neurotransmitters.insert(
    //             DopaGluGABANeurotransmitterType::Glutamate,
    //             BoundedNeurotransmitterKinetics::default(),
    //         );

    //         inh_neuron.synaptic_neurotransmitters.insert(
    //             DopaGluGABANeurotransmitterType::Glutamate,
    //             BoundedNeurotransmitterKinetics::default(),
    //         );

    //         let mut poisson: PoissonNeuron<DopaGluGABANeurotransmitterType, BoundedNeurotransmitterKinetics, DeltaDiracRefractoriness> = PoissonNeuron::default();

    //         poisson.synaptic_neurotransmitters.insert(
    //             DopaGluGABANeurotransmitterType::Glutamate,
    //             BoundedNeurotransmitterKinetics::default(),
    //         );

    //         let mut cue: SpikeTrainLattice<_, _, SpikeTrainGridHistory> = SpikeTrainLattice::default();
    //         cue.populate(&poisson, num_rows, num_cols).unwrap();
            
    //         let random_patterns = generate_random_patterns(num_rows, num_cols, 3, 0.5);
    //         let binary_connections = generate_binary_hopfield_network::<AdjacencyMatrix<(usize, usize), f32>>(
    //             1,
    //             &random_patterns,
    //             -1.,
    //             0.,
    //             0.25,
    //         )?;

    //         let pattern_index = 0;
    //         let input_pattern = distort_pattern(&random_patterns[pattern_index], 0.1);
    //         cue.apply_given_position(|pos: (usize, usize), neuron: &mut PoissonNeuron<_, _, _>| {
    //             if input_pattern[pos.0][pos.1] {
    //                 neuron.chance_of_firing = 0.001;
    //             } else {
    //                 neuron.chance_of_firing = 0.;
    //             }
    //         });

    //         cue.set_id(0);
        
    //         let mut exc: Lattice<_, _, SpikeHistory, STDP, _> = Lattice::default();
    //         exc.update_grid_history = true;
    //         exc.populate(&base_neuron, num_rows, num_cols).unwrap();
    //         exc.apply(|neuron: &mut _| {
    //             neuron.current_voltage = rand::thread_rng().gen_range(neuron.c..neuron.v_th);
    //         });
        
    //         exc.set_graph(binary_connections)?;
    //         exc.set_id(1);

    //         let mut inh: Lattice<_, _, _, _, _> = Lattice::default();
    //         inh.populate(&inh_neuron, inh_num_rows, inh_num_cols)?;
    //         inh.apply(|neuron: &mut _| {
    //             neuron.current_voltage = rand::thread_rng().gen_range(neuron.c..neuron.v_th);
    //         });  
    //         inh.connect(&(|_, _| rand::thread_rng().gen_range((0.)..1.) < 0.4), Some(&(|_, _| 0.05)));
    //         inh.set_id(2);

    //         let lattices = vec![exc, inh];
    //         let spike_train_lattices = vec![cue];
        
    //         let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices).unwrap();
    //         network.parallel = true;
    //         network.connect(
    //             0, 1, &(|_, _| true), Some(&(|_, _| 5.))
    //         ).unwrap();
    //         network.connect(
    //             2, 1, &(|_, _| rand::thread_rng().gen_range((0.)..1.) < 0.25), Some(&(|_, _| 0.15))
    //         )?;
    //         network.connect(
    //             1, 2, &(|_, _| rand::thread_rng().gen_range((0.)..1.) < 0.25), Some(&(|_, _| 0.15))
    //         )?;

    //         network.electrical_synapse = false;
    //         network.chemical_synapse = true;
        
    //         network.set_dt(1.);
    //         network.run_lattices(ITERATIONS)?;
        
    //         let firing_rates = network.get_lattice(&1).expect("Could not retrieve lattice")
    //             .grid_history.aggregate();

    //         let mut current_accuracies = vec![];
    //         let firing_min = *firing_rates.iter().flatten()
    //             .min_by(|a, b| a.partial_cmp(b).unwrap())
    //             .unwrap();
    //         let firing_max = *firing_rates.iter().flatten()
    //             .max_by(|a, b| a.partial_cmp(b).unwrap())
    //             .unwrap() + 1;
    //         for firing_threshold in firing_min..firing_max {
    //             let predicted_pattern: Vec<Vec<bool>> = firing_rates.iter()
    //                 .map(|row| {
    //                     row.iter().map(|i| *i >= firing_threshold).collect::<Vec<bool>>()
    //                 })
    //                 .collect();
                            
    //             let mut accuracy = 0.;
    //             for (row1, row2) in predicted_pattern.iter().zip(random_patterns[pattern_index].iter()) {
    //                 for (item1, item2) in row1.iter().zip(row2.iter()) {
    //                     if item1 == item2 {
    //                         accuracy += 1.;
    //                     }
    //                 }
    //             }

    //             current_accuracies.push(accuracy);
    //         }

    //         accuracies.push(
    //             current_accuracies.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() / 
    //             ((num_rows * num_cols) as f32)
    //         );
    //     }

    //     println!("{:#?}", accuracies);        
    //     assert!(accuracies.iter().sum::<f32>() / trials as f32 >= 0.75);
 
    //     Ok(())
    // }
}