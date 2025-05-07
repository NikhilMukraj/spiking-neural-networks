#[cfg(test)]
mod tests {
    use rand::Rng;
    use std::result::Result;
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, graph::{AdjacencyMatrix, GraphPosition}, neuron::{
            gpu_lattices::{
                LatticeGPU, LatticeNetworkGPU,
            }, integrate_and_fire::{
                QuadraticIntegrateAndFireNeuron, 
                SimpleLeakyIntegrateAndFire,
            }, iterate_and_spike::{
                AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor,
                IonotropicReceptorNeurotransmitterType, LigandGatedChannel
            }, 
            plasticity::STDP, 
            spike_train::{DeltaDiracRefractoriness, PoissonNeuron}, 
            GridVoltageHistory, Lattice, LatticeNetwork, RunLattice, RunNetwork, 
            SpikeTrainGrid, SpikeTrainGridHistory, SpikeTrainLattice
        }
    };

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
        rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
        x != y
    }

    // fn check_entire_history(cpu_grid_history: &[Vec<Vec<f32>>], gpu_grid_history: &[Vec<Vec<f32>>]) {
    //     for (cpu_cell_grid, gpu_cell_grid) in cpu_grid_history.iter()
    //         .zip(gpu_grid_history) {
    //         for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
    //             for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
    //                 let error = (voltage1 - voltage2).abs();
    //                 assert!(
    //                     error <= 2., "error: {}, voltage1: {}, voltage2: {}", 
    //                     error,
    //                     voltage1,
    //                     voltage2,
    //                 );
    //             }
    //         }
    //     }
    // }

    // check if history over time is within 2 mV of each other
    // check if last firing time is within 2 timesteps of one another

    #[test]
    pub fn test_electrical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        let base_neuron = SimpleLeakyIntegrateAndFire {
            gap_conductance: 0.1,
            ..SimpleLeakyIntegrateAndFire::default_impl()
        };
    
        let iterations = 1000;
        let (num_rows, num_cols) = (2, 2);

        let mut lattice = Lattice::default_impl();
        
        lattice.populate(
            &base_neuron, 
            num_rows, 
            num_cols, 
        )?;
    
        lattice.connect(&connection_conditional, None);

        lattice.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });
    
        lattice.update_grid_history = true;
    
        let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
    
        lattice.run_lattice(iterations)?;
    
        gpu_lattice.run_lattice(iterations)?;
    
        for (row1, row2) in lattice.cell_grid().iter().zip(gpu_lattice.cell_grid().iter()) {
            for (neuron1, neuron2) in row1.iter().zip(row2.iter()) {
                let error = (neuron1.current_voltage - neuron2.current_voltage).abs();
                assert!(
                    error <= 2., "error: {}, neuron1: {}, neuron2: {}\n{:#?}\n{:#?}", 
                    error,
                    neuron1.current_voltage,
                    neuron2.current_voltage,
                    lattice.cell_grid().iter()
                        .map(|i| i.iter().map(|j| j.current_voltage).collect::<Vec<f32>>())
                        .collect::<Vec<Vec<f32>>>(),
                    gpu_lattice.cell_grid().iter()
                        .map(|i| i.iter().map(|j| j.current_voltage).collect::<Vec<f32>>())
                        .collect::<Vec<Vec<f32>>>(),
                );
    
                let error = (
                    neuron1.last_firing_time.unwrap_or(0) as isize - 
                    neuron2.last_firing_time.unwrap_or(0) as isize
                ).abs();
                assert!(
                    error <= 2, "error: {:#?}, neuron1: {:#?}, neuron2: {:#?}",
                    error,
                    neuron1.last_firing_time,
                    neuron2.last_firing_time,
                );
            }
        }
    
        for (cpu_cell_grid, gpu_cell_grid) in lattice.grid_history.history.iter()
            .zip(gpu_lattice.grid_history.history.iter()) {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= 2., "error: {}, voltage1: {}, voltage2: {}", 
                        error,
                        voltage1,
                        voltage2,
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_chemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

            base_neuron.ligand_gates
                .insert(IonotropicReceptorNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default())
                .expect("Valid neurotransmitter pairing");
            base_neuron.synaptic_neurotransmitters
                .insert(IonotropicReceptorNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        
            let iterations = 1000;
            let (num_rows, num_cols) = (2, 2);

            let mut lattice = Lattice::default_impl();

            lattice.electrical_synapse = false;
            lattice.chemical_synapse = true;
            
            lattice.populate(
                &base_neuron, 
                num_rows, 
                num_cols, 
            )?;
        
            lattice.connect(&connection_conditional, None);

            lattice.apply(|neuron: &mut _| {
                let mut rng = rand::thread_rng();
                neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
            });
        
            lattice.update_grid_history = true;
        
            let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
        
            lattice.run_lattice(iterations)?;
        
            gpu_lattice.run_lattice(iterations)?;

            for (cpu_cell_grid, gpu_cell_grid) in lattice.grid_history.history.iter()
                .zip(gpu_lattice.grid_history.history.iter()) {
                for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                    for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                        let error = (voltage1 - voltage2).abs();
                        assert!(
                            error <= 5., "error: {}, voltage1: {}, voltage2: {}", 
                            error,
                            voltage1,
                            voltage2,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_electrochemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

            base_neuron.ligand_gates
                .insert(IonotropicReceptorNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default())
                .expect("Valid neurotransmitter pairing");
            base_neuron.synaptic_neurotransmitters
                .insert(IonotropicReceptorNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        
            let iterations = 1000;
            let (num_rows, num_cols) = (2, 2);

            let mut lattice = Lattice::default_impl();

            lattice.electrical_synapse = true;
            lattice.chemical_synapse = true;
            
            lattice.populate(
                &base_neuron, 
                num_rows, 
                num_cols, 
            )?;
        
            lattice.connect(&connection_conditional, None);

            lattice.apply(|neuron: &mut _| {
                let mut rng = rand::thread_rng();
                neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
            });
        
            lattice.update_grid_history = true;
        
            let mut gpu_lattice = LatticeGPU::from_lattice(lattice.clone())?;
        
            lattice.run_lattice(iterations)?;
        
            gpu_lattice.run_lattice(iterations)?;

            for (cpu_cell_grid, gpu_cell_grid) in lattice.grid_history.history.iter()
                .zip(gpu_lattice.grid_history.history.iter()) {
                for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                    for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                        let error = (voltage1 - voltage2).abs();
                        assert!(
                            error <= 5., "error: {}, voltage1: {}, voltage2: {}", 
                            error,
                            voltage1,
                            voltage2,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_isolated_lattices_electrical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        test_isolated_lattices_accuracy(true, false)?;

        Ok(())
    }

    fn test_isolated_lattices_accuracy(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
        base_neuron.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            ApproximateNeurotransmitter::ampa_default()
        );
        base_neuron.ligand_gates.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            LigandGatedChannel::ampa_default(),
        )?;

        let iterations = 1000;

        let mut lattice1 = Lattice::default_impl();
        lattice1.populate(
            &base_neuron, 
            2, 
            2, 
        )?;
        
        lattice1.connect(&connection_conditional, None);
        lattice1.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });
        lattice1.update_grid_history = true;

        let mut lattice2 = Lattice::default_impl();

        lattice2.set_id(1);
        lattice2.populate(
            &base_neuron, 
            3, 
            3, 
        )?;

        lattice2.connect(&connection_conditional, None);
        lattice2.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });

        lattice2.update_grid_history = true;

        let mut network = LatticeNetwork::default_impl();

        network.parallel = true;
        network.add_lattice(lattice1)?;
        network.add_lattice(lattice2)?;

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;
        
        gpu_network.run_lattices(iterations)?;

        for i in network.get_all_ids() {
            let cpu_grid_history = &network.get_lattice(&i).unwrap().grid_history;
            let gpu_grid_history = &gpu_network.get_lattice(&i).unwrap().grid_history;
    
            for (cpu_cell_grid, gpu_cell_grid) in cpu_grid_history.history.iter()
                .zip(gpu_grid_history.history.iter()) {
                for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                    for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                        let error = (voltage1 - voltage2).abs();
                        assert!(
                            error <= 5., "error: {}, voltage1: {}, voltage2: {}", 
                            error,
                            voltage1,
                            voltage2,
                        );
                    }
                }
            }
        }

        Ok(())
    }
    
    fn inter_lattice_connection_conditional(_x: (usize, usize), _y: (usize, usize)) -> bool {
        rand::thread_rng().gen_range(0.0..=1.0) <= 0.8
    }

    #[test]
    pub fn test_connected_lattices_electrical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        test_connected_lattices_accuracy(true, false)?;

        Ok(())
    }

    fn test_connected_lattices_accuracy(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
        base_neuron.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            ApproximateNeurotransmitter::ampa_default()
        );
        base_neuron.ligand_gates.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            LigandGatedChannel::ampa_default(),
        )?;

        let iterations = 1000;

        let mut lattice1 = Lattice::default_impl();
        lattice1.populate(
            &base_neuron, 
            2, 
            2, 
        )?;

        lattice1.connect(&connection_conditional, None);
        lattice1.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });
        lattice1.update_grid_history = true;

        let mut lattice2 = Lattice::default_impl();

        lattice2.set_id(1);
        lattice2.populate(
            &base_neuron, 
            3, 
            3, 
        )?;

        lattice2.connect(&connection_conditional, None);
        lattice2.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });

        lattice2.update_grid_history = true;

        let mut network = LatticeNetwork::default_impl();

        network.parallel = true;
        network.add_lattice(lattice1)?;
        network.add_lattice(lattice2)?;

        network.connect(0, 1, &inter_lattice_connection_conditional, None)?;

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        gpu_network.run_lattices(iterations)?;

        for i in network.get_all_ids() {
            let cpu_grid_history = &network.get_lattice(&i).unwrap().grid_history;
            let gpu_grid_history = &gpu_network.get_lattice(&i).unwrap().grid_history;
    
            for (cpu_cell_grid, gpu_cell_grid) in cpu_grid_history.history.iter()
                .zip(gpu_grid_history.history.iter()) {
                for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                    for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                        let error = (voltage1 - voltage2).abs();
                        assert!(
                            error <= 5., "error: {}, voltage1: {}, voltage2: {}", 
                            error,
                            voltage1,
                            voltage2,
                        );
                    }
                }
            }
        }
        
        Ok(())
    }
    
    #[test]
    pub fn test_spike_train_lattice_firing_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_firing(true, false)?;

        Ok(())
    }

    #[test]
    pub fn test_isolated_spike_train_lattice_firing_with_neurons_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_isolated_spike_train_lattice_firing_with_neurons(true, false)?;

        Ok(())
    }

    fn test_isolated_spike_train_lattice_firing_with_neurons(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
        base_neuron.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            ApproximateNeurotransmitter::ampa_default()
        );
        base_neuron.ligand_gates.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            LigandGatedChannel::ampa_default(),
        )?;
    
        let iterations = 1000;

        let mut lattice1 = Lattice::default_impl();
        
        lattice1.populate(
            &base_neuron, 
            3, 
            3, 
        )?;
    
        lattice1.connect(&connection_conditional, None);
        lattice1.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });
        lattice1.update_grid_history = true;

        lattice1.set_id(1);

        let mut base_spike_train = PoissonNeuron::default_impl();
        base_spike_train.chance_of_firing = 0.1;

        let mut spike_train_lattice = SpikeTrainLattice::default_impl();
        spike_train_lattice.populate(&base_spike_train, 3, 3)?;
        spike_train_lattice.update_grid_history = true;

        let lattices = vec![lattice1];
        let spike_train_lattices = vec![spike_train_lattice];

        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        gpu_network.run_lattices(iterations)?;

        network.run_lattices(iterations)?;

        let cpu_grid_history = &network.get_lattice(&1).unwrap().grid_history;
        let gpu_grid_history = &gpu_network.get_lattice(&1).unwrap().grid_history;
        
        for (cpu_cell_grid, gpu_cell_grid) in cpu_grid_history.history.iter()
            .zip(gpu_grid_history.history.iter()) {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= 5., "error: {}, voltage1: {}, voltage2: {}", 
                        error,
                        voltage1,
                        voltage2,
                    );
                }
            }
        }   

        let history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert!(!history.is_empty());

        let mut spiking_occured = false;
        for grid in history.iter() {
            for row in grid {
                for i in row.iter() {
                    assert!((i - base_spike_train.v_resting).abs() < 2. 
                        || (i - base_spike_train.v_th).abs() < 2.);
                    if (i - base_spike_train.v_th.abs()) < 2. {
                        spiking_occured = true;
                    }
                }
            }
        }

        assert!(spiking_occured);

        Ok(())
    }

    #[test]
    pub fn test_spike_train_lattice_firing_with_neurons_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_firing_with_neurons(true, false)?;
        
        Ok(())
    }

    #[test]
    pub fn test_isolated_lattices_chemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_isolated_lattices_accuracy(false, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_connected_lattices_chemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_connected_lattices_accuracy(false, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_lattice_firing_chemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_firing(false, true)?;

        Ok(())
    }

    fn test_spike_train_firing(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let mut base_spike_train = PoissonNeuron::default_impl();
        base_spike_train.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            ApproximateNeurotransmitter::ampa_default()
        );
        base_spike_train.chance_of_firing = 0.1;

        let mut spike_train_lattice = SpikeTrainLattice::default_impl();
        spike_train_lattice.populate(&base_spike_train, 3, 3)?;
        spike_train_lattice.update_grid_history = true;

        #[allow(clippy::type_complexity)]
        let mut network: LatticeNetwork<
            QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, 
            AdjacencyMatrix<(usize, usize), f32>, 
            GridVoltageHistory, 
            PoissonNeuron<IonotropicReceptorNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, 
            SpikeTrainGridHistory, 
            AdjacencyMatrix<GraphPosition, f32>, 
            STDP, 
            IonotropicReceptorNeurotransmitterType,
        > = LatticeNetwork::default_impl();
        
        network.add_spike_train_lattice(spike_train_lattice)?;

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network)?;

        gpu_network.run_lattices(1000)?;

        let history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert!(!history.is_empty());

        let mut spiking_occured = false;
        for grid in history.iter() {
            for row in grid {
                for i in row.iter() {
                    assert!((i - base_spike_train.v_resting).abs() < 2. 
                        || (i - base_spike_train.v_th).abs() < 2.);
                    if (i - base_spike_train.v_th.abs()) < 2. {
                        spiking_occured = true;
                    }
                }
            }
        }

        assert!(spiking_occured);

        Ok(())
    }

    #[test]
    pub fn test_isolated_spike_train_lattice_firing_with_neurons_chemical() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_isolated_spike_train_lattice_firing_with_neurons(false, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_lattice_firing_with_neurons_chemical() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_spike_train_firing_with_neurons(false, true)?;
        }
            
        Ok(())
    }

    fn test_spike_train_firing_with_neurons(electrical_synapse: bool, chemical_synapse: bool) -> Result<(), SpikingNeuralNetworksError> {
        let neurotransmitter = ApproximateNeurotransmitter {
            t_max: 1.,
            t: 0.,
            clearance_constant: 0.001,
        };

        let mut base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 10.,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
        base_neuron.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            neurotransmitter
        );
        base_neuron.ligand_gates.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            LigandGatedChannel::ampa_default(),
        )?;
    
        let iterations = 1000;

        let lattice_size = 3;

        let mut lattice1 = Lattice::default_impl();
        
        lattice1.populate(
            &base_neuron, 
            3, 
            3, 
        )?;
    
        // lattice1.connect(&connection_conditional, None);
        lattice1.apply(|neuron: &mut _| {
            let mut rng = rand::thread_rng();
            neuron.current_voltage = rng.gen_range(neuron.v_init..=neuron.v_th);
        });
        lattice1.update_grid_history = true;

        lattice1.set_id(1);

        let mut base_spike_train = PoissonNeuron::default_impl();
        base_spike_train.synaptic_neurotransmitters.insert(
            IonotropicReceptorNeurotransmitterType::AMPA, 
            neurotransmitter
        );

        let mut spike_train_lattice = SpikeTrainLattice::default_impl();
        spike_train_lattice.populate(&base_spike_train, lattice_size, lattice_size)?;
        spike_train_lattice.apply_given_position(|pos, i| {
            if (pos.0 * lattice_size + pos.1) % 2 == 0 {
                i.chance_of_firing = 0.005;
            } else {
                i.chance_of_firing = 0.;
            }
        });
        spike_train_lattice.update_grid_history = true;

        let lattices = vec![lattice1];
        let spike_train_lattices = vec![spike_train_lattice];

        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;

        network.connect(0, 1, &|x, y| {(x.0 * lattice_size + x.1) % 2 == 0 && x == y}, None)?;
        network.set_dt(1.);

        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network)?;

        gpu_network.run_lattices(iterations)?;

        // check which neurons are firing and which are not
        // check that the only spike trains that are firing are correct

        // check that spike trains in right row/col are firing, same with neurons

        let history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        let mut has_fired = false;

        for grid in history {
            for row in grid {
                for j in row {
                    if (j - base_spike_train.v_th).abs() < 2. {
                        has_fired = true;
                    }
                }
            }
        }

        assert!(has_fired);

        for i in 0..lattice_size {
            for j in 0..lattice_size {
                let mut spiking_count = 0;

                #[allow(clippy::needless_range_loop)]
                for n in 0..iterations {
                    if (history[n][i][j] - base_spike_train.v_th).abs() < 2. {
                        spiking_count += 1;
                    }
                }

                if (i * lattice_size + j) % 2 == 0 {
                    assert!(
                        gpu_network.get_spike_train_lattice(&0)
                            .unwrap()
                            .spike_train_grid()[i][j].chance_of_firing != 0.
                    );
                    assert!(spiking_count > 3, "({}, {}) | spiking count: {}", i, j, spiking_count);
                } else {
                    assert!(
                        gpu_network.get_spike_train_lattice(&0)
                            .unwrap()
                            .spike_train_grid()[i][j].chance_of_firing == 0.
                    );
                    assert!(spiking_count <= 3, "({}, {}) | spiking count: {}", i, j, spiking_count);
                }
            }
        }

        let history = &gpu_network.get_lattice(&1).unwrap().grid_history.history;

        for i in 0..lattice_size {
            for j in 0..lattice_size {
                let mut spiking_count = 0;

                #[allow(clippy::needless_range_loop)]
                for n in 0..iterations {
                    if (history[n][i][j] - base_neuron.v_th).abs() < 2. {
                        spiking_count += 1;
                    }
                }

                if (i * lattice_size + j) % 2 == 0 {
                    assert!(spiking_count > 3, "({}, {}) | spiking count: {}", i, j, spiking_count);
                } else {
                    assert!(spiking_count <= 3, "({}, {}) | spiking count: {}", i, j, spiking_count);
                }
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_isolated_lattices_electrochemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_isolated_lattices_accuracy(true, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_connected_lattices_electrochemical_accuracy() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_connected_lattices_accuracy(true, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_lattice_firing_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_firing(true, true)?;

        Ok(())
    }

    #[test]
    pub fn test_isolated_spike_train_lattice_firing_with_neurons_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_isolated_spike_train_lattice_firing_with_neurons(true, true)?;
        }

        Ok(())
    }

    #[test]
    pub fn test_spike_train_lattice_firing_with_neurons_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        for _ in 0..3 {
            test_spike_train_firing_with_neurons(true, true)?;
        }
            
        Ok(())
    }
}
