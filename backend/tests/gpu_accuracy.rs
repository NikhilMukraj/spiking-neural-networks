#[cfg(test)]
mod tests {
    use rand::Rng;
    use std::result::Result;
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError,
        neuron::{
            gpu_lattices::{
                LatticeGPU, LatticeNetworkGPU,
            }, 
            integrate_and_fire::{
                QuadraticIntegrateAndFireNeuron, 
                SimpleLeakyIntegrateAndFire,
            }, iterate_and_spike::{
                AMPADefault, ApproximateNeurotransmitter, 
                IonotropicNeurotransmitterType, LigandGatedChannel
            }, Lattice, LatticeNetwork, RunLattice, RunNetwork,
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
        );
    
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
                .insert(IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default())
                .expect("Valid neurotransmitter pairing");
            base_neuron.synaptic_neurotransmitters
                .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        
            let iterations = 1000;
            let (num_rows, num_cols) = (2, 2);

            let mut lattice = Lattice::default_impl();

            lattice.electrical_synapse = false;
            lattice.chemical_synapse = true;
            
            lattice.populate(
                &base_neuron, 
                num_rows, 
                num_cols, 
            );
        
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
                .insert(IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default())
                .expect("Valid neurotransmitter pairing");
            base_neuron.synaptic_neurotransmitters
                .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        
            let iterations = 1000;
            let (num_rows, num_cols) = (2, 2);

            let mut lattice = Lattice::default_impl();

            lattice.electrical_synapse = true;
            lattice.chemical_synapse = true;
            
            lattice.populate(
                &base_neuron, 
                num_rows, 
                num_cols, 
            );
        
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
        let base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
    
        let iterations = 1000;

        let mut lattice1 = Lattice::default_impl();
        
        lattice1.populate(
            &base_neuron, 
            2, 
            2, 
        );
    
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
        );

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

        network.electrical_synapse = true;
        network.chemical_synapse = false;

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
        let base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };
    
        let iterations = 1000;

        let mut lattice1 = Lattice::default_impl();
        
        lattice1.populate(
            &base_neuron, 
            2, 
            2, 
        );
    
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
        );

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

        network.electrical_synapse = true;
        network.chemical_synapse = false;

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
}
