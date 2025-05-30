#[cfg(test)]
mod test {
    use rand::Rng;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        graph::{AdjacencyMatrix, GraphPosition}, 
        neuron::{
            gpu_lattices::LatticeNetworkGPU, 
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                AMPAReceptor, ApproximateNeurotransmitter, ApproximateReceptor, 
                IonotropicNeurotransmitterType, IonotropicType, Receptors, Timestep
            }, 
            plasticity::STDP, 
            spike_train::{DeltaDiracRefractoriness, RateSpikeTrain}, 
            GridVoltageHistory, Lattice, LatticeNetwork, RunNetwork, 
            SpikeTrainGridHistory, SpikeTrainLattice
        }
    };


    type SpikeTrainType = RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;
    type SpikeTrainLatticeType = SpikeTrainLattice<IonotropicNeurotransmitterType, SpikeTrainType, SpikeTrainGridHistory>;
    type NeuronType = QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>;
    type LatticeType = Lattice<NeuronType, AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, STDP, IonotropicNeurotransmitterType>;
    type NetworkType = LatticeNetwork<
        NeuronType,
        AdjacencyMatrix<(usize, usize), f32>,
        GridVoltageHistory,
        SpikeTrainType,
        SpikeTrainGridHistory,
        AdjacencyMatrix<GraphPosition, f32>,
        STDP,
        IonotropicNeurotransmitterType,
    >;

    fn check_history(history: &[Vec<Vec<f32>>], gpu_history: &[Vec<Vec<f32>>], tolerance: f32) {
        assert_eq!(history.len(), gpu_history.len());

        for (n, (cpu_cell_grid, gpu_cell_grid)) in history.iter().zip(gpu_history.iter()).enumerate() {
            for (row1, row2) in cpu_cell_grid.iter().zip(gpu_cell_grid) {
                for (voltage1, voltage2) in row1.iter().zip(row2.iter()) {
                    if *voltage1 == -75. || *voltage2 == -75. {
                        continue;
                    }
                    let error = (voltage1 - voltage2).abs();
                    assert!(
                        error <= tolerance, "{} | error: {}, voltage1: {}, voltage2: {}", 
                        n,
                        error,
                        voltage1,
                        voltage2,
                    );
                }
            }
        }
    }

    const ITERATIONS: usize = 1000;

    #[test]
    fn test_spike_train_lattice_alone() -> Result<(), SpikingNeuralNetworksError> {
        let spike_train: SpikeTrainType = RateSpikeTrain { rate: 100., dt: 1., ..Default::default() }; 

        let mut spike_train_lattice: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice.populate(&spike_train, 3, 3)?;
        spike_train_lattice.apply(|neuron: &mut SpikeTrainType| neuron.step = rand::thread_rng().gen_range(0.0..=100.));
        spike_train_lattice.update_grid_history = true;
        let spike_train_lattices: Vec<SpikeTrainLatticeType> = vec![spike_train_lattice];

        let mut network: NetworkType = LatticeNetwork::generate_network(vec![], spike_train_lattices)?;
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(ITERATIONS)?;
        gpu_network.run_lattices(ITERATIONS)?;

        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        Ok(())
    }

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 && x != y
    }

    fn test_spike_train_lattice_connected_to_lattice(electrical_synapse: bool, chemical_synapse: bool, rate: f32) -> Result<(), SpikingNeuralNetworksError> {
        let mut spike_train: SpikeTrainType = RateSpikeTrain { rate, dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        let mut spike_train_lattice: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice.populate(&spike_train, 3, 3)?;
        spike_train_lattice.apply(|neuron: &mut SpikeTrainType| neuron.step = rand::thread_rng().gen_range(0.0..=rate));
        spike_train_lattice.update_grid_history = true;
        let spike_train_lattices: Vec<SpikeTrainLatticeType> = vec![spike_train_lattice];

        let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        base_neuron.set_dt(1.);

        let mut lattice: LatticeType = Lattice::default();
        lattice.set_id(1);
        lattice.populate(&base_neuron, 3, 3)?;
        lattice.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th));
        lattice.update_grid_history = true;
        lattice.connect(&connection_conditional, None);
        let lattices: Vec<LatticeType> = vec![lattice];

        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;
        network.connect(0, 1, &(|_, _| true), Some(&(|_, _| 1.)))?; 
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(ITERATIONS)?;
        gpu_network.run_lattices(ITERATIONS)?;

        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        let cpu_history = &network.get_lattice(&1).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&1).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 5.);

        Ok(())
    }

    #[test]
    fn test_spike_train_lattice_connected_to_lattice_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_lattice_connected_to_lattice(true, false, 100.)?;

        Ok(())
    }

    #[test]
    fn test_spike_train_lattice_connected_to_lattice_chemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_lattice_connected_to_lattice(false, true, 100.)?;

        Ok(())
    }

    #[test]
    fn test_spike_train_lattice_connected_to_lattice_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_lattice_connected_to_lattice(true, true, 100.)?;

        Ok(())
    }

    fn test_multiple_spike_train_lattices_connected_to_lattice(electrical_synapse: bool, chemical_synapse: bool, rate1: f32, rate2: f32) -> Result<(), SpikingNeuralNetworksError> { 
        let mut spike_train1: SpikeTrainType = RateSpikeTrain { rate: rate1, dt: 1., ..Default::default() }; 

        spike_train1.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        let mut spike_train_lattice1: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice1.populate(&spike_train1, 3, 3)?;
        spike_train_lattice1.apply(|neuron: &mut SpikeTrainType| neuron.step = rand::thread_rng().gen_range(0.0..=rate1));
        spike_train_lattice1.update_grid_history = true;

        let mut spike_train2: SpikeTrainType = RateSpikeTrain { rate: rate2, dt: 1., ..Default::default() };

        spike_train2.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default()); 

        let mut spike_train_lattice2: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice2.set_id(1);
        spike_train_lattice2.populate(&spike_train2, 3, 3)?;
        spike_train_lattice2.apply(|neuron: &mut SpikeTrainType| neuron.step = rand::thread_rng().gen_range(0.0..=rate2));
        spike_train_lattice2.update_grid_history = true;
        let spike_train_lattices: Vec<SpikeTrainLatticeType> = vec![spike_train_lattice1, spike_train_lattice2];
        
        let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        base_neuron.set_dt(1.);

        let mut lattice: LatticeType = Lattice::default();
        lattice.set_id(2);
        lattice.populate(&base_neuron, 3, 3)?;
        lattice.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th));
        lattice.update_grid_history = true;
        lattice.connect(&connection_conditional, None);
        let lattices: Vec<LatticeType> = vec![lattice];

        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;
        network.connect(0, 2, &(|_, _| true), Some(&(|_, _| 0.2)))?; 
        network.connect(1, 2, &(|_, _| true), Some(&(|_, _| 0.3)))?; 
        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(ITERATIONS)?;
        gpu_network.run_lattices(ITERATIONS)?;

        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        let cpu_history = &network.get_spike_train_lattice(&1).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&1).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        let cpu_history = &network.get_lattice(&2).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&2).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 5.);

        Ok(())
    }

    #[test]
    fn test_mutiple_spike_train_lattices_connected_to_lattice_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_multiple_spike_train_lattices_connected_to_lattice(true, false, 100., 150.)?;

        Ok(())
    }

    #[test]
    fn test_mutiple_spike_train_lattices_connected_to_lattice_chemical() -> Result<(), SpikingNeuralNetworksError> {
        test_multiple_spike_train_lattices_connected_to_lattice(false, true, 100., 150.)?;

        Ok(())
    }

    #[test]
    fn test_mutiple_spike_train_lattices_connected_to_lattice_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        test_multiple_spike_train_lattices_connected_to_lattice(true, true, 100., 150.)?;

        Ok(())
    }

    fn test_spike_train_with_multiple_lattices(electrical_synapse: bool, chemical_synapse: bool, rate: f32) -> Result<(), SpikingNeuralNetworksError> { 
        let mut spike_train: SpikeTrainType = RateSpikeTrain { rate, dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        let mut spike_train_lattice: SpikeTrainLatticeType = SpikeTrainLattice::default();
        spike_train_lattice.set_id(0);
        spike_train_lattice.populate(&spike_train, 3, 3)?;
        spike_train_lattice.apply(|neuron: &mut SpikeTrainType| neuron.step = rand::thread_rng().gen_range(0.0..=100.));
        spike_train_lattice.update_grid_history = true;

        let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());
        base_neuron.set_dt(1.);

        let mut lattice1: LatticeType = Lattice::default();
        lattice1.set_id(1);
        lattice1.populate(&base_neuron, 3, 3)?;
        lattice1.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th));
        lattice1.connect(&(|x, y| x != y), Some(&(|_, _| 5.0)));
        lattice1.update_grid_history = true;

        let mut lattice2: LatticeType = Lattice::default();
        lattice2.set_id(2);
        lattice2.populate(&base_neuron, 2, 2)?;
        lattice2.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th));
        lattice2.connect(&(|x, y| x != y), Some(&(|_, _| 3.0)));
        lattice2.update_grid_history = true;

        let lattices = vec![lattice1, lattice2];
        let spike_trains = vec![spike_train_lattice];
        let mut network: NetworkType = LatticeNetwork::generate_network(lattices, spike_trains)?;

        network.connect(1, 2, &(|x, y| x == y), Some(&(|_, _| 5.0)))?;
        network.connect(2, 1, &(|x, y| x == y), Some(&(|_, _| 3.0)))?;
        network.connect(0, 1, &(|x, y| x == y), Some(&(|_, _| 5.0)))?;
        
        network.electrical_synapse = electrical_synapse;
        network.chemical_synapse = chemical_synapse;

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(ITERATIONS)?;
        gpu_network.run_lattices(ITERATIONS)?;

        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        let cpu_history = &network.get_lattice(&1).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&1).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 3.);

        let cpu_history = &network.get_lattice(&2).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&2).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 3.);

        Ok(())
    }

    #[test]
    fn test_spike_train_with_multiple_lattices_electrical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_with_multiple_lattices(true, false, 100.)?;

        Ok(())
    }

    #[test]
    fn test_spike_train_with_multiple_lattices_chemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_with_multiple_lattices(false, true, 100.)?;

        Ok(())
    }

    #[test]
    fn test_spike_train_with_multiple_lattices_electrochemical() -> Result<(), SpikingNeuralNetworksError> {
        test_spike_train_with_multiple_lattices(true, true, 100.)?;

        Ok(())
    }

    #[test]
    fn test_single_spike_train_lattice_and_lattice() -> Result<(), SpikingNeuralNetworksError> {
        let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain { rate: 100., dt: 1., ..Default::default() }; 

        spike_train.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        let mut spike_train_lattice: SpikeTrainLattice<IonotropicNeurotransmitterType, RateSpikeTrain<_, _, _>, _> = SpikeTrainLattice::default();
        spike_train_lattice.set_id(0);
        spike_train_lattice.populate(&spike_train, 2, 2)?;
        spike_train_lattice.apply(|neuron: &mut _| neuron.step = rand::thread_rng().gen_range(0.0..=100.));
        spike_train_lattice.update_grid_history = true;

        let mut base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();
        base_neuron.gap_conductance = 10.;
        base_neuron.c_m = 25.;

        base_neuron.receptors
            .insert(IonotropicNeurotransmitterType::AMPA, IonotropicType::AMPA(AMPAReceptor::default()))
            .expect("Valid neurotransmitter pairing");
        base_neuron.synaptic_neurotransmitters
            .insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        let mut lattice = Lattice::default_impl();
        lattice.set_id(1);
        lattice.populate(&base_neuron, 2, 2)?;
        lattice.apply(|neuron: &mut _| neuron.current_voltage = rand::thread_rng().gen_range(neuron.v_reset..=neuron.v_th));
        lattice.connect(&(|x, y| x != y), Some(&(|_, _| 5.0)));
        lattice.update_grid_history = true;

        let lattices: Vec<Lattice<QuadraticIntegrateAndFireNeuron<_, _>, _, GridVoltageHistory, STDP, IonotropicNeurotransmitterType>> = vec![lattice];
        let spike_train_lattices: Vec<SpikeTrainLattice<IonotropicNeurotransmitterType, RateSpikeTrain<_, _, _>, SpikeTrainGridHistory>> = vec![spike_train_lattice];

        let mut network = LatticeNetwork::generate_network(lattices, spike_train_lattices)?;
        network.connect(0, 1, &(|x, y| x == y), Some(&(|_, _| 5.)))?;
        network.electrical_synapse = true;
        network.chemical_synapse = false;
        network.parallel = true;
        network.set_dt(1.);

        let mut gpu_network = LatticeNetworkGPU::from_network(network.clone())?;

        network.run_lattices(ITERATIONS)?;
        gpu_network.run_lattices(ITERATIONS)?;

        let cpu_history = &network.get_spike_train_lattice(&0).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_spike_train_lattice(&0).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 1.);

        let cpu_history = &network.get_lattice(&1).unwrap().grid_history.history;
        let gpu_history = &gpu_network.get_lattice(&1).unwrap().grid_history.history;

        assert_eq!(cpu_history.len(), ITERATIONS);
        assert_eq!(gpu_history.len(), ITERATIONS);

        check_history(cpu_history, gpu_history, 5.);

        Ok(())
    }
}
