#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use opencl3::{command_queue::CL_QUEUE_PROFILING_ENABLE, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}};
    use rand::Rng;
    use spiking_neural_networks::{error::SpikingNeuralNetworksError, graph::{AdjacencyMatrix, GraphPosition}, neuron::{gpu_lattices::LatticeNetworkGPU, integrate_and_fire::QuadraticIntegrateAndFireNeuron, iterate_and_spike::{ApproximateNeurotransmitter, ApproximateReceptor, IonotropicNeurotransmitterType}, plasticity::STDP, spike_train::{DeltaDiracRefractoriness, RateSpikeTrain}, GridVoltageHistory, Lattice, LatticeNetwork, RunNetwork, SpikeTrainGridHistory, SpikeTrainLattice}};


    neuron_builder!(
        "[neural_refractoriness]
            type: TestRefractoriness
            effect: (v_th - v_resting) * exp((-1 / (decay / dt)) * (time_difference ^ 2)) + v_resting
        [end]"
    );

    type GridType = Vec<Vec<TestRefractoriness>>;

    #[test]
    pub fn test_empty_grid_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![];

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = TestRefractoriness::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        TestRefractoriness::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_grid_of_empty_grids_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let cell_grid: GridType = vec![vec![], vec![]];

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = TestRefractoriness::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        TestRefractoriness::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        Ok(())
    }

    #[test]
    pub fn test_refractoriness_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![TestRefractoriness::default(), TestRefractoriness::default()],
            vec![TestRefractoriness::default(), TestRefractoriness::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.decay = rand::thread_rng().gen_range(0.0..10000.);
            }
        }

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = TestRefractoriness::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        TestRefractoriness::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.decay, 
                    expected.decay,
                );
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_refractoriness_conversion_non_square() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![TestRefractoriness::default(), TestRefractoriness::default()],
            vec![TestRefractoriness::default(), TestRefractoriness::default()],
            vec![TestRefractoriness::default(), TestRefractoriness::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.decay = rand::thread_rng().gen_range(0.0..10000.);
            }
        }

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = TestRefractoriness::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        TestRefractoriness::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            3,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.decay, 
                    expected.decay,
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_kernel_compiles() {
        #[allow(clippy::type_complexity)]
        let network: Result<
            LatticeNetworkGPU<
                QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>,
                AdjacencyMatrix<(usize, usize), f32>,
                GridVoltageHistory,
                RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, TestRefractoriness>,
                SpikeTrainGridHistory,
                STDP,
                IonotropicNeurotransmitterType,
                TestRefractoriness,
                AdjacencyMatrix<GraphPosition, f32>,
            >, 
            GPUError> = LatticeNetworkGPU::try_default();

        assert!(network.is_ok());
    }

    #[test]
    fn test_gpu_effect() -> Result<(), SpikingNeuralNetworksError> {
        let reference_spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain { rate: 100., ..Default::default() };
        let test_spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, TestRefractoriness> = RateSpikeTrain { rate: 100., ..Default::default() };

        let base_neuron = QuadraticIntegrateAndFireNeuron::default_impl();

        let mut reference_spike_train_lattice = SpikeTrainLattice::default_impl();
        reference_spike_train_lattice.populate(&reference_spike_train, 1, 1)?;

        let mut test_spike_train_lattice = SpikeTrainLattice::default_impl();
        test_spike_train_lattice.populate(&test_spike_train, 1, 1)?;

        let mut lattice = Lattice::default_impl();
        lattice.set_id(1);
        lattice.update_grid_history = true;
        lattice.populate(&base_neuron, 1, 1)?;

        let lattices = vec![lattice];
        let reference_spike_trains = vec![reference_spike_train_lattice];
        let reference_network = LatticeNetwork::generate_network(lattices.clone(), reference_spike_trains)?;
        let mut reference_network = LatticeNetworkGPU::from_network(reference_network)?;
        reference_network.connect(0, 1, &(|x, y| x == y), None)?;

        let test_spike_trains = vec![test_spike_train_lattice];
        let test_network = LatticeNetwork::generate_network(lattices, test_spike_trains)?;
        let mut test_network = LatticeNetworkGPU::from_network(test_network)?;
        test_network.connect(0, 1, &(|x, y| x == y), None)?;

        reference_network.run_lattices(1000)?;
        test_network.run_lattices(1000)?;

        assert_eq!(reference_network.get_lattice(&1).unwrap().grid_history.history.len(), 1000);
        assert_eq!(test_network.get_lattice(&1).unwrap().grid_history.history.len(), 1000);

        for (ref_grid, test_grid) in reference_network.get_lattice(&1).unwrap().grid_history.history.iter()
            .zip(test_network.get_lattice(&1).unwrap().grid_history.history.iter()) {
                for (ref_row, test_row) in ref_grid.iter().zip(test_grid.iter()) {
                    for (ref_voltage, test_voltage) in ref_row.iter().zip(test_row.iter()) {
                        assert!((ref_voltage - test_voltage).abs() < 1.);
                    }
                }
        }

        Ok(())
    }
}