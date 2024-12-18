mod tests {
    use rand::Rng;
    use opencl3::{
        command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
        context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    };
    extern crate spiking_neural_networks;
    use spiking_neural_networks::{
        error::SpikingNeuralNetworksError, 
        neuron::{
            iterate_and_spike::{
                AMPADefault, ApproximateNeurotransmitter, IonotropicNeurotransmitterType, NMDADefault, Neurotransmitters
            }, spike_train::{DeltaDiracRefractoriness, PoissonNeuron, SpikeTrainGPU}
        }
    };

    type GridType = Vec<Vec<PoissonNeuron<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>>>;

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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let mut cpu_conversion: GridType = vec![];

        let gpu_conversion = PoissonNeuron::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        PoissonNeuron::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        let gpu_conversion = PoissonNeuron::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        PoissonNeuron::convert_electrochemical_to_cpu(
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
    pub fn test_neuron_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![PoissonNeuron::default(), PoissonNeuron::default()],
            vec![PoissonNeuron::default(), PoissonNeuron::default()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                if rand::thread_rng().gen_range(0.0f32..1.0) < 0.5 {
                    i.current_voltage = i.v_resting;
                } else {
                    i.current_voltage = i.v_th;
                }
                i.chance_of_firing = rand::thread_rng().gen_range(0.0f32..0.5f32);
                i.dt = rand::thread_rng().gen_range(0.0f32..0.5f32);
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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = PoissonNeuron::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        PoissonNeuron::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.chance_of_firing, 
                    expected.chance_of_firing,
                );
                assert_eq!(
                    actual.dt, 
                    expected.dt,
                );
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_neuron_electrochemical_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid: GridType = vec![
            vec![PoissonNeuron::default(), PoissonNeuron::default()],
            vec![PoissonNeuron::default(), PoissonNeuron::default()],
        ];

        let mut neurotransmitters = Neurotransmitters::default();
        neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );

        cell_grid[1][1].synaptic_neurotransmitters = neurotransmitters.clone();

        neurotransmitters.insert(
            IonotropicNeurotransmitterType::NMDA, ApproximateNeurotransmitter::nmda_default()
        );

        cell_grid[0][0].synaptic_neurotransmitters = neurotransmitters;

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                if rand::thread_rng().gen_range(0.0f32..1.0) < 0.5 {
                    i.current_voltage = i.v_resting;
                } else {
                    i.current_voltage = i.v_th;
                }
                i.chance_of_firing = rand::thread_rng().gen_range(0.0f32..0.5f32);
                i.dt = rand::thread_rng().gen_range(0.0f32..0.5f32);
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
                CL_QUEUE_SIZE,
            )
            .expect("CommandQueue::create_default failed");

        let gpu_conversion = PoissonNeuron::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        PoissonNeuron::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            2,
            2,
            &queue,
        )?;

        for (row1, row2) in cpu_conversion.iter().zip(cell_grid.iter()) {
            for (actual, expected) in row1.iter().zip(row2.iter()) {
                assert_eq!(
                    actual.current_voltage, 
                    expected.current_voltage,
                );
                assert_eq!(
                    actual.chance_of_firing, 
                    expected.chance_of_firing,
                );
                assert_eq!(
                    actual.dt, 
                    expected.dt,
                );

                assert_eq!(
                    actual.synaptic_neurotransmitters,
                    expected.synaptic_neurotransmitters,
                );
            }
        }

        Ok(())
    }
}