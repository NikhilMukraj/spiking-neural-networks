// check regular conversion
// check electrochemical conversion

#[cfg(test)]
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
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, 
            iterate_and_spike::{
                AMPADefault, ApproximateNeurotransmitter, ApproximateReceptor, 
                IonotropicNeurotransmitterType, IterateAndSpikeGPU, LigandGatedChannel, 
                LigandGatedChannels, Neurotransmitters
            }
        }
    };

    type GridType = Vec<Vec<QuadraticIntegrateAndFireNeuron<ApproximateNeurotransmitter, ApproximateReceptor>>>;

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

        let gpu_conversion = QuadraticIntegrateAndFireNeuron::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        QuadraticIntegrateAndFireNeuron::convert_to_cpu(
            &mut cpu_conversion,
            &gpu_conversion,
            0,
            0,
            &queue,
        )?;

        assert_eq!(cpu_conversion.len(), 0);

        let gpu_conversion = QuadraticIntegrateAndFireNeuron::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        QuadraticIntegrateAndFireNeuron::convert_electrochemical_to_cpu(
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
        let mut cell_grid = vec![
            vec![QuadraticIntegrateAndFireNeuron::default_impl(), QuadraticIntegrateAndFireNeuron::default_impl()],
            vec![QuadraticIntegrateAndFireNeuron::default_impl(), QuadraticIntegrateAndFireNeuron::default_impl()],
        ];

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
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

        let gpu_conversion = QuadraticIntegrateAndFireNeuron::convert_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        QuadraticIntegrateAndFireNeuron::convert_to_cpu(
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
            }
        }

        Ok(())
    }

    #[test]
    pub fn test_neuron_electrochemical_conversion() -> Result<(), SpikingNeuralNetworksError> {
        let mut cell_grid = vec![
            vec![QuadraticIntegrateAndFireNeuron::default_impl(), QuadraticIntegrateAndFireNeuron::default_impl()],
            vec![QuadraticIntegrateAndFireNeuron::default_impl(), QuadraticIntegrateAndFireNeuron::default_impl()],
        ];

        let mut neurotransmitters = Neurotransmitters::default();
        neurotransmitters.insert(
            IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::ampa_default()
        );
        let mut ligand_gates = LigandGatedChannels::default();
        ligand_gates.insert(
            IonotropicNeurotransmitterType::AMPA, LigandGatedChannel::ampa_default()
        )?;

        cell_grid[1][1].synaptic_neurotransmitters = neurotransmitters;
        cell_grid[1][1].ligand_gates = ligand_gates;

        let mut cpu_conversion = cell_grid.clone();

        for row in cell_grid.iter_mut() {
            for i in row.iter_mut() {
                i.current_voltage = rand::thread_rng().gen_range(-75.0..-65.0);
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

        let gpu_conversion = QuadraticIntegrateAndFireNeuron::convert_electrochemical_to_gpu(
            &cell_grid,
            &context,
            &queue,
        )?;

        QuadraticIntegrateAndFireNeuron::convert_electrochemical_to_cpu(
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
                    actual.synaptic_neurotransmitters,
                    expected.synaptic_neurotransmitters,
                );
                assert_eq!(
                    actual.ligand_gates,
                    expected.ligand_gates,
                );
            }
        }

        Ok(())
    }
}