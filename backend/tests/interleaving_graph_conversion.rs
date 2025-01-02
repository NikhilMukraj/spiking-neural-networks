#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use opencl3::{command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE}, context::Context, device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU}};
    use rand::Rng;
    use spiking_neural_networks::{
        error::{GPUError, SpikingNeuralNetworksError},
        graph::{AdjacencyMatrix, Graph, GraphPosition, InterleavingGraphGPU}, 
        neuron::{
            integrate_and_fire::QuadraticIntegrateAndFireNeuron, Lattice, LatticeNetwork
        }
    };

    fn connection_conditional(x: (usize, usize), y: (usize, usize)) -> bool {
        ((x.0 as f64 - y.0 as f64).powf(2.) + (x.1 as f64 - y.1 as f64).powf(2.)).sqrt() <= 2. && 
        rand::thread_rng().gen_range(0.0..=1.0) <= 0.8 &&
        x != y
    }

    #[test]
    pub fn test_graph_conversion_isolated_lattices() -> Result<(), SpikingNeuralNetworksError> {
        let base_neuron = QuadraticIntegrateAndFireNeuron {
            gap_conductance: 0.1,
            ..QuadraticIntegrateAndFireNeuron::default_impl()
        };

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

        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Could not get GPU devices")
            .first()
            .expect("No GPU found");
        let device = Device::new(device_id);

        let context = match Context::from_device(&device) {
            Ok(value) => value,
            Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
        };

        let queue = match CommandQueue::create_default_with_properties(
                &context, 
                CL_QUEUE_PROFILING_ENABLE,
                CL_QUEUE_SIZE,
            ) {
                Ok(value) => value,
                Err(_) => return Err(Into::into(GPUError::GetDeviceFailure)),
            };

        let connecting_graph = network.get_connecting_graph();

        let gpu_graph = InterleavingGraphGPU::convert_to_gpu(
            &context, &queue, network.get_lattices(), network.get_spike_train_lattices(), connecting_graph
        )?;

        let mut editable_connecting_graph = AdjacencyMatrix::<GraphPosition, f32>::default();
        let mut editable_lattices: HashMap<usize, _> = network.get_all_lattice_ids()
            .iter()
            .map(|key| 
                (*key, {
                    let lattice_to_copy = network.get_lattice(key).unwrap();
                    let mut current_lattice = Lattice::default_impl();
                    current_lattice.set_id(*key);
                    current_lattice.populate(
                        &base_neuron, 
                        lattice_to_copy.cell_grid().len(),
                        lattice_to_copy.cell_grid().first().unwrap_or(&vec![]).len(),
                    );

                    current_lattice
                })
            )
            .collect();

        InterleavingGraphGPU::convert_to_cpu(
            &queue, &gpu_graph, &mut editable_lattices, &mut editable_connecting_graph
        )?;

        for row in editable_connecting_graph.matrix.iter() {
            for i in row {
                assert!(i.is_none());
            }
        }

        for i in network.get_all_ids() {
            let actual_index_to_position = &editable_lattices.get(&i).unwrap().graph().index_to_position;
            let expected_index_to_position = &network.get_lattice(&i).unwrap().graph().index_to_position;
            assert_eq!(
                actual_index_to_position, 
                expected_index_to_position,
            );

            for pre in actual_index_to_position.values() {
                for post in actual_index_to_position.values() {
                    let actual_weight = editable_lattices.get(&i).unwrap().graph().lookup_weight(pre, post)
                        .unwrap();
                    let expected_weight = network.get_lattice(&i).unwrap().graph().lookup_weight(pre, post)
                        .unwrap();

                    assert_eq!(actual_weight, expected_weight);
                }
            }
        }

        Ok(())
    }

    // #[test]
    // pub fn test_graph_conversion_connected_lattices() -> Result<(), SpikingNeuralNetworksError> {

    // }

    // #[test]
    // pub fn test_graph_conversion_spike_train_lattices() -> Result<(), SpikingNeuralNetworksError> {
        
    // }

    // #[test]
    // pub fn test_graph_conversion_spike_train_and_regular_lattices() -> Result<(), SpikingNeuralNetworksError> {

    // }
}
