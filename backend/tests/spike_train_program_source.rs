// check that program compiles
// check that program emits spikes
// check that program correctly modifies neurotransmitters

// mod tests {
//     #[test]
//     pub fn test_program_source() {
//         let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
//             .expect("Could not get GPU devices")
//             .first()
//             .expect("No GPU found");
//         let device = Device::new(device_id);

//         let context = Context::from_device(&device).expect("Context::from_device failed");

//         let kernel_function = PoissonNeuron::<Generics>::
//             iterate_and_spike_electrical_kernel(&context);

//         assert!(kernel_function.is_ok());

//         let kernel_function = PoissonNeuron::<Generics>::
//             iterate_and_spike_electrochemical_kernel(&context);

//         assert!(kernel_function.is_ok());
//     }
// }
