// #[path = "../neuron/mod.rs"]
// mod neuron;
// use crate::neuron::{Cell, HodgkinHuxleyCell};
// #[path = "../ga/mod.rs"]
// use crate::ga::{BitString, decode, genetic_algo};


// struct FittingSettings {
//     hodgkin_huxley_mode: HodgkinHuxleyCell,
//     izhikevich_defaults: &HashMap<&str, f64>,
// }

// // bounds should be a, b, c, d, and v_th for now
// // if fitting does not generalize, optimize other coefs in equation
// fn objective(
//     bitstring: &BitString, 
//     bounds: &Vec<Vec<f64>>, 
//     n_bits: usize, 
//     settings: &HashMap<&str, FittingSettings>
// ) -> Result<f64> {
    // let decoded = match decode(bitstring, bounds, n_bits) {
    //     Ok(decoded_value) => decoded_value,
    //     Err(e) => return Err(e),
    // };

    // let a: f64 = decoded[0];
    // let b: f64 = decoded[1];
    // let c: f64 = decoded[2];
    // let d: f64 = decoded[3];
    // let v_th: f64 = decoded[4];
// }