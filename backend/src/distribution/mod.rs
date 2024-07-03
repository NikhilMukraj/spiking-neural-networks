//! A tool to generate and clamp noise.

use rand_distr::{Normal, Distribution};


/// Calculates the normal distribution at the given mean and standard deviation and clamps
/// the output value between the given minimum and maximum, if standard deviation is `0.` the 
/// mean is always returned
pub fn limited_distr(mean: f32, std: f32, minimum: f32, maximum: f32) -> f32 {
    if std == 0.0 {
        return mean;
    }

    let normal = Normal::new(mean, std).unwrap();
    let output: f32 = normal.sample(&mut rand::thread_rng());
   
    output.max(minimum).min(maximum)
}
