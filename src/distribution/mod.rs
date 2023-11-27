use rand_distr::{Normal, Distribution};


pub fn limited_distr(mean: f64, std_dev: f64, minimum: f64, maximum: f64) -> f64 {
    if std_dev == 0.0 {
        return mean;
    }

    let normal = Normal::new(mean, std_dev).unwrap();
    let output: f64 = normal.sample(&mut rand::thread_rng());
   
    output.max(minimum).min(maximum)
}
