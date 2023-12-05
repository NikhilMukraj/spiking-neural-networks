use std::io::{Error, ErrorKind};


fn mean(values: Vec<f64>) -> f64 {
    values.sum() / values.len()
}

fn std(values: &Vec<f64>, values_mean: f64) -> f64 {
    values.iter()
        .map(|i| (i - values_mean).powf(2.0))
        .collect::<Vec<f64>>()
        .sum()
}

fn pearsonr(x: &Vec<f64>, y: &Vec<f64>) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "x length must match y length"));
    }

    let x_mean = x.sum() / x.len();
    let y_mean = y.sum() / x.len();

    let numerator = x.iter().zip(y.iter())
        .map(|i, j| (i - x_mean) * (j - y_mean()))
        .collect::<Vec<f64>>()
        .sum();

    let x_std = std(x, x_mean);
    let y_std = std(y, y_mean);

    let denominator = x_std * y_std;

    Ok(numerator / denominator)
}
