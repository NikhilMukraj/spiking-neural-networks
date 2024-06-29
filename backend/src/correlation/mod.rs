//! A tool to calculate the Pearson correlation coefficient.

use std::result::Result;
use crate::error::{TimeSeriesProcessingError, TimeSeriesProcessingErrorKind};


fn mean(values: &Vec<f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &Vec<f64>, values_mean: f64) -> f64 {
    values.iter()
        .map(|i| (i - values_mean).powf(2.0))
        .sum()
}

/// Calculates the Pearson correlation coefficient given two vectors of the same length (if standard
/// deviation of either of the vectors is 0, `f64::NAN` is returned)
pub fn pearsonr(x: &Vec<f64>, y: &Vec<f64>) -> Result<f64, TimeSeriesProcessingError> {
    if x.len() != y.len() {
        return Err(
            TimeSeriesProcessingError::new(
                TimeSeriesProcessingErrorKind::SeriesAreNotSameLength, file!(), line!()
            )
        );
    }
    
    let x_mean: f64 = mean(x);
    let y_mean: f64 = mean(y);

    let numerator: f64 = x.iter().zip(y.iter())
        .map(|(i, j)| (i - x_mean) * (j - y_mean))
        .sum();

    let x_std: f64 = std(x, x_mean);
    let y_std: f64 = std(y, y_mean);

    let denominator: f64 = (x_std * y_std).powf(0.5);

    Ok(numerator / denominator) // returns nan x_std or y_std is 0
}
