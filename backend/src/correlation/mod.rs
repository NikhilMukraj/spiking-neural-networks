//! A tool to calculate the Pearson correlation coefficient.

use std::result::Result;
use crate::error::TimeSeriesProcessingError;


fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn std(values: &[f32], values_mean: f32) -> f32 {
    values.iter()
        .map(|i| (i - values_mean).powf(2.0))
        .sum()
}

/// Calculates the Pearson correlation coefficient given two vectors of the same length (if standard
/// deviation of either of the vectors is 0, `f32::NAN` is returned)
pub fn pearsonr(x: &[f32], y: &[f32]) -> Result<f32, TimeSeriesProcessingError> {
    if x.len() != y.len() {
        return Err(
            TimeSeriesProcessingError::SeriesAreNotSameLength
        );
    }
    
    let x_mean: f32 = mean(x);
    let y_mean: f32 = mean(y);

    let numerator: f32 = x.iter().zip(y.iter())
        .map(|(i, j)| (i - x_mean) * (j - y_mean))
        .sum();

    let x_std: f32 = std(x, x_mean);
    let y_std: f32 = std(y, y_mean);

    let denominator: f32 = (x_std * y_std).powf(0.5);

    Ok(numerator / denominator) // returns nan x_std or y_std is 0
}
