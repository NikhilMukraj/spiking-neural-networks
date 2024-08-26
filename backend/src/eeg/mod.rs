//! A set of tools to analyze power spectral density for EEG time series.

use std::result::Result;
use ndarray::{Array1, s};
use num_complex::Complex;
use rustfft::{FftPlanner, FftDirection};
mod emd;
use emd::earth_moving_distance;
use crate::error::TimeSeriesProcessingError;


/// Retrieves the power density of the given time series based on the given timestep (ms)
/// and total time elapsed by the end of the series (ms), returns tuple of power spectrum 
/// and associated frequency respectively
pub fn get_power_density(x: Vec<f32>, dt: f32, total_time: f32) -> (Array1<f32>, Array1<f32>) {
    let x_mean = x.iter().sum::<f32>() / x.len() as f32;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(x.len(), FftDirection::Forward);
    
    let mut x_fft: Array1<Complex<f32>> = Array1::zeros(x.len());
    for (i, &x_i) in x.iter().enumerate() {
        x_fft[i] = Complex::new(x_i - x_mean, 0.0);
    }
    
    let x_fft_slice: &mut [Complex<f32>] = x_fft.as_slice_mut().unwrap();
    fft.process(x_fft_slice);

    let x_fft_array: Array1<Complex<f32>> = x_fft_slice.to_owned().into();

    let sxx: Array1<Complex<f32>> = x_fft_array.mapv(|val| {
        let conj = val.conj();
        2.0 * dt.powi(2) / (x.len() as f32 * dt) * (val * conj)
    });

    let sxx_positive = sxx.slice(s![0..(x.len() / 2)]);
    let sxx_positive = sxx_positive.mapv(|val| val.re);

    let df: f32 = 1.0 / total_time;

    let fnq: f32 = 1.0 / (2.0 * dt);

    let faxis: Array1<f32> = Array1::range(0.0, fnq, df);

    (faxis, sxx_positive.to_owned())
}

fn find_max(arr: &Array1<f32>) -> Option<&f32> {
    arr.iter().max_by(|a, b| a.total_cmp(b))
}

/// Compares two power densities spectra using the earth moving distance, 
/// it assumes the same frequency range for each argument, 
/// (only compares the second item of [`get_power_density`])
pub fn power_density_comparison(sxx1: &Array1<f32>, sxx2: &Array1<f32>) -> Result<f32, TimeSeriesProcessingError> {
    if sxx1.len() != sxx2.len() {
        return Err(TimeSeriesProcessingError::SeriesAreNotSameLength);
    }

    let values = (0..sxx1.len()).map(|x| x as f32)
        .collect::<Vec<f32>>();

    let u_values = Array1::from(values.clone());
    let v_values = Array1::from(values);

    let u_max = find_max(sxx1).expect("Cannot find maximum");
    let v_max = find_max(sxx2).expect("Cannot find maximum");

    let u_weights = sxx1.map(|x| x / u_max);
    let v_weights = sxx2.map(|x| x / v_max);

    // scale earth moving distance based on heights
    Ok(earth_moving_distance(u_values, v_values, u_weights, v_weights) * (u_max - v_max).powf(2.))
}
