use std::io::{Result, Error, ErrorKind};
use csv::Reader;
use ndarray::{Array1, s};
use num_complex::Complex;
use rustfft::{FftPlanner, FftDirection};
mod emd;
use emd::earth_moving_distance;


pub fn get_power_density(x: Vec<f64>, dt: f64, total_time: f64) -> (Array1<f64>, Array1<f64>) {
    let x_mean = x.iter().sum::<f64>() / x.len() as f64;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(x.len(), FftDirection::Forward);
    
    let mut x_fft: Array1<Complex<f64>> = Array1::zeros(x.len());
    for (i, &x_i) in x.iter().enumerate() {
        x_fft[i] = Complex::new(x_i - x_mean, 0.0);
    }
    
    let x_fft_slice: &mut [Complex<f64>] = x_fft.as_slice_mut().unwrap();
    fft.process(x_fft_slice);

    let x_fft_array: Array1<Complex<f64>> = x_fft_slice.to_owned().into();

    let sxx: Array1<Complex<f64>> = x_fft_array.mapv(|val| {
        let conj = val.conj();
        2.0 * dt.powi(2) / (x.len() as f64 * dt) * (val * conj)
    });

    let sxx_positive = sxx.slice(s![0..(x.len() / 2)]);
    let sxx_positive = sxx_positive.mapv(|val| val.re);

    let df: f64 = 1.0 / total_time as f64;

    let fnq: f64 = 1.0 / (2.0 * dt);

    let faxis: Array1<f64> = Array1::range(0.0, fnq, df);

    return (faxis, sxx_positive.to_owned())
}

pub fn read_eeg_csv(filename: &str) -> Result<(Vec<f64>, f64, f64)> {
    let mut x: Vec<f64> = Vec::new();
    let mut reader = Reader::from_path(filename)?;

    let headers = reader.headers()?;
    let headers = headers
        .iter()
        .collect::<Vec<&str>>()[0];

    let header_parsed = headers.split(":").collect::<Vec<&str>>();
    let dt = match header_parsed[0].parse::<f64>() {
        Ok(parsed_val) => { parsed_val },
        Err(e) => { return Err(Error::new(ErrorKind::InvalidInput, e.to_string())) },
    };
    let total_time = match header_parsed[1].parse::<f64>() {
        Ok(parsed_val) => { parsed_val },
        Err(e) => { return Err(Error::new(ErrorKind::InvalidInput, e.to_string())) },
    };
    println!("EEG parameters:\ndt: {}\nT: {}", dt, total_time);

    for record in reader.records() {
        let record = match record {
            Ok(record_val) => { record_val },
            Err(e) => { return Err(Error::new(ErrorKind::InvalidInput, e.to_string())) },
        };

        let record = match record[0].parse::<f64>() {
            Ok(record_float) => { record_float },
            Err(e) => { return Err(Error::new(ErrorKind::InvalidInput, e.to_string())) },
        };

        x.push(record);
    }

    Ok((x, dt, total_time))
}

// to be replaced with more detailed metric
// pub fn power_density_mse(sxx1: &Array1<f64>, sxx2: &Array1<f64>) -> Result<f64> {
//     if sxx1.len() != sxx2.len() {
//         return Err(Error::new(ErrorKind::InvalidInput, "Lengths of inputs must match"));
//     }

//     let mse = sxx1.iter()
//         .zip(sxx2.iter())
//         .map(|(x, y)| (x - y).powf(2.0))
//         .sum();

//     Ok(mse)
// }

fn find_max(arr: &Array1<f64>) -> Option<&f64> {
    arr.iter().max_by(|a, b| a.total_cmp(b))
}

// mse isnt great metric
// this should be more applicable
pub fn power_density_comparison(sxx1: &Array1<f64>, sxx2: &Array1<f64>) -> Result<f64> {
    if sxx1.len() != sxx2.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "Lengths of inputs must match"));
    }

    let values = (0..sxx1.len()).map(|x| x as f64)
        .collect::<Vec<f64>>();

    let u_values = Array1::from(values.clone());
    let v_values = Array1::from(values);

    let u_max = find_max(&sxx1).expect("Cannot find maximum");
    let v_max = find_max(&sxx2).expect("Cannot find maximum");

    let u_weights = sxx1.map(|x| x / u_max);
    let v_weights = sxx2.map(|x| x / v_max);

    // scale earth moving distance based on heights
    Ok(earth_moving_distance(u_values, v_values, u_weights, v_weights) * (u_max - v_max).powf(2.))
}
