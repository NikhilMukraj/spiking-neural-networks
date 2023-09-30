use std::io;
use std::io::{Error, ErrorKind};
use std::collections::HashMap;
use rand::Rng;
use rayon::prelude::*;


#[derive(Clone)]
struct BitString {
    string: String
}

impl BitString {
    fn check(&self) -> io::Result<()> {
        for i in self.string.chars() {
            if i != '1' && i != '0' {
                return Err(Error::new(ErrorKind::Other, "Non binary found"));
            }
        }

        return Ok(());
    }

    fn set(&mut self, new_string: String) -> io::Result<()> {
        // check after initalization

        self.string = new_string;

        return self.check();
    }

    fn length(&self) -> i32 {
        self.string.len() as i32
    }
}

fn crossover(parent1: &BitString, parent2: &BitString, r_cross: f64) -> (BitString, BitString) {
    let mut rng_thread = rand::thread_rng(); 
    let (mut clone1, mut clone2) = (parent1.clone(), parent2.clone());

    if rng_thread.gen::<f64>() <= r_cross {
        let end_point = parent1.length();
        let crossover_point = rng_thread.gen_range(1..end_point); // change for variable length
        
        // c1 = p1[:pt] + p2[pt:]
		// c2 = p2[:pt] + p1[pt:]

        let string1 = format!("{}{}", &parent1.string[0..crossover_point as usize], &parent2.string[crossover_point as usize..]);
        let string2 = format!("{}{}", &parent2.string[0..crossover_point as usize], &parent1.string[crossover_point as usize..]);

        clone1.set(string1).expect("Error setting bitstring");
        clone2.set(string2).expect("Error setting bitstring");
    }

    return (clone1, clone2);
}

fn mutate(bitstring: &mut BitString, r_mut: f64) {
    let mut rng_thread = rand::thread_rng(); 
    for i in 0..bitstring.length() as usize {
        let do_mut = rng_thread.gen::<f64>() <= r_mut;

        // does in place bit flip if do_mut
        if do_mut && bitstring.string.chars().nth(i).unwrap() == '1' {
            bitstring.string.replace_range(i..i+1, "0"); 
        } else if do_mut && bitstring.string.chars().nth(i).unwrap() == '0' {
            bitstring.string.replace_range(i..i+1, "1");
        } 
    }
}

fn selection(pop: &Vec::<BitString>, scores: &Vec::<f64>, k: i32) -> BitString {
    // default should be 3
    let mut rng_thread = rand::thread_rng(); 
    let mut selection_index = rng_thread.gen_range(1..pop.len());

    let indices = (0..k-1)
        .into_iter()
        .map(|_x| rng_thread.gen_range(1..pop.len()));

    // performs tournament selection to select parents
    for i in indices {
        if scores[i] < scores[selection_index] {
            selection_index = i;
        }
    }

    return pop[selection_index].clone();
}

// decode in objective function
// write settings for objective function into hashmap

fn decode(bitstring: &BitString, bounds: &Vec<Vec<f64>>, n_bits: i32) -> Result<Vec<f64>, io::Error> {
    // decode for non variable length
    // for variable length just keep bounds consistent across all
    // determine substrings by calculating string.len() / n_bits
    if bounds.len() != bitstring.length() as usize / n_bits as usize {
        return Err(Error::new(ErrorKind::Other, "Bounds length does not match n_bits"));
    }
    if bitstring.length() as usize % n_bits as usize != 0 {
        return Err(Error::new(ErrorKind::Other, "String length is indivisible by n_bits"));
    }

    let maximum = i32::pow(2, n_bits as u32) as f64 - 1.;
    let mut decoded_vec = vec![0.; bounds.len()];

    let n_bits = n_bits as usize;
    for i in 0..bounds.len() {
        let (start, end) = (i * n_bits, (i * n_bits) + n_bits);
        let substring = &bitstring.string[start..end];

        // let mut value = i32::from_str_radix(substring, 2).expect("Non binary") as f64;
        let mut value = match i32::from_str_radix(substring, 2) {
            Ok(value_result) => value_result as f64,
            Err(_e) => return Err(Error::new(ErrorKind::Other, "Non binary found")),
        };
        value = value * (bounds[i][1] - bounds[i][0]) / maximum + bounds[i][0];

        decoded_vec[i] = value;
    }

    return Ok(decoded_vec);
}

fn objective<T>(
    bitstring: &BitString, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: i32, 
    settings: &HashMap<&str, T>
) -> Result<f64, io::Error> {
    // example objective function
    if bounds.len() != 1 {
        return Err(Error::new(ErrorKind::Other, "Bounds length must be 1"));
    }
    if !settings.contains_key("val") {
        return Err(Error::new(ErrorKind::Other, r#""val" not found"#));
    }

    let decoded = match decode(bitstring, bounds, n_bits) {
        Ok(decoded_value) => decoded_value,
        Err(_e) => return Err(Error::new(ErrorKind::Other, "Non binary found")),
    };
    let val: f64 = *settings.get("val").unwrap();

    return Ok(-1. * (decoded[0] - val));
}

fn create_random_string(length: usize) -> BitString {
    let mut rng_thread = rand::thread_rng(); 
    let mut random_string = String::from("");
    for _ in 0..length {
        if rng_thread.gen::<f64>() <= 0.5 {
            random_string.push('0');
        } else {
            random_string.push('1');
        }
    }

    return BitString {string: random_string};
}

// use par_iter to calculate objective scores
fn genetic_algo<T>(
    f: fn(&BitString, &Vec<Vec<f64>>, i32, &HashMap<&str, f64>) -> Result<f64, io::Error>, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: i32, 
    n_iter: i32, 
    n_pop: i32, 
    r_cross: f64,
    r_mut: f64, 
    k: i32, settings: &HashMap<&str, T>
) -> Result<(BitString, f64, Vec<Vec<f64>>), io::Error> {
    let mut pop: Vec<BitString> = (0..n_pop)
        .map(|_x| create_random_string(n_bits as usize * bounds.len()))
        .collect();
    // let (mut best, mut best_eval) = (&pop[0], objective(&pop[0], &bounds, n_bits, &settings));
    let mut best = pop[0].clone();
    let mut best_eval = match f(&pop[0], &bounds, n_bits, &settings) {
        Ok(best_eval_result) => best_eval_result,
        Err(e) => return Err(Error::new(ErrorKind::Other, e)),
    };

    let mut all_scores = vec![vec![]];

    for gen in 0..n_iter {
        println!("gen: {}", gen + 1);
        let scores_results: &Result<Vec<f64>, _> = &pop
            .par_iter() // https://users.rust-lang.org/t/calling-a-trait-object-within-a-rayon-par-iter-closure/63521
            .map(|p| f(p, &bounds, n_bits, &settings))
            .collect(); // maybe replace with for each and see if par iter works 
        // maybe try this instead
        // https://docs.rs/par-map/latest/par_map/
        // maybe implement by hand
        // https://doc.rust-lang.org/book/ch16-01-threads.html
        
        // check if objective failed anywhere
        let scores = match scores_results {
            Ok(scores_results) => scores_results,
            Err(e) => return Err(Error::new(ErrorKind::Other, e.to_string())),
        };

        all_scores.push(scores.clone());

        for i in 0..n_pop as usize {
            if scores[i] < best_eval {
                // let (mut best, mut best_eval) = (&pop[i], &scores[i]);
                best = pop[i].clone();
                best_eval = scores[i];
                println!("new string: {}, score: {}", &pop[i].string, &scores[i]);
            }
        }

        let selected: Vec<BitString> = (0..n_pop)
            .map(|_x| selection(&pop, &scores, k))
            .collect();

        // let new_strings = (0..n_pop)
        //     .into_par_iter()
        //     .map(|_| selection(&pop, &scores, k))
        //     .collect();

        let mut children: Vec<BitString> = Vec::new();
        for i in (0..n_pop as usize).step_by(2) {
            // let (parent1, parent2) = (selected[i].clone(), selected[i+1].clone());
            let new_children = crossover(&selected[i], &selected[i+1], r_cross);
            let mut new_children_vec = vec![new_children.0, new_children.1];
            for child in new_children_vec.iter_mut() {
                mutate(child, r_mut);
                children.push(child.clone());
            }
        }

        // let children = (0..n_pop)
        //     .into_par_iter()
        //     .step_by(2)
        //     .flat_map(|i| {
        //         let new_children = crossover(&selected[i], &selected[i + 1], r_cross);
        //         vec![new_children.0, new_children.1]
        //     })
        //     .map(|mut child| {
        //         mutate(&mut child, r_mut);
        //         child
        //     })
        //     .collect();

        pop = children;
    }

    return Ok((best, best_eval, all_scores));
}
