use std::{
    io::{Error, ErrorKind, Result},
    collections::HashMap,
    marker::Sync,
};
use rand::Rng;
use rayon::prelude::*;


#[derive(Clone)]
pub struct BitString {
    pub string: String
}

impl BitString {
    fn check(&self) -> Result<()> {
        for i in self.string.chars() {
            if i != '1' && i != '0' {
                return Err(Error::new(ErrorKind::Other, format!("Non binary found: {}", self.string)));
            }
        }

        return Ok(());
    }

    fn set(&mut self, new_string: String) -> Result<()> {
        // check after initalization

        self.string = new_string;

        return self.check();
    }

    fn length(&self) -> usize {
        self.string.len()
    }
}

fn crossover(parent1: &BitString, parent2: &BitString, r_cross: f64) -> (BitString, BitString) {
    let mut rng_thread = rand::thread_rng(); 
    let (mut clone1, mut clone2) = (parent1.clone(), parent2.clone());

    if rng_thread.gen::<f64>() <= r_cross {
        let end_point = parent1.length();
        let crossover_point = rng_thread.gen_range(1..end_point); // change for variable length
       
        let string1 = format!("{}{}", &parent1.string[0..crossover_point as usize], &parent2.string[crossover_point as usize..]);
        let string2 = format!("{}{}", &parent2.string[0..crossover_point as usize], &parent1.string[crossover_point as usize..]);

        clone1.set(string1).expect("Error setting bitstring");
        clone2.set(string2).expect("Error setting bitstring");
    }

    return (clone1, clone2);
}

fn mutate(bitstring: &mut BitString, r_mut: f64) {
    let mut rng_thread = rand::thread_rng(); 
    for i in 0..bitstring.length() {
        let do_mut = rng_thread.gen::<f64>() <= r_mut;

        // does in place bit flip if do_mut
        if do_mut && bitstring.string.chars().nth(i).unwrap() == '1' {
            bitstring.string.replace_range(i..i+1, "0"); 
        } else if do_mut && bitstring.string.chars().nth(i).unwrap() == '0' {
            bitstring.string.replace_range(i..i+1, "1");
        } 
    }
}

fn selection(pop: &Vec::<BitString>, scores: &Vec::<f64>, k: usize) -> BitString {
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

pub fn decode(bitstring: &BitString, bounds: &Vec<Vec<f64>>, n_bits: usize) -> Result<Vec<f64>> {
    // decode for non variable length
    // for variable length just keep bounds consistent across all
    // determine substrings by calculating string.len() / n_bits
    if bounds.len() != bitstring.length() / n_bits {
        return Err(Error::new(ErrorKind::Other, "Bounds length does not match n_bits"));
    }
    if bitstring.length() % n_bits != 0 {
        return Err(Error::new(ErrorKind::Other, "String length is indivisible by n_bits"));
    }

    let maximum = i32::pow(2, n_bits as u32) as f64 - 1.;
    let mut decoded_vec = vec![0.; bounds.len()];

    for i in 0..bounds.len() {
        let (start, end) = (i * n_bits, (i * n_bits) + n_bits);
        let substring = &bitstring.string[start..end];

        let mut value = match i32::from_str_radix(substring, 2) {
            Ok(value_result) => value_result as f64,
            Err(_e) => return Err(
                Error::new(ErrorKind::Other, format!("Non binary substring found or overflow: {}", substring))
            ),
        };
        value = value * (bounds[i][1] - bounds[i][0]) / maximum + bounds[i][0];

        decoded_vec[i] = value;
    }

    return Ok(decoded_vec);
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
pub fn genetic_algo<T: Sync>(
    f: fn(&BitString, &Vec<Vec<f64>>, usize, &HashMap<&str, T>) -> Result<f64>, 
    bounds: &Vec<Vec<f64>>, 
    n_bits: usize, 
    n_iter: usize, 
    n_pop: usize, 
    r_cross: f64,
    r_mut: f64, 
    k: usize, 
    settings: &HashMap<&str, T>,
) -> Result<(BitString, f64, Vec<Vec<f64>>)> {
    let mut pop: Vec<BitString> = (0..n_pop)
        .map(|_x| create_random_string(n_bits * bounds.len()))
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
        let scores_results: &Result<Vec<f64>> = &pop
            .par_iter() 
            .map(|p| f(p, &bounds, n_bits, &settings))
            .collect(); 
        
        // check if objective failed anywhere
        let scores = match scores_results {
            Ok(scores_results) => scores_results,
            Err(e) => return Err(Error::new(ErrorKind::Other, e.to_string())),
        };

        all_scores.push(scores.clone());

        for i in 0..n_pop {
            if scores[i] < best_eval {
                best = pop[i].clone();
                best_eval = scores[i];
                println!("new string: {}, score: {}", &pop[i].string, &scores[i]);
            }
        }

        // parallel refactor needs testing
        // let min_score_index: Option<usize> = scores
        //     .iter()
        //     .enumerate()
        //     .min_by(|(_, a), (_, b)| a.total_cmp(b))
        //     .map(|(index, _)| index);

        // if scores[min_score_index] < best_eval {
        //     best = pop[min_score_index];
        //     best_eval = scores[min_score_index];
        //     println!("new string: {}, score: {}", &pop[min_score_index].string, &scores[min_score_index]);
        // }

        // let selected: Vec<BitString> = (0..n_pop)
        //     .map(|_x| selection(&pop, &scores, k))
        //     .collect();

        let selected: Vec<BitString> = (0..n_pop)
            .into_par_iter()
            .map(|_| selection(&pop, &scores, k))
            .collect();

        // let mut children: Vec<BitString> = Vec::new();
        // for i in (0..n_pop as usize).step_by(2) {
        //     // let (parent1, parent2) = (selected[i].clone(), selected[i+1].clone());
        //     let new_children = crossover(&selected[i], &selected[i+1], r_cross);
        //     let mut new_children_vec = vec![new_children.0, new_children.1];
        //     for child in new_children_vec.iter_mut() {
        //         mutate(child, r_mut);
        //         children.push(child.clone());
        //     }
        // }

        let children = (0..n_pop)
            .into_par_iter()
            .step_by(2)
            .flat_map(|i| {
                let new_children = crossover(&selected[i], &selected[i + 1], r_cross);
                vec![new_children.0, new_children.1]
            })
            .map(|mut child| {
                mutate(&mut child, r_mut);
                child
            })
            .collect();

        pop = children;
    }

    return Ok((best, best_eval, all_scores));
}
