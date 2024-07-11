//! A basic parallelized genetic algorithm implementation.

use std::{
    result::Result,
    collections::HashMap,
    marker::Sync,
};
use rand::Rng;
use rayon::prelude::*;
use crate::error::GeneticAlgorithmError;


/// Bit string to use as a chromosome
#[derive(Clone)]
pub struct BitString {
    pub string: String
}

impl BitString {
    pub fn new(new_string: String) -> Result<Self, GeneticAlgorithmError> {
        let bitstring = BitString { string: new_string };
        bitstring.check()?;

        Ok(bitstring)
    }

    fn check(&self) -> Result<(), GeneticAlgorithmError> {
        for i in self.string.chars() {
            if i != '1' && i != '0' {
                return Err(
                    GeneticAlgorithmError::NonBinaryInBitstring(self.string.clone())
                );
            }
        }

        return Ok(());
    }

    fn set(&mut self, new_string: String) -> Result<(), GeneticAlgorithmError> {
        self.string = new_string;

        return self.check();
    }

    /// Returns the length of the bit string
    pub fn length(&self) -> usize {
        self.string.len()
    }
}

fn crossover(parent1: &BitString, parent2: &BitString, r_cross: f32) -> (BitString, BitString) {
    let mut rng_thread = rand::thread_rng(); 
    let (mut clone1, mut clone2) = (parent1.clone(), parent2.clone());

    if rng_thread.gen::<f32>() <= r_cross {
        let end_point = parent1.length();
        let crossover_point = rng_thread.gen_range(1..end_point); // change for variable length
       
        let string1 = format!("{}{}", &parent1.string[0..crossover_point as usize], &parent2.string[crossover_point as usize..]);
        let string2 = format!("{}{}", &parent2.string[0..crossover_point as usize], &parent1.string[crossover_point as usize..]);

        clone1.set(string1).expect("Error setting bitstring");
        clone2.set(string2).expect("Error setting bitstring");
    }

    return (clone1, clone2);
}

fn mutate(bitstring: &mut BitString, r_mut: f32) {
    let mut rng_thread = rand::thread_rng(); 
    for i in 0..bitstring.length() {
        let do_mut = rng_thread.gen::<f32>() <= r_mut;

        // does in place bit flip if do_mut
        if do_mut && bitstring.string.chars().nth(i).unwrap() == '1' {
            bitstring.string.replace_range(i..i+1, "0"); 
        } else if do_mut && bitstring.string.chars().nth(i).unwrap() == '0' {
            bitstring.string.replace_range(i..i+1, "1");
        } 
    }
}

fn selection(pop: &Vec<BitString>, scores: &Vec<f32>, k: usize) -> BitString {
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

/// Decodes the given [`BitString`] given the number of bits in each bit substring and scales 
/// the output based on the given `bounds`, the length of the `bounds` must match the number
/// of substrings in the [`BitString`], `bounds` should be a vector of tuples where the first item is the 
/// minimum value for scaling and the second item is the maximal value for scaling
pub fn decode(
    bitstring: 
    &BitString, bounds: &Vec<(f32, f32)>, 
    n_bits: usize
) -> Result<Vec<f32>, GeneticAlgorithmError> {
    // decode for non variable length
    // for variable length just keep bounds consistent across all
    // determine substrings by calculating string.len() / n_bits
    if bounds.len() != bitstring.length() / n_bits {
        return Err(GeneticAlgorithmError::InvalidBoundsLength);
    }
    if bitstring.length() % n_bits != 0 {
        return Err(GeneticAlgorithmError::InvalidBitstringLength);
    }

    let maximum = i32::pow(2, n_bits as u32) as f32 - 1.;
    let mut decoded_vec = vec![0.; bounds.len()];

    for i in 0..bounds.len() {
        let (start, end) = (i * n_bits, (i * n_bits) + n_bits);
        let substring = &bitstring.string[start..end];

        let mut value = match i32::from_str_radix(substring, 2) {
            Ok(value_result) => value_result as f32,
            Err(_e) => return Err(
                GeneticAlgorithmError::DecodingBitstringFailure(String::from(substring))
            ),
        };
        value = value * (bounds[i].1 - bounds[i].0) / maximum + bounds[i].0;

        decoded_vec[i] = value;
    }

    return Ok(decoded_vec);
}

fn create_random_string(length: usize) -> BitString {
    let mut rng_thread = rand::thread_rng(); 
    let mut random_string = String::from("");
    for _ in 0..length {
        if rng_thread.gen::<f32>() <= 0.5 {
            random_string.push('0');
        } else {
            random_string.push('1');
        }
    }

    return BitString {string: random_string};
}

/// Parameter set for genetic algorithm functionality
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmParameters {
    /// `bounds` should be a vector of tuples where the first item is the 
    /// minimum value for scaling and the second item is the maximal value for scaling
    pub bounds: Vec<(f32, f32)>, 
    /// `n_bits` should be the number of bits per substring in each chromosomal [`BitString`]
    pub n_bits: usize, 
    /// `n_iter` should be the number of iterations to use
    pub n_iter: usize, 
    /// `n_pop` should be the size of the population and even
    pub n_pop: usize, 
    /// `r_cross` should be the chance of cross over
    pub r_cross: f32,
    /// `r_mut` should be the chance of mutation
    pub r_mut: f32, 
    /// `k` controls size of tournament during selection process, recommended to keep at `3`
    pub k: usize, 
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        GeneticAlgorithmParameters {
            bounds: vec![(0., 1.)],
            n_bits: 8,
            n_iter: 100,
            n_pop: 100,
            r_cross: 0.9,
            r_mut: 0.1,
            k: 3,
        }
    }
}

/// Executes the genetic algorithm given a objective function, (`f`), that takes in parameters
/// to decode the given [`BitString`] as well as additional settings to use in the
/// objective function, returns the best [`BitString`] in first item of tuple, the score of the
/// [`BitString`], and a vector of vectors containing the scores for each [`BitString`] over time
/// 
/// - `f` : the objective function to minimize the output of, should take in the [`BitString`], bounds,
/// number of bits per bit substring, and a hashmap of any necessary parameters as arguments
/// 
/// - `params` : a set of genetic algorithm parameters
///
/// - `settings` : any additional parameters necessary in the objective function
/// 
/// - `verbose` : use `true` to print extra information
pub fn genetic_algo<T: Sync>(
    f: fn(&BitString, &Vec<(f32, f32)>, usize, &HashMap<&str, T>) -> Result<f32, GeneticAlgorithmError>, 
    params: &GeneticAlgorithmParameters,
    settings: &HashMap<&str, T>,
    verbose: bool,
) -> Result<(BitString, f32, Vec<Vec<f32>>), GeneticAlgorithmError> {
    if params.n_pop % 2 != 0 {
        return Err(GeneticAlgorithmError::PopulationMustBeEven)
    }

    let mut pop: Vec<BitString> = (0..params.n_pop)
        .map(|_x| create_random_string(params.n_bits * params.bounds.len()))
        .collect();

    let mut best = pop[0].clone();
    let mut best_eval = match f(&pop[0], &params.bounds, params.n_bits, &settings) {
        Ok(best_eval_result) => best_eval_result,
        Err(e) => return Err(e),
    };

    let mut all_scores = vec![vec![]];

    for gen in 0..params.n_iter {
        if verbose {
            println!("gen: {}", gen + 1);
        }

        let scores_results: &Result<Vec<f32>, GeneticAlgorithmError> = &pop
            .par_iter() 
            .map(|p| f(p, &params.bounds, params.n_bits, &settings))
            .collect(); 
        
        // check if objective failed anywhere
        let scores = match scores_results {
            Ok(scores_results) => scores_results,
            Err(e) => return Err(e.clone()),
        };

        all_scores.push(scores.clone());

        for i in 0..params.n_pop {
            if scores[i] < best_eval {
                best = pop[i].clone();
                best_eval = scores[i];
                if verbose {
                    println!("new string: {}, score: {}", &pop[i].string, &scores[i]);
                }
            }
        }

        let selected: Vec<BitString> = (0..params.n_pop)
            .into_par_iter()
            .map(|_| selection(&pop, &scores, params.k))
            .collect();

        let children = (0..params.n_pop)
            .into_par_iter()
            .step_by(2)
            .flat_map(|i| {
                let new_children = crossover(&selected[i], &selected[i + 1], params.r_cross);
                vec![new_children.0, new_children.1]
            })
            .map(|mut child| {
                mutate(&mut child, params.r_mut);
                child
            })
            .collect();

        pop = children;
    }

    return Ok((best, best_eval, all_scores));
}
