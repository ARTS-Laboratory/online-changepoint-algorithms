pub mod beta_cache;
pub mod bocpd_model;
pub mod dist_params;
mod element;
pub mod sparse_probs;

use statrs::function::beta::beta;
use std::collections::{HashMap, VecDeque};
use std::iter::zip;

pub struct NormalInverseGamma {
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub kappa: f64,
}

pub fn bocpd<T: element::Element>(
    data: impl IntoIterator<Item = T> + ExactSizeIterator,
    mu: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
    lamb: f64,
) -> Vec<f64> {
    let threshold = 1e-16;
    let mut out: Vec<f64> = Vec::with_capacity(data.len());
    let mut run_lengths: VecDeque<i64> = VecDeque::new();
    let mut probabilities: VecDeque<f64> = VecDeque::new();
    let mut parameters: VecDeque<NormalInverseGamma> = VecDeque::new();
    let initial_params = NormalInverseGamma {
        alpha,
        beta,
        mu,
        kappa,
    };
    let mut prev_max;
    let mut curr_max = -1;
    let mut cache: HashMap<(u64, u64), f64> = HashMap::new();
    // push values
    run_lengths.push_back(0);
    probabilities.push_back(1.0);
    parameters.push_back(initial_params);
    for event in data {
        let event = event.get_data();
        // calculate priors
        // calculate_probabilities(event, lamb, &mut parameters, &mut run_lengths, &mut probabilities);
        calculate_probabilities_2(
            event,
            lamb,
            &mut parameters,
            &mut run_lengths,
            &mut probabilities,
            &mut cache,
        );
        // truncate vectors
        truncate_vectors(
            threshold,
            &mut parameters,
            &mut run_lengths,
            &mut probabilities,
        );
        // check arg max
        let mut max_value = -f64::INFINITY;
        let mut max_idx = 0;
        for (&run_length, &probability) in zip(run_lengths.iter(), probabilities.iter()) {
            if probability > max_value {
                max_idx = run_length;
                max_value = probability;
            }
        }
        (prev_max, curr_max) = (curr_max, max_idx);
        if curr_max < prev_max {
            probabilities.clear();
            probabilities.push_back(1.0);
            run_lengths.clear();
            run_lengths.push_back(0);
            parameters.clear();
            parameters.push_back(NormalInverseGamma {
                alpha,
                beta,
                mu,
                kappa,
            });
        } else {
            update_no_attack(event, &mut parameters, alpha, beta, mu, kappa);
        }
        let change_probs = calculate_priors_cached(event, &parameters, &mut cache);
        // let change_probs = calculate_priors(event, &parameters);
        // let mut val_prob = 0.0;
        // for (change_prob, prob) in zip(change_probs.iter(), probabilities.iter()) {
        //     val_prob += change_prob * prob;
        // }
        let val_prob = zip(change_probs, probabilities.iter())
            .map(|(change, prob)| change * prob)
            .sum();
        // let val_prob = zip(change_probs.iter(), probabilities.iter()).map(|(change, prob)| {
        //     change * prob
        // }).sum();
        out.push(val_prob);
    }
    out
}

fn calculate_probabilities(
    point: f64,
    lamb: f64,
    parameters: &VecDeque<NormalInverseGamma>,
    run_lengths: &mut VecDeque<i64>,
    probabilities: &mut VecDeque<f64>,
) {
    let priors = calculate_priors(point, parameters);
    let mut head = 0.0;
    let hazard = hazard_function(lamb);
    let negative_hazard = 1.0 - hazard;
    for (probability, prior) in zip(probabilities.iter_mut(), priors) {
        let val = *probability * prior;
        head += val;
        *probability = val * negative_hazard;
    }
    head *= hazard;
    probabilities.push_front(head);
    // normalize if not 0
    let prob_sum: f64 = probabilities.iter().sum();
    if prob_sum != 0.0 {
        for probability in probabilities.iter_mut() {
            *probability /= prob_sum;
        }
    }
    for run_length in run_lengths.iter_mut() {
        *run_length += 1;
    }
    run_lengths.push_front(0);
}

fn calculate_probabilities_2(
    point: f64,
    lamb: f64,
    parameters: &VecDeque<NormalInverseGamma>,
    run_lengths: &mut VecDeque<i64>,
    probabilities: &mut VecDeque<f64>,
    cache: &mut HashMap<(u64, u64), f64>,
) {
    // let priors = calculate_priors(point, &parameters);
    let priors = calculate_priors_cached(point, parameters, cache);
    // let priors = match cache {
    //     Some(cache) => calculate_priors_cached(point, &parameters, cache),
    //     None => calculate_priors(point, &parameters),
    // };
    let mut head = 0.0;
    let hazard = hazard_function(lamb);
    let negative_hazard = 1.0 - hazard;
    for (probability, prior) in zip(probabilities.iter_mut(), priors) {
        let val = *probability * prior;
        head += val;
        *probability = val * negative_hazard;
    }
    head *= hazard;
    probabilities.push_front(head);
    // normalize if not 0
    let prob_sum: f64 = probabilities.iter().sum();
    if prob_sum != 0.0 {
        for probability in probabilities.iter_mut() {
            *probability /= prob_sum;
        }
    }
    for run_length in run_lengths.iter_mut() {
        *run_length += 1;
    }
    run_lengths.push_front(0);
}
pub fn truncate_vectors(
    threshold: f64,
    parameters: &mut VecDeque<NormalInverseGamma>,
    run_lengths: &mut VecDeque<i64>,
    probabilities: &mut VecDeque<f64>,
) {
    let threshold_filter: Vec<bool> = probabilities
        .iter()
        .map(|&probability| probability >= threshold)
        .collect();
    let mut tf_iter = threshold_filter.iter();
    parameters.retain_mut(|_| *tf_iter.next().unwrap());
    let mut tf_iter = threshold_filter.iter();
    run_lengths.retain_mut(|_| *tf_iter.next().unwrap());
    let mut tf_iter = threshold_filter.into_iter();
    probabilities.retain_mut(|_| tf_iter.next().unwrap());
    // for (idx, probability) in probabilities.clone().iter().enumerate() {
    //     if *probability < threshold {
    //         parameters.remove(idx);
    //         run_lengths.remove(idx);
    //         probabilities.remove(idx);
    //     }
    // }
}

fn update_no_attack(
    point: f64,
    parameters: &mut VecDeque<NormalInverseGamma>,
    alpha: f64,
    beta: f64,
    mu: f64,
    kappa: f64,
) {
    for params in parameters.iter_mut() {
        let kappa_plus = params.kappa + 1.0;
        let new_kappa = kappa_plus;
        let new_alpha = params.alpha + 0.5;
        let new_mu = (params.kappa * params.mu + point) / kappa_plus;
        let new_beta =
            params.beta + params.kappa * (point - params.mu).powi(2) / (2.0 * kappa_plus);
        params.kappa = new_kappa;
        params.alpha = new_alpha;
        params.mu = new_mu;
        params.beta = new_beta;
    }
    parameters.push_front(NormalInverseGamma {
        alpha,
        beta,
        mu,
        kappa,
    });
}

fn calculate_priors(point: f64, parameters: &VecDeque<NormalInverseGamma>) -> Vec<f64> {
    // let mut out: Vec<f64> = Vec::with_capacity(parameters.len());
    // for params in parameters.iter() {
    //     let denom = 2.0 * params.beta * (params.kappa + 1.0) / params.kappa;
    //     let exponent = -(params.alpha + 0.5);
    //     let t_value = ((point - params.mu).powi(2) / denom + 1.0).powf(exponent);
    //     let result = t_value / (denom.sqrt() * beta(0.5, params.alpha));
    //     out.push(result);
    // }
    let out = parameters
        .iter()
        .map(|params| {
            let denom = 2.0 * params.beta * (params.kappa + 1.0) / params.kappa;
            let exponent = -(params.alpha + 0.5);
            let t_value = ((point - params.mu).powi(2) / denom + 1.0).powf(exponent);
            t_value / (denom.sqrt() * beta(0.5, params.alpha))
        })
        .collect();
    out
}

fn calculate_priors_cached<'a>(
    point: f64,
    parameters: &'a VecDeque<NormalInverseGamma>,
    cache: &'a mut HashMap<(u64, u64), f64>,
) -> impl IntoIterator<Item = f64> + 'a {
    // let out = parameters.iter().map(|params| {
    //     let denom = 2.0 * params.beta * (params.kappa + 1.0) / params.kappa;
    //     let exponent = -(params.alpha + 0.5);
    //     let t_value = ((point - params.mu).powi(2) / denom + 1.0).powf(exponent);
    //     let result = t_value / (denom.sqrt() * get_beta(0.5, params.alpha, cache));
    //     result
    // }).collect();
    // out
    let out = parameters.iter().map(move |params| {
        let denom = 2.0 * params.beta * (params.kappa + 1.0) / params.kappa;
        let exponent = -(params.alpha + 0.5);
        let t_value = ((point - params.mu).powi(2) / denom + 1.0).powf(exponent);
        t_value / (denom.sqrt() * get_beta(0.5, params.alpha, cache))
    });
    out.into_iter()
}

// tries to calculate beta using Beta(x, y + 1) = Beta(x, y) * y / (x + y)
pub fn get_beta(steady_x: f64, increase_y: f64, cache: &mut HashMap<(u64, u64), f64>) -> f64 {
    let key = (steady_x.to_bits(), increase_y.to_bits());
    if cache.contains_key(&key) {
        let value = cache
            .get(&key)
            .expect("Cache should always have key here since we checked just before.");
        *value
    } else {
        // base case
        let res = match increase_y {
            ..0.0 => todo!("does not expect negative numbers. What should we do in this case?"), // beta(steady_x, increase_y), // technically, should throw error
            0.0..=1.0 => beta(steady_x, increase_y),
            1.0.. => {
                (increase_y / (steady_x + increase_y)) * get_beta(steady_x, increase_y - 1.0, cache)
            }
            _ => todo!("add functionality for other float types."),
        };
        cache.insert(key, res);
        res
    }
}

fn hazard_function(lambda: f64) -> f64 {
    lambda.recip() // 1.0 / x
}
