pub mod em_early_stop_model;
pub mod em_model;
pub mod em_model_builder;

mod normal;
mod normal_params;
mod pos_int;
mod probability;

use std::{f64::consts::PI, iter::zip};
// use crate::element::Element;
// struct Normal {
//     mu: f64,
//     var: f64
// }

// pub fn expectation_maximization<'a, T: 'a + Element>(
//     safe: &[f64], not_safe: &[f64], unknowns: impl IntoIterator<Item=T> + ExactSizeIterator, mean_1: f64,
//      mean_2: f64, var_1: f64, var_2: f64, pi: f64, epochs: i64) -> Vec<f64> {
pub fn expectation_maximization<'a>(
    safe: &[f64],
    not_safe: &[f64],
    unknowns: impl IntoIterator<Item = &'a f64> + ExactSizeIterator,
    mean_1: f64,
    mean_2: f64,
    var_1: f64,
    var_2: f64,
    pi: f64,
    epochs: i64,
) -> Vec<f64> {
    let mut data = [safe, not_safe].concat();
    let len = safe.len() + not_safe.len() + 1;
    let i_len = len as i64;
    let converge_tolerance = 1e-4;
    let mut mu1_hat = mean_1;
    let mut mu2_hat = mean_2;
    let mut sig1_hat = var_1;
    let mut sig2_hat = var_2;
    let mut pi_hat = pi;
    let mut prev_out = vec![f64::NEG_INFINITY; len];
    let mut out_1 = vec![0.0; len];
    let mut out_2 = vec![0.0; len];
    let mut out: Vec<f64> = Vec::with_capacity(unknowns.len());
    for &unknown in unknowns {
        // let value = unknown;
        // let value = unknown.get_data();
        // let value = match unknown {
        //     DataLike::Rs64(item) => item,
        //     DataLike::Py(item) => item.extract().expect(""),
        // };
        // data.push(value);
        data.push(unknown);
        for _ in 0..epochs {
            posterior_probs_inplace(
                &data, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat, &mut out_1, &mut out_2,
            );
            // (prev_mu1_hat, prev_mu2_hat, prev_sig1_hat, prev_sig2_hat, prev_pi_hat) = (mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat);
            (mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat) = maximization(
                &data, &out_1, &out_2, mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat, i_len,
            );
            if close_enough(converge_tolerance, &prev_out, &out_1) {
                break;
            }
            prev_out = out_1.clone();
            // if close_enough(
            //     prev_mu1_hat, prev_mu2_hat, prev_sig1_hat, prev_sig2_hat,
            //     prev_pi_hat, mu1_hat, mu2_hat, sig1_hat, sig2_hat, pi_hat,
            //     converge_tolerance) {
            //     break;
            // }
        }
        let prob = posterior_prob(unknown, pi_hat, mu2_hat, sig2_hat, mu1_hat, sig1_hat);
        out.push(prob);
        data.pop();
    }
    out
}

// pub fn expectation_maximization_obj(safe: &[f64], not_safe: &[f64], unknowns: &[f64], mean_1: f64,
//                                     mean_2: f64, var_1: f64, var_2: f64, pi: f64, epochs: i64, convergence_tolerance: Option<f64>) -> Vec<f64> {
//     let mut data = [safe, not_safe].concat();
//     let len = safe.len() + not_safe.len() + 1;
//     let i_len = len as i64;
//     let normal = Normal { mu: mean_1, var: var_1 };
//     let change = Normal { mu: mean_2, var: var_2 };
//     let mut pi_hat = pi;
//     let mut prev_out = vec![f64::NEG_INFINITY; len];
//     let mut out_1 = vec![0.0; len];
//     let mut out_2 = vec![0.0; len];
//     let mut out: Vec<f64> = Vec::with_capacity(unknowns.len());
//     for unknown in unknowns.iter() {
//         data.push(*unknown);
//         for _ in 0..epochs {
//             posterior_probs_inplace_normal_obj(
//                 &data, pi_hat, &change, &normal, &mut out_1, &mut out_2);
//         }
//
//     }
//     out
// }

fn close_enough(tol: f64, prev_probs: &[f64], probs: &[f64]) -> bool {
    zip(prev_probs.iter(), probs.iter()).all(|(prev, prob)| (prev - prob).abs() < tol)
}

// fn close_enough(p0: f64, p1: f64, p2: f64, p3: f64, p4: f64, p5: f64, p6: f64, p7: f64, p8: f64, p9: f64, tol: f64) -> bool {
//     if (p0 - p5).abs() < tol && (p1 - p6).abs() < tol &&
//         (p2 - p7).abs() < tol && (p3 - p8).abs() < tol &&
//         (p4 - p8).abs() < tol && (p5 - p9).abs() < tol {
//         true
//     } else {
//         false
//     }
// }

fn maximization(
    points: &[f64],
    probs: &[f64],
    inverse: &[f64],
    attack_mean: f64,
    attack_var: f64,
    normal_mean: f64,
    normal_var: f64,
    pi_hat: f64,
    size: i64,
) -> (f64, f64, f64, f64, f64) {
    let density: f64 = probs.iter().sum();
    let inverse_density: f64 = inverse.iter().sum();
    if density == 0.0 || inverse_density == 0.0 {
        (attack_mean, normal_mean, attack_var, normal_var, pi_hat)
    } else {
        let (attack_mean_prime, normal_mean_prime) =
            update_means(probs, inverse, density, inverse_density, points);
        let (attack_var_prime, normal_var_prime) = update_variances(
            probs,
            inverse,
            density,
            inverse_density,
            points,
            attack_mean,
            normal_mean,
        );
        let pi_prime = update_attack_prob(density, size);
        (
            attack_mean_prime,
            normal_mean_prime,
            attack_var_prime,
            normal_var_prime,
            pi_prime,
        )
    }
}

fn posterior_probs_inplace(
    points: &[f64],
    attack_prob: f64,
    attack_mean: f64,
    attack_var: f64,
    normal_mean: f64,
    normal_var: f64,
    out_1: &mut [f64],
    out_2: &mut [f64],
) {
    phi_inplace(points, attack_mean, attack_var, out_1);
    for value in out_1.iter_mut() {
        *value *= attack_prob;
    }
    phi_inplace(points, normal_mean, normal_var, out_2);
    let neg_val = 1.0 - attack_prob;
    for value in out_2.iter_mut() {
        *value *= neg_val;
    }
    for (item_1, item_2) in zip(out_1.iter_mut(), out_2.iter_mut()) {
        let denom = *item_1 + *item_2;
        if denom != 0.0 {
            *item_1 /= denom;
            *item_2 /= denom;
        }
    }
}

// fn posterior_probs_inplace_normal_obj(points: &[f64], attack_prob: f64, attack_params: &Normal, normal_params: &Normal, out_1: &mut [f64], out_2: &mut [f64]) {
//     phi_inplace(points, attack_params.mu, attack_params.var, out_1);
//     for value in &mut *out_1 {
//         *value *= attack_prob;
//     }
//     phi_inplace(points, normal_params.mu, normal_params.var, out_2);
//     let neg_val = 1.0 - attack_prob;
//     for value in &mut *out_2 {
//         *value *= neg_val;
//     }
//     for (item_1, item_2) in zip(out_1, out_2) {
//         let denom = *item_1 + *item_2;
//         if denom != 0.0 {
//             *item_1 /= denom;
//             *item_2 /= denom;
//         }
//     }
// }

fn posterior_prob(
    point: f64,
    attack_prob: f64,
    attack_mean: f64,
    attack_var: f64,
    normal_mean: f64,
    normal_var: f64,
) -> f64 {
    let num = attack_prob * phi_single(point, attack_mean, attack_var);
    let denom = num + (1.0 - attack_prob) * phi_single(point, normal_mean, normal_var);
    match denom {
        0.0 => num,
        _ => num / denom,
    }
}

fn phi_inplace(values: &[f64], mean: f64, variance: f64, out: &mut [f64]) {
    if variance == 0.0 {
        for (out_val, item) in zip(out, values) {
            if *item == mean {
                *out_val = 1.0;
            } else {
                *out_val = 0.0;
            }
        }
    } else {
        let scalar: f64 = 1.0 / (2.0 * variance).sqrt();
        for (out_val, item) in zip(out, values) {
            let val = scalar * -((item - mean).powi(2)) / (2.0 * variance).exp();
            *out_val = val;
        }
    }
}

fn phi_single(value: f64, mean: f64, variance: f64) -> f64 {
    if variance == 0.0 {
        if value == mean {
            1.0
        } else {
            0.0
        }
    } else {
        let denom = (variance * 2.0 * PI).sqrt();
        let ex = (-0.5 * (value - mean).powi(2) / variance).exp();
        ex / denom
    }
}

fn update_means(
    probs: &[f64],
    inverse: &[f64],
    density: f64,
    inverse_density: f64,
    events: &[f64],
) -> (f64, f64) {
    let mean_1 = dot_product(inverse, events) / inverse_density;
    let mean_2 = dot_product(probs, events) / density;
    (mean_1, mean_2)
}

fn dot_product(arr_1: &[f64], arr_2: &[f64]) -> f64 {
    zip(arr_1.iter(), arr_2.iter()).map(|(a, b)| a * b).sum()
}

fn variance_helper(probs: &[f64], events: &[f64], mean: f64) -> f64 {
    zip(probs.iter(), events.iter())
        .map(|(a, b)| a * (b - mean).powi(2))
        .sum()
}

fn update_variances(
    probs: &[f64],
    inverse: &[f64],
    density: f64,
    inverse_density: f64,
    events: &[f64],
    mean_1: f64,
    mean_2: f64,
) -> (f64, f64) {
    let var_1 = variance_helper(inverse, events, mean_1) / inverse_density;
    let var_2 = variance_helper(probs, events, mean_2) / density;
    (var_1, var_2)
}

fn update_attack_prob(density: f64, size: i64) -> f64 {
    let float_size = size as f64;
    density / float_size
}
