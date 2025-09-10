use std::iter::{once, zip};
// use bocpd::beta_cache::BetaCache;
use bocpd::bocpd_model::BocpdModel;
// use bocpd::dist_params::DistParams;
// use bocpd::sparse_probs::{SparseProb, SparseProbs};
use cusum::{CusumV0, CusumV1};
use expect_max::em_early_stop_model::EmLikelihoodCheck;
use expect_max::em_model::EmModel;
// use expect_max::em_model_builder::EmBuilder;
use expect_max::em_model_builder::EmBuilderOne;

use pyo3::prelude::*;
use rand::distr::Distribution;

pub mod bocpd;
pub mod cusum;
pub mod expect_max;

// /// Updates the probability distribution for a set of T-distributions with observed point.
// #[pyfunction]
// fn calc_probabilities(
//     point: f64,
//     lamb: f64,
//     params: &DistParams,
//     probs: &mut SparseProbs,
// ) -> PyResult<()> {
//     let hazard = lamb.recip();
//     let priors = params.priors(point);
//     probs.update_probs(priors, hazard)?;
//     probs.normalize();
//     Ok(())
// }
//
// /// Updates the probability distribution for a set of T-distributions with
// /// observed point and cache for beta.
// #[pyfunction]
// fn calc_probabilities_cached(
//     point: f64,
//     lamb: f64,
//     params: &DistParams,
//     probs: &mut SparseProbs,
//     cache: &mut BetaCache,
// ) -> PyResult<()> {
//     let hazard = lamb.recip();
//     let priors = params.priors_cached(point, cache);
//     probs.update_probs(priors, hazard)?;
//     probs.normalize();
//     Ok(())
// }
//
// /// Truncate vectors
// #[pyfunction]
// fn truncate_vectors(
//     threshold: f64,
//     params: &mut DistParams,
//     probs: &mut SparseProbs,
// ) -> usize {
//     let threshold_filter: Vec<bool> = probs
//         .iter()
//         .map(|prob| prob.get_value() >= threshold)
//         .collect();
//
//     let mut tf_iter = threshold_filter.iter();
//     params.retain_mut(|_| *tf_iter.next().unwrap());
//     let mut tf_iter = threshold_filter.into_iter();
//     probs.retain_mut(|_| tf_iter.next().unwrap());
//     params.len()
// }
//
// #[pyfunction]
// fn get_change_prob(priors: Vec<f64>, probs: &SparseProbs) -> f64 {
//     zip(priors.iter(), probs.iter())
//         .map(|(change, prob)| change * prob.get_value())
//         .sum()
// }

/// Use builder to construct expectation maximization model.
#[pyfunction]
fn build_em_model(
    normal: (f64, f64, f64),
    abnormals: Vec<(f64, f64, f64)>,
    arr_sizes: Vec<u32>,
    epochs: u32,
) -> PyResult<EmModel> {
    let (mean, stddev, prob) = normal;
    let normal_iter = once(&normal);
    let param_iter = normal_iter.chain(abnormals.iter());
    let samples: Vec<f64> = zip(param_iter, arr_sizes.iter())
        .map(|(params, &size)| {
            let (mean, stddev, _probs) = params;
            let rng = rand::rng();
            let n = rand_distr::Normal::new(*mean, *stddev).expect("please don't panic");
            n.sample_iter(rng).take(size as usize)
        })
        .flatten()
        .collect();
    // let mut em_builder = EmBuilder::new();
    // em_builder
    //     .build_normal(mean, stddev, prob)?
    //     .build_abnormal_from_tuples(&abnormals)?
    //     .build_epochs(epochs)?
    //     .build_samples_from_slice(&samples);
    // Ok(em_builder.get_model())
    let mut em_builder = EmBuilderOne::new();
    let final_builder = em_builder.build_normal(mean, stddev, prob)?
        .build_abnormal_from_tuples(&abnormals)?
        .build_epochs(epochs)?
        .build_samples_from_slice(&samples)
        .next_builder()?
        .build_likelihoods()
        .next_builder()?;
    Ok(final_builder.get_standard_model())
}

#[pyfunction]
fn build_em_early_stop_model(
    normal: (f64, f64, f64),
    abnormals: Vec<(f64, f64, f64)>,
    arr_sizes: Vec<u32>,
    epochs: u32,
) -> PyResult<EmLikelihoodCheck> {
    let (mean, stddev, prob) = normal;
    let normal_iter = once(&normal);
    let param_iter = normal_iter.chain(abnormals.iter());
    let samples: Vec<f64> = zip(param_iter, arr_sizes.iter())
        .map(|(params, &size)| {
            let (mean, stddev, _probs) = params;
            let rng = rand::rng();
            let n = rand_distr::Normal::new(*mean, *stddev).expect("please don't panic");
            n.sample_iter(rng).take(size as usize)
        })
        .flatten()
        .collect();
    let mut em_builder = EmBuilderOne::new();
    let mut final_builder = em_builder.build_normal(mean, stddev, prob)?
        .build_abnormal_from_tuples(&abnormals)?
        .build_epochs(epochs)?
        .build_samples_from_slice(&samples)
        .next_builder()?
        .build_likelihoods()
        .next_builder()?;
    final_builder.build_likelihood_converge_checker();
    let wrapped_model = EmLikelihoodCheck::from_early_stop_model(final_builder.get_early_stop_model());
    Ok(wrapped_model)
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_change_point_algorithms")]
fn change_point_algorithms(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_em_model, m)?)?;
    m.add_function(wrap_pyfunction!(build_em_early_stop_model, m)?)?;
    m.add_class::<BocpdModel>()?;
    m.add_class::<EmModel>()?;
    m.add_class::<EmLikelihoodCheck>()?;
    m.add_class::<CusumV0>()?;
    m.add_class::<CusumV1>()?;
    Ok(())
}
