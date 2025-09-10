use super::beta_cache::BetaCache;
use super::dist_params::DistParams;
use super::sparse_probs::SparseProbs;
use crate::bocpd::NormalInverseGamma;
use pyo3::{pyclass, pymethods, PyResult};
use std::iter::zip;

const DEFAULT_THRESHOLD: f64 = 1e-8;
const BETA_FIXED: f64 = 0.5;

/// A class implementing Bayesian Online Change Point Detection.
#[pyclass]
pub struct BocpdModel {
    initial_params: NormalInverseGamma,
    threshold: f64,
    prev_max: usize,
    curr_max: usize,
    probs: SparseProbs,
    params: DistParams,
    beta_cache: Option<BetaCache>,
}

#[pymethods]
impl BocpdModel {
    #[new]
    pub fn new_py(
        alpha: f64,
        beta: f64,
        mu: f64,
        kappa: f64,
        with_cache: bool,
        threshold: Option<f64>,
    ) -> PyResult<Self> {
        let initial = NormalInverseGamma {
            alpha,
            beta,
            mu,
            kappa,
        };
        let threshold = threshold.unwrap_or(DEFAULT_THRESHOLD);
        let prev_max = 0;
        let curr_max = 0;
        let mut probs = SparseProbs::new_py();
        probs.new_entry(0, 1.0)?;
        let cache = match with_cache {
            true => Some(BetaCache::new_py(BETA_FIXED)),
            false => None,
        };
        let params: DistParams = DistParams::new_py(alpha, beta, mu, kappa)?;
        Ok(Self {
            initial_params: initial,
            threshold,
            prev_max,
            curr_max,
            probs,
            params,
            beta_cache: cache,
        })
    }

    /// Calculate and return likelihood estimates for each distribution in params.
    fn get_priors(&mut self, point: f64) -> Vec<f64> {
        match &mut self.beta_cache {
            Some(x) => self.params.priors_cached(point, x),
            None => self.params.priors(point),
        }
    }

    /// Update model parameters using given input value.
    pub fn update(&mut self, point: f64, lamb: f64) -> PyResult<()> {
        self.calculate_probabilities(point, lamb)?;
        self.truncate_vectors();
        // self.probs.normalize();
        self.update_params(point);
        Ok(())
    }

    /// give probability of seeing input value
    pub fn predict(&mut self, point: f64) -> f64 {
        let priors = self.get_priors(point);
        // dot product
        zip(priors.iter(), self.probs.iter())
            .map(|(change, prob)| change * prob.get_value())
            .sum()
    }

    fn calculate_probabilities(&mut self, point: f64, lamb: f64) -> PyResult<()> {
        let hazard = lamb.recip();
        let priors = self.get_priors(point);
        self.probs.update_probs(priors, hazard)?;
        self.probs.normalize();
        Ok(())
    }

    fn truncate_vectors(&mut self) -> usize {
        let mut threshold_filter: Vec<bool> = self
            .probs
            .iter()
            .map(|prob| prob.get_value() >= self.threshold)
            .collect();
        threshold_filter[0] = true;
        let mut tf_iter = threshold_filter.iter();
        self.params.retain_mut(|_| *tf_iter.next().unwrap());
        let mut tf_iter = threshold_filter.into_iter();
        self.probs.retain_mut(|_| tf_iter.next().unwrap());
        self.params.len()
    }

    /// Update parameters based on new observation.
    fn update_params(&mut self, point: f64) {
        let (max_idx, _max_val) = self.probs.max_prob();
        self.prev_max = self.curr_max;
        self.curr_max = max_idx as usize;
        let NormalInverseGamma {
            alpha,
            beta,
            mu,
            kappa,
        } = self.initial_params;
        if self.curr_max < self.prev_max {
            self.probs.reset();
            self.params
                .reset(alpha, beta, mu, kappa)
                .expect("Initial params should have been validated at construction.");
        } else {
            self.params.update_no_change(point, alpha, beta, mu, kappa);
        }
    }
}
