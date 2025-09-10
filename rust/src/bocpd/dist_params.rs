use super::beta_cache::BetaCache;
use crate::bocpd::NormalInverseGamma;
use pyo3::{pyclass, pymethods, PyResult};
use statrs::function::beta::beta;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

#[pyclass]
pub struct DistParams {
    params: VecDeque<NormalInverseGamma>,
}

#[pymethods]
impl DistParams {
    #[new]
    pub fn new_py(alpha: f64, beta: f64, mu: f64, kappa: f64) -> PyResult<Self> {
        let initial_params = NormalInverseGamma {
            alpha,
            beta,
            mu,
            kappa,
        };
        let mut params = VecDeque::new();
        params.push_back(initial_params);
        Ok(Self { params })
    }

    pub fn reset(&mut self, alpha: f64, beta: f64, mu: f64, kappa: f64) -> PyResult<()> {
        self.params.clear();
        self.params.push_back(NormalInverseGamma {
            alpha,
            beta,
            mu,
            kappa,
        });
        Ok(())
    }

    pub fn priors(&self, value: f64) -> Vec<f64> {
        self.params
            .iter()
            .map(|param| {
                let denom = 2.0 * param.beta * (param.kappa + 1.0) / param.kappa;
                let exponent = -(param.alpha + 0.5);
                let t_value = ((value - param.mu).powi(2) / denom + 1.0).powf(exponent);
                t_value / (denom.sqrt() * beta(0.5, param.alpha))
            })
            .collect()
    }

    pub fn priors_cached(&self, value: f64, cache: &mut BetaCache) -> Vec<f64> {
        self.params
            .iter()
            .map(|param| {
                let denom = 2.0 * param.beta * (param.kappa + 1.0) / param.kappa;
                let exponent = -(param.alpha + 0.5);
                let t_value = ((value - param.mu).powi(2) / denom + 1.0).powf(exponent);
                // try to use the cache if it matches, else just calculate normally.
                let beta_value = match cache.get_fixed_value() {
                    0.5 => cache.get_value(param.alpha),
                    _ => beta(0.5, param.alpha),
                };
                t_value / (denom.sqrt() * beta_value)
            })
            .collect()
    }

    pub fn update_no_change(&mut self, value: f64, alpha: f64, beta: f64, mu: f64, kappa: f64) {
        for params in self.params.iter_mut() {
            let kappa_plus = params.kappa + 1.0;
            let new_kappa = kappa_plus;
            let new_alpha = params.alpha + 0.5;
            let new_mu = (params.kappa * params.mu + value) / kappa_plus;
            let new_beta =
                params.beta + params.kappa * (value - params.mu).powi(2) / (2.0 * kappa_plus);
            params.kappa = new_kappa;
            params.alpha = new_alpha;
            params.mu = new_mu;
            params.beta = new_beta;
        }
        self.params.push_front(NormalInverseGamma {
            alpha,
            beta,
            mu,
            kappa,
        });
    }
}

impl Deref for DistParams {
    type Target = VecDeque<NormalInverseGamma>;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl DerefMut for DistParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.params
    }
}
