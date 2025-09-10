use super::em_model::EmModel;
use super::normal_params::NormalParamsError;
use ndarray::{Array2, ArrayView2};
use pyo3::{pyclass, pymethods};

/// Trait for any struct that checks if em model has converged
pub trait HasConverged<T> {
    fn update_checker(&mut self, model: &EmModel);
    fn has_converged(&self, model: &EmModel, threshold: T) -> bool;
}

// #[derive(Clone, Debug)]
// #[non_exhaustive]
// pub(super) enum ConvergenceCheckKind {
//     Likelihood(LikelihoodChecker<f64>),
// }

// impl ConvergenceCheckKind {
//     fn update_checker(&mut self, model: &EmModel) {
//         match self {
//             ConvergenceCheckKind::Likelihood(checker) => checker.update_prev_likelihood(&model),
//             // _ => unimplemented!("Convergence update check not implemented yet"),
//         }
//     }
// }
//
// impl HasConverged<f64> for ConvergenceCheckKind {
//     fn has_converged(&self, model: &EmModel, threshold: f64) -> bool {
//         match self {
//             ConvergenceCheckKind::Likelihood(checker) => checker.has_converged(model, threshold),
//             // _ => unimplemented!("Convergence type not implemented yet."),
//         }
//     }
// }

/// Expectation Maximization model that incorporates a check for early stopping.
#[derive(Clone)]
pub struct EarlyStopEmModel<T: HasConverged<f64>> {
    pub(super) em_model: EmModel,
    pub(super) converge_checker: T,
}

impl<T: HasConverged<f64>> EarlyStopEmModel<T> {
    pub fn update_check_convergence(
        &mut self,
        point: f64,
        threshold: f64,
    ) -> Result<(), NormalParamsError> {
        self.em_model.swap_last_sample(point);
        for _ in 0..self.em_model.epochs().value() {
            self.converge_checker.update_checker(&self.em_model);
            self.em_model.expectation();
            if self.has_converged(threshold) {
                break;
            }
            self.em_model.maximization()?;
        }
        Ok(())
    }

    pub fn has_converged(&self, threshold: f64) -> bool {
        self.converge_checker
            .has_converged(&self.em_model, threshold)
    }
}

#[derive(Clone, Debug)]
pub struct LikelihoodChecker<T> {
    pub(super) prev_likelihood: Array2<T>,
}

impl LikelihoodChecker<f64> {
    // pub fn has_converged(&self, em: &EmModel, threshold: f64) -> bool {
    //     let diffs = em.likelihoods() - self.prev_likelihood();
    //     diffs.iter().all(|&diff| diff.abs() <= threshold)
    // }

    pub fn update_prev_likelihood(&mut self, em: &EmModel) {
        em.likelihoods().clone_into(&mut self.prev_likelihood);
    }

    pub fn prev_likelihood(&self) -> &Array2<f64> {
        &self.prev_likelihood
    }

    pub fn prev_likelihood_view(&self) -> ArrayView2<'_, f64> {
        self.prev_likelihood.view()
    }
}

impl HasConverged<f64> for LikelihoodChecker<f64> {
    fn update_checker(&mut self, em: &EmModel) {
        self.update_prev_likelihood(em);
    }

    fn has_converged(&self, em: &EmModel, threshold: f64) -> bool {
        let diffs = em.likelihoods() - self.prev_likelihood();
        diffs.iter().all(|&diff| diff.abs() <= threshold)
    }
}

// Now we add a macro so we can use this in a concrete way
macro_rules! create_interface {
    ($name: ident, $type: ty) => {
        #[pyclass]
        pub struct $name {
            inner: EarlyStopEmModel<$type>,
        }

        #[pymethods]
        impl $name {
            pub fn update_check_convergence(
        &mut self,
        point: f64,
        threshold: f64,
        ) -> Result<(), NormalParamsError> { self.inner.update_check_convergence(point, threshold) }

            pub fn predict(&self, point: f64) -> f64 { self.inner.em_model.predict(point) }
        }

        impl $name {
            pub fn from_early_stop_model(early_stop_em_model: EarlyStopEmModel<$type>) -> Self {
                Self { inner: early_stop_em_model }
            }

            pub fn from_model_and_checker(model: EmModel, checker: $type) -> Self {
                Self { inner: EarlyStopEmModel { em_model: model, converge_checker: checker }}
            }
        }
    };
}
create_interface!(EmLikelihoodCheck, LikelihoodChecker<f64>);
