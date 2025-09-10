use super::pos_int::PositiveInteger;
use crate::expect_max::normal_params::{NormalParams, NormalParamsError};
use itertools::izip;
use ndarray::{s, Array1, Array2, ArrayView2, Axis, Zip};
use pyo3::{pyclass, pymethods};
use std::iter::zip;

#[derive(Debug)]
pub enum EmModelError {
    PositiveError(u32),
    InvalidFloatError(f64),
    ProbabilityError(f64),
}

// impl fmt::Display for EmModelError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match *self {
//             EmModelError::PositiveError(x) => {}
//         }
//     }
// }

// impl From<PositiveError> for EmModelError {
//     fn from(err: PositiveError) -> EmModelError {
//         EmModelError::PositiveError(err)
//     }
// }

#[pyclass]
#[derive(Clone)]
pub struct EmModel {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    samples: Array1<f64>,
    likelihoods: Array2<f64>,
    epochs: PositiveInteger,
}

#[pymethods]
impl EmModel {
    // #[new]

    pub fn update(&mut self, point: f64) -> Result<(), NormalParamsError> {
        self.swap_last_sample(point);
        for _ in 0..self.epochs.value() {
            self.expectation();
            self.maximization()?;
        }
        Ok(())
    }

    pub fn predict(&self, point: f64) -> f64 {
        self.posterior_prob(point)
    }

    pub fn expectation(&mut self) {
        // raw probabilities
        let sample_view = self.samples.view();
        let mut normal_view = self.likelihoods.row_mut(0);
        self.normal
            .probs_inplace_arr(&sample_view, &mut normal_view);
        let mut abnormals = self.likelihoods.slice_mut(s![1.., ..]);
        let abnormals_view = abnormals.rows_mut();
        for (mut likelihood, abnormal) in zip(abnormals_view, self.abnormals.iter()) {
            abnormal.probs_inplace_arr(&sample_view, &mut likelihood);
        }
        // normalize
        let norms = self.likelihoods.sum_axis(Axis(0));
        let likelihood_view = self.likelihoods.columns_mut();
        for (mut likelihood, &norm) in zip(likelihood_view, norms.iter()) {
            if norm != 0.0 {
                likelihood /= norm;
            }
        }
    }

    pub fn maximization(&mut self) -> Result<(), NormalParamsError> {
        let densities = self.likelihoods.sum_axis(Axis(1));
        // if any densities are zero, leave
        if densities.iter().all(|&density| density.is_normal()) {
            // Get new parameter estimates
            let means = self.update_means(&densities);
            let variances = self.update_variances(&densities, &means);
            let size = self.samples.len();
            let weights = self.update_weights(&densities, size);
            // update parameters
            self.normal
                .update_params(means[0], variances[0].sqrt(), weights[0])?;
            let param_iter = izip!(&means, &variances, &weights).skip(1);
            for (param, (&mean, &variance, &weight)) in zip(&mut self.abnormals, param_iter) {
                param.update_params(mean, variance.sqrt(), weight)?;
            }
        }
        // else
        Ok(())
    }

    fn posterior_prob(&self, point: f64) -> f64 {
        let num: f64 = self.normal.likelihood(point);
        let denom: f64 = num
            + self
                .abnormals
                .iter()
                .map(|param| param.likelihood(point))
                .sum::<f64>();
        match denom {
            0.0 => num,
            _ => num / denom,
        }
    }

    // todo make this an error
    pub fn swap_last_sample(&mut self, point: f64) -> f64 {
        if let Some(last) = self.samples.last_mut() {
            let out = *last;
            *last = point;
            out
        } else {
            panic!("samples is empty");
        }
    }
}

impl EmModel {
    pub fn new(
        normal: NormalParams,
        abnormals: Vec<NormalParams>,
        samples: Array1<f64>,
        epochs: PositiveInteger,
    ) -> Self {
        let sample_size = samples.len();
        let num_params = abnormals.len() + 1;
        let likelihoods = Array2::<f64>::zeros((num_params, sample_size));
        // let last_likelihoods = Array2::from(&likelihoods);
        Self {
            normal,
            abnormals,
            samples,
            likelihoods,
            epochs,
        }
    }

    pub fn epochs(&self) -> PositiveInteger {
        self.epochs
    }

    pub fn likelihoods(&self) -> &Array2<f64> {
        &self.likelihoods
    }

    pub fn likelihoods_view(&self) -> ArrayView2<'_, f64> {
        self.likelihoods.view()
    }

    fn update_means(&self, densities: &Array1<f64>) -> Array1<f64> {
        let sample_view = self.samples.view();
        // let means = (self.likelihoods * densities).sum_axis(Axis(1));
        let means: Array1<f64> = self.likelihoods.map_axis(
            Axis(1),
            |row| row.dot(&sample_view), /*dot_product(&row, &sample_view)*/
        ) / densities;
        means
    }

    fn update_variances(&self, densities: &Array1<f64>, means: &Array1<f64>) -> Array1<f64> {
        let sample_view = self.samples.view();
        let means_view = means.view();
        let variances = Zip::from(self.likelihoods.rows())
            .and(&means_view)
            .map_collect(|row, &mean| {
                let value = (&sample_view - mean).powi(2);
                row.dot(&value)
            })
            / densities;
        variances
    }

    fn update_weights(&self, densities: &Array1<f64>, size: usize) -> Array1<f64> {
        densities / (size as f64)
    }
}

// fn dot_product<A, S, T>(a: &ArrayBase<S, Ix1>, b: &ArrayBase<T, Ix1>) -> A
// where S: Data<Elem = A>,
//     T: Data<Elem = A>,
//     A: NdFloat {
//     (a * b).sum()
// }
