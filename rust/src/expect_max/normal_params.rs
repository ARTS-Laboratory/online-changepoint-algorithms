use crate::expect_max::normal::{Normal, NormalError};
use crate::expect_max::probability::{Probability, ProbabilityError};
use ndarray::{ArrayBase, Data, DataMut, Ix1};
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;

#[derive(Debug)]
pub enum NormalParamsError {
    ParameterError(NormalError),
    ProbabilityError(ProbabilityError),
}

impl fmt::Display for NormalParamsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NormalParamsError::ParameterError(ref err) => write!(f, "Parameter error: {}", err),
            NormalParamsError::ProbabilityError(ref err) => write!(f, "Probability error: {}", err),
        }
    }
}

impl From<NormalError> for NormalParamsError {
    fn from(err: NormalError) -> NormalParamsError {
        NormalParamsError::ParameterError(err)
    }
}

impl From<ProbabilityError> for NormalParamsError {
    fn from(err: ProbabilityError) -> NormalParamsError {
        NormalParamsError::ProbabilityError(err)
    }
}

impl From<NormalParamsError> for PyErr {
    fn from(err: NormalParamsError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NormalParams {
    dist: Normal,
    prob: Probability,
}

impl NormalParams {
    pub fn new(dist: Normal, prob_value: f64) -> Result<Self, NormalParamsError> {
        let prob = Probability::new(prob_value)?;
        Ok(Self { dist, prob })
    }

    // Construct from 3-tuple of mean, standard deviation, and probability
    pub fn from_tuple(tuple: (f64, f64, f64)) -> Result<Self, NormalParamsError> {
        let normal = Normal::new(tuple.0, tuple.1)?;
        Ok(Self {
            dist: normal,
            prob: Probability::new(tuple.2)?,
        })
    }

    pub fn likelihood(&self, point: f64) -> f64 {
        self.prob.value() * self.dist.phi(point)
    }

    pub fn probs_inplace(&self, points: &[f64], out: &mut [f64]) {
        for (res, &point) in out.iter_mut().zip(points.iter()) {
            *res = self.likelihood(point);
        }
    }

    pub fn probs_inplace_arr<S, T>(&self, points: &ArrayBase<S, Ix1>, out: &mut ArrayBase<T, Ix1>)
    where
        S: Data<Elem = f64>, // must be f64 to work with phi method
        T: DataMut<Elem = f64>,
    {
        out.zip_mut_with(points, |res, &point| {
            *res = self.likelihood(point);
        });
    }

    pub fn update_params(
        &mut self,
        mean: f64,
        stddev: f64,
        prob: f64,
    ) -> Result<(f64, f64, f64), NormalParamsError> {
        self.dist.update_params(mean, stddev)?;
        self.prob(prob)?;
        Ok((mean, stddev, prob))
    }

    pub fn prob(&mut self, prob: f64) -> Result<f64, NormalParamsError> {
        self.prob.probability(prob)?;
        Ok(prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_params_new_success() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let params = NormalParams::new(normal, 0.5).unwrap();
        assert_eq!(params.dist.mean(), 0.0);
        assert_eq!(params.dist.stddev(), 1.0);
        assert_eq!(params.prob.value(), 0.5);
    }

    #[test]
    fn test_normal_params_new_success_negative_mean() {
        let normal = Normal::new(-1.0, 1.0).unwrap();
        let params = NormalParams::new(normal, 0.5);
        assert!(params.is_ok());
        let params = params.unwrap();
        assert_eq!(params.dist.mean(), -1.0);
        assert_eq!(params.dist.stddev(), 1.0);
        assert_eq!(params.prob.value(), 0.5);
    }

    #[test]
    fn test_normal_params_from_tuple_success() {
        let params = NormalParams::from_tuple((0.0, 1.0, 0.5)).unwrap();
        assert_eq!(params.dist.mean(), 0.0);
        assert_eq!(params.dist.stddev(), 1.0);
        assert_eq!(params.prob.value(), 0.5);
    }

    fn get_params() -> NormalParams {
        NormalParams::from_tuple((0.0, 1.0, 0.5)).unwrap()
    }

    #[test]
    fn test_update_params_success() {
        let mut params = get_params();
        params.update_params(2.0, 3.0, 0.7).unwrap();
        assert_eq!(params.dist.mean(), 2.0);
        assert_eq!(params.dist.stddev(), 3.0);
        assert_eq!(params.prob.value(), 0.7);
    }

    #[test]
    fn test_update_params_failure_all_bad() {
        let mut params = get_params();
        let res = params.update_params(f64::NAN, -2.0, 5.8);
        assert!(res.is_err());
        assert_eq!(params.dist.mean(), 0.0);
        assert_eq!(params.dist.stddev(), 1.0);
        assert_eq!(params.prob.value(), 0.5);
    }
}
