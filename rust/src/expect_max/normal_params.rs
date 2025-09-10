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
