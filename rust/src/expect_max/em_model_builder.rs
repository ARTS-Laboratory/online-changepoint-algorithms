use super::em_early_stop_model::{EarlyStopEmModel, LikelihoodChecker};
use super::em_model_builder::BuildError::BadNormalValues;
use super::em_model_builder::FieldStatus::Complete;
use super::normal_params::{NormalParams, NormalParamsError};
use ndarray::{Array1, Array2};
use std::iter::zip;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use super::em_model::EmModel;
use super::normal::{InvalidFloatError, Normal};
use super::pos_int::{PositiveError, PositiveInteger};

// trait EmBuild {
//     fn build_normal();
//     fn build_abnormal();
//     fn build_samples();
//     fn build_epochs();
//     fn get_model() -> EmModel;
// }

#[derive(Debug)]
enum FieldStatus<T> {
    Complete(T),
    Incomplete(T),
    NotStarted,
}

#[derive(Debug)]
pub enum BuildError<T: Send + Sync> {
    BadEpoch(PositiveError),
    BadNormalValues(NormalParamsError),
    // FieldConstructionError(T),
    IncompleteBuildError(MissingFieldError<T>),
}

impl<T: Send + Sync> From<BuildError<T>> for PyErr {
    fn from(value: BuildError<T>) -> Self {
        match value {
            BuildError::BadEpoch(e) => e.into(),
            BadNormalValues(e) => e.into(),
            BuildError::IncompleteBuildError(e) => e.into(),
        }
    }
}

// impl<T> From<NormalParamsError> for BuildError<T> {
//     fn from(err: NormalParamsError) -> Self {
//         BuildError::FieldConstructionError(err)
//     }
// }

impl<T: Send + Sync> From<PositiveError> for BuildError<T> {
    fn from(err: PositiveError) -> Self {
        BuildError::BadEpoch(err)
    }
}

impl<T: Send + Sync> From<NormalParamsError> for BuildError<T> {
    fn from(err: NormalParamsError) -> Self {
        BadNormalValues(err)
    }
}

#[derive(Debug)]
pub struct MissingFieldError<T> {
    pub my_struct: T,
    pub field: String,
}

impl<T: Send + Sync> From<MissingFieldError<T>> for BuildError<T> {
    fn from(value: MissingFieldError<T>) -> Self {
        BuildError::IncompleteBuildError(value)
    }
}

impl<T> From<MissingFieldError<T>> for PyErr {
    fn from(err: MissingFieldError<T>) -> PyErr {
        PyValueError::new_err(format!("Tried to build object without completing <{}> field", err.field))
    }
}

// #[pyclass]
pub struct EmBuilder {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: Option<Array1<f64>>,
    // likelihoods_arr: Option<Array2<f64>>,
    epochs: PositiveInteger,
}

// todo make constrained members of classes.
// #[pymethods]
impl EmBuilder {
    // #[new]
    pub fn new() -> Self {
        let normal = NormalParams::new(
            Normal::new(0.0, 1.0).expect("The default values used should never fail"),
            1.0,
        )
        .expect("The default parameters should never fail");
        let abnormals: Vec<NormalParams> = Vec::new();
        let epochs: u32 = 1;
        Self {
            normal,
            abnormals,
            sample_arr: None,
            // likelihoods_arr: None,
            epochs: PositiveInteger::new(epochs).expect("The default value used should never fail"),
        }
    }

    pub fn build_normal(
        &mut self,
        mean: f64,
        stddev: f64,
        prob: f64,
    ) -> Result<&mut EmBuilder, NormalParamsError> {
        self.normal.update_params(mean, stddev, prob)?;
        Ok(self)
    }

    pub fn build_abnormal(
        &mut self,
        abnormals: &[NormalParams],
    ) -> Result<&mut EmBuilder, InvalidFloatError> {
        abnormals.clone_into(&mut self.abnormals);
        Ok(self)
    }

    pub fn build_abnormal_from_tuples(
        &mut self,
        abnormals: &[(f64, f64, f64)],
    ) -> Result<&mut EmBuilder, NormalParamsError> {
        for &(mean, stddev, prob) in abnormals {
            let dist = Normal::new(mean, stddev)?;
            let params = NormalParams::new(dist, prob)?;
            self.abnormals.push(params);
        }
        Ok(self)
    }

    pub fn build_epochs(&mut self, epochs: u32) -> Result<&mut EmBuilder, PositiveError> {
        self.epochs.set(epochs)?;
        Ok(self)
    }

    pub fn build_samples_from_slice(&mut self, samples: &[f64]) -> &mut EmBuilder {
        let mut sample_arr = Array1::zeros(samples.len() + 1);
        for (out, &sample) in zip(&mut sample_arr, samples) {
            *out = sample;
        }
        debug_assert_eq!(samples.len() + 1, sample_arr.len());
        self.sample_arr = Some(sample_arr);
        self
    }
}

// todo deprecate this
impl EmBuilder {
    pub fn get_model(&self) -> EmModel {
        let Some(build_samples) = &self.sample_arr else {
            panic!("Sample array not initialized");
        };
        let samples = build_samples.clone();
        EmModel::new(self.normal, self.abnormals.clone(), samples, self.epochs)
    }
}

#[derive(Debug)]
pub struct EmBuilderOne<T> {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: FieldStatus<Array1<T>>,
    epochs: PositiveInteger,
}

#[derive(Debug)]
pub struct EmBuilderTwo<T> {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: Array1<T>,
    likelihoods_arr: FieldStatus<Array2<T>>,
    epochs: PositiveInteger,
}

#[derive(Debug)]
pub struct EmBuilderLast<T> {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: Array1<T>,
    likelihoods_arr: Array2<T>,
    converge_checker: Option<LikelihoodChecker<f64>>,
    epochs: PositiveInteger,
}

impl EmBuilderOne<f64> {
    // #[new]
    pub fn new() -> Self {
        let normal = NormalParams::new(
            Normal::new(0.0, 1.0).expect("The default values used should never fail"),
            1.0,
        )
        .expect("The default parameters should never fail");
        let abnormals: Vec<NormalParams> = Vec::new();
        let epochs: u32 = 1;
        Self {
            normal,
            abnormals,
            sample_arr: FieldStatus::NotStarted,
            // likelihoods_arr: None,
            epochs: PositiveInteger::new(epochs).expect("The default value used should never fail"),
        }
    }

    pub fn build_normal(
        &mut self,
        mean: f64,
        stddev: f64,
        prob: f64,
    ) -> Result<&mut Self, BuildError<Self>> {
        // self.normal.update_params(mean, stddev, prob)?;
        self.normal.update_params(mean, stddev, prob)?;
        Ok(self)
    }

    pub fn build_abnormal(&mut self, abnormals: &[NormalParams]) -> &mut Self {
        abnormals.clone_into(&mut self.abnormals);
        self
    }

    pub fn build_abnormal_from_tuples(
        &mut self,
        abnormals: &[(f64, f64, f64)],
    ) -> Result<&mut Self, BuildError<&mut Self>> {
        for &(mean, stddev, prob) in abnormals {
            let abnormal = NormalParams::from_tuple((mean, stddev, prob))?;
            self.abnormals.push(abnormal);
        }
        Ok(self)
    }

    pub fn build_epochs(&mut self, epochs: u32) -> Result<&mut Self, BuildError<Self>> {
        self.epochs.set(epochs)?;
        Ok(self)
    }

    pub fn build_samples_from_slice(&mut self, samples: &[f64]) -> &mut Self {
        let mut sample_arr = Array1::zeros(samples.len() + 1);
        for (out, &sample) in zip(&mut sample_arr, samples) {
            *out = sample;
        }
        debug_assert_eq!(samples.len() + 1, sample_arr.len());
        self.sample_arr = Complete(sample_arr);
        self
    }

    pub fn next_builder(&mut self) -> Result<EmBuilderTwo<f64>, BuildError<&mut Self>> {
        if let Complete(sample_arr) = &self.sample_arr {
            let abnormals = self.abnormals.clone();
            let sample_arr = sample_arr.clone();
            Ok(EmBuilderTwo {
                normal: self.normal,
                abnormals,
                sample_arr,
                likelihoods_arr: FieldStatus::NotStarted,
                epochs: self.epochs,
            })
        } else {
            // Err(BuildError::IncompleteBuildError(self.sample_arr))
            Err(BuildError::from(MissingFieldError { my_struct: self, field: String::from("sample_arr") }))
        }
    }
}

impl<T: Clone + num_traits::identities::Zero + Send + Sync> EmBuilderTwo<T> {
    pub fn build_likelihoods(&mut self) -> &mut Self {
        let sample_size = self.sample_arr.len();
        let num_params = self.abnormals.len() + 1;
        let likelihoods = Array2::<T>::zeros((num_params, sample_size));
        self.likelihoods_arr = Complete(likelihoods);
        self
    }

    pub fn next_builder(&mut self) -> Result<EmBuilderLast<T>, BuildError<&mut Self>> {
        if let Complete(likelihoods_arr) = &self.likelihoods_arr {
            let sample_arr: Array1<T> = self.sample_arr.clone();
            Ok(EmBuilderLast {
                normal: self.normal,
                abnormals: self.abnormals.clone(),
                sample_arr,
                likelihoods_arr: likelihoods_arr.clone(),
                epochs: self.epochs,
                converge_checker: None,
            })
        } else {
            // Err(BuildError::IncompleteBuildError(self))
            Err(BuildError::from(MissingFieldError { my_struct: self, field: String::from("likelihoods_arr" )}))
        }
    }
}

impl EmBuilderLast<f64> {
    // pub fn get_model(&self) {
    //     match self.converge_checker {
    //         Some(_) => self.get_early_stop_model(),
    //         None => self.get_standard_model()
    //     }
    // }

    pub fn get_standard_model(&self) -> EmModel {
        let samples = self.sample_arr.clone();
        EmModel::new(self.normal, self.abnormals.clone(), samples, self.epochs)
    }

    pub fn get_early_stop_model(&self) -> EarlyStopEmModel<LikelihoodChecker<f64>> {
        let Some(checker) = &self.converge_checker else {
            panic!("Converge checker not initialized");
        };
        let em_model = self.get_standard_model();
        let converge_checker = checker.clone();
        EarlyStopEmModel {
            em_model,
            converge_checker,
        }
    }

    pub fn build_likelihood_converge_checker(&mut self) -> &mut Self {
        let likelihood_check = Array2::zeros(self.likelihoods_arr.raw_dim());
        self.converge_checker = Some(LikelihoodChecker {
            prev_likelihood: likelihood_check,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_builder_default_constructor() {
        let _em = EmBuilder::new();
        // assert!(em);
        // let NormalParams {dist, prob} = em.normal.;
        // assert_eq!(prob, 1.0);
        // let Normal { mean, stddev } = dist;
        // assert_eq!(mean, 0.0);
        // assert_eq!(stddev, 1.0);
    }

    #[test]
    fn test_em_builder_build_normal() {
        let mut em = EmBuilder::new();
        let result = em.build_normal(-2.0, 10.0, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_em_builder_build_normal_fails() {
        let mut em = EmBuilder::new();
        let result = em.build_normal(-2.0, -10.0, 0.5);
        assert!(result.is_err());
    }

    // #[test]
    // fn test_em_builder_build_abnormal() {
    //
    // }

    #[test]
    fn test_em_builder_build_samples_from_slice() {
        let mut em = EmBuilder::new();
        em.build_samples_from_slice(&[-2.0, 1.0, 1.0]);
        assert_eq!(
            em.sample_arr,
            Some(Array1::from_vec(vec![-2.0, 1.0, 1.0, 0.0]))
        );
    }
}
