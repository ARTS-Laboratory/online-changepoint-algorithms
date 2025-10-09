use super::em_early_stop_model::{EarlyStopEmModel, LikelihoodChecker};
use super::em_model_builder::BuildError::BadNormalValues;
use super::em_model_builder::FieldStatus::Complete;
use super::normal_params::{NormalParams, NormalParamsError};
use ndarray::{Array1, Array2};
use std::iter::zip;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use super::em_model::EmModel;
use super::normal::Normal;
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

#[derive(Debug)]
pub struct EmBuilderOne<T> {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: FieldStatus<Array1<T>>,
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
    ) -> Result<&mut Self, BuildError<()>> {
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

    /// Set the number of epochs to run.
    ///
    /// # Errors
    ///
    /// If failure occurs while trying to set errors to a given value,
    /// then an error will happen. This is almost exclusively caused by epochs being 0.
    // pub fn build_epochs(&mut self, epochs: u32) -> Result<&mut Self, BuildError<Box<Self>>> {
    pub fn build_epochs(&mut self, epochs: u32) -> Result<&mut Self, BuildError<()>> {
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

    /// Finish current builder changes and return the next builder.
    ///
    /// # Errors
    ///
    /// If sample_arr does not have FieldStatus Complete, then BuildError will be returned.
    pub fn next_builder(&mut self) -> Result<EmBuilderTwo<f64>, BuildError<Box<&mut Self>>> {
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
            Err(BuildError::from(MissingFieldError { my_struct: Box::new(self), field: String::from("sample_arr") }))
        }
    }
}

impl Default for EmBuilderOne<f64> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct EmBuilderTwo<T> {
    normal: NormalParams,
    abnormals: Vec<NormalParams>,
    sample_arr: Array1<T>,
    likelihoods_arr: FieldStatus<Array2<T>>,
    epochs: PositiveInteger,
}

impl<T: Clone + num_traits::identities::Zero + Send + Sync> EmBuilderTwo<T> {

    /// Initialize array for likelihoods
    pub fn build_likelihoods(&mut self) -> &mut Self {
        let sample_size = self.sample_arr.len();
        let num_params = self.abnormals.len() + 1;
        let likelihoods = Array2::<T>::zeros((num_params, sample_size));
        self.likelihoods_arr = Complete(likelihoods);
        self
    }

    /// Finish current builder changes and return the next builder.
    ///
    /// # Errors
    ///
    /// If likelihoods_arr does not have FieldStatus Complete, then BuildError will be returned.
    pub fn next_builder(&mut self) -> Result<EmBuilderLast<T>, BuildError<Box<&mut Self>>> {
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
            Err(BuildError::from(MissingFieldError { my_struct: Box::new(self), field: String::from("likelihoods_arr" )}))
        }
    }
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

impl EmBuilderLast<f64> {
    // pub fn get_model(&self) {
    //     match self.converge_checker {
    //         Some(_) => self.get_early_stop_model(),
    //         None => self.get_standard_model()
    //     }
    // }

    pub fn get_standard_model(&self) -> EmModel {
        let samples = self.sample_arr.clone();
        let abnormals = self.abnormals.clone();
        let likelihoods = self.likelihoods_arr.clone();
        EmModel { normal: self.normal, abnormals, samples, likelihoods, epochs: self.epochs}
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

    // EmBuilderOne tests

    #[test]
    fn test_em_builder_one_build_normal() {
        let mut em = EmBuilderOne::new();
        let result = em.build_normal(-2.0, 10.0, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_em_builder_one_build_normal_fails_bad_mean() {
        let mut em = EmBuilderOne::new();
        let result = em.build_normal(f64::INFINITY, 2.0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_em_builder_one_build_normal_fails_bad_std_dev() {
        let mut em = EmBuilderOne::new();
        let result = em.build_normal(0.0, -2.0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_em_builder_one_build_normal_fails_bad_prob() {
        let mut em = EmBuilderOne::new();
        let result = em.build_normal(0.0, 2.0, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_em_builder_one_build_abnormal() {
        let mut em = EmBuilderOne::new();
        assert!(em.abnormals.is_empty());
        let values = vec![
            NormalParams::new(Normal::new(0.0, 1.0).unwrap(), 0.5).unwrap(),
            NormalParams::new(Normal::new(1.0, 2.0).unwrap(), 0.5).unwrap()];
        em.build_abnormal(&values);
        assert_eq!(em.abnormals.get(0), Some(&values[0]));
        assert_eq!(em.abnormals.get(1), Some(&values[1]));
    }

    #[test]
    fn test_em_builder_one_build_abnormal_from_tuples() {
        let mut em = EmBuilderOne::new();
        assert!(em.abnormals.is_empty());
        let values = vec![
            (0.0, 1.0, 0.5),
            (1.0, 2.0, 0.5)];
        let result = em.build_abnormal_from_tuples(&values);
        assert!(result.is_ok());
        assert_eq!(em.abnormals.get(0), Some(&NormalParams::from_tuple(values[0]).unwrap()));
        assert_eq!(em.abnormals.get(1), Some(&NormalParams::from_tuple(values[1]).unwrap()));
    }

    #[test]
    fn test_em_builder_one_build_epochs() {
        let mut em = EmBuilderOne::new();
        let result = em.build_epochs(10);
        assert!(result.is_ok());
        assert_eq!(em.epochs, PositiveInteger::new(10).unwrap());
        // Failing case
        let result = em.build_epochs(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_samples_from_slice() {
        let mut em = EmBuilderOne::new();
        let samples = vec![0.0, 2.0, -1.0];
        em.build_samples_from_slice(&samples);
        let Complete(new_samples) = em.sample_arr else { panic!("Sample array not initialized") };
        assert_eq!(new_samples, Array1::from_vec(vec![0.0, 2.0, -1.0, 0.0]));
    }

    #[test]
    fn test_em_builder_one_next_builder() {
        let mut em = EmBuilderOne::new();
        let result = em.next_builder();
        assert!(result.is_err());
        let samples = vec![0.0, 2.0, -1.0];
        em.build_samples_from_slice(&samples);
        let result = em.next_builder();
        assert!(result.is_ok());
        let new_samples = result.unwrap().sample_arr;
        assert_eq!(new_samples, Array1::from_vec(vec![0.0, 2.0, -1.0, 0.0]));
    }

    // EmBuilderTwo tests

    fn make_em_builder_two() -> EmBuilderTwo<f64> {
        let mut em = EmBuilderOne::new();
        let samples = vec![0.0, 2.0, -1.0];
        em.build_samples_from_slice(&samples);
        let result = em.next_builder();
        assert!(result.is_ok());
        result.unwrap()
    }

    #[test]
    fn test_em_builder_two_build_likelihoods() {
        let mut em = make_em_builder_two();
        em.build_likelihoods();
        let Complete(new_likelihoods) = em.likelihoods_arr else { panic!("Likelihoods array not initialized") };
        assert_eq!(new_likelihoods.shape(), &[1, 4]);
    }

    #[test]
    fn test_em_builder_two_next_builder() {
        let mut em = make_em_builder_two();
        let result = em.next_builder();
        assert!(result.is_err());
        if let Err(e) = result {
            println!("{:?}", e);
        }
        em.build_likelihoods();
        let result = em.next_builder();
        assert!(result.is_ok());
        let new_likelihoods = result.unwrap().likelihoods_arr;
        assert_eq!(new_likelihoods.shape(), &[1, 4]);
    }

    // EmBuilderLast tests

    fn make_em_builder_last() -> EmBuilderLast<f64> {
        let mut em = EmBuilderOne::new();
        let samples = vec![0.0, 2.0, -1.0];
        em.build_samples_from_slice(&samples);
        let result = em.next_builder();
        assert!(result.is_ok());
        result.unwrap().build_likelihoods().next_builder().unwrap()
    }

    #[test]
    fn test_get_standard_model() {
        let mut em = make_em_builder_last();
        let standard_model = em.get_standard_model();
        assert_eq!(standard_model.normal, em.normal);
        assert_eq!(standard_model.abnormals, em.abnormals);
        assert_eq!(standard_model.samples, em.sample_arr);
        assert_eq!(standard_model.likelihoods, em.likelihoods_arr);
        assert_eq!(standard_model.epochs, em.epochs);
    }

    #[test]
    fn test_get_early_stop_model() {
        let mut em = make_em_builder_last();
        em.build_likelihood_converge_checker();
        let early_stop_model = em.get_early_stop_model();
        let EarlyStopEmModel { em_model, converge_checker } = early_stop_model;
        assert_eq!(em_model.normal, em.normal);
        assert_eq!(em_model.abnormals, em.abnormals);
        assert_eq!(em_model.samples, em.sample_arr);
        assert_eq!(em_model.likelihoods, em.likelihoods_arr);
        assert_eq!(em_model.epochs, em.epochs);
        assert_eq!(converge_checker.prev_likelihood.shape(), em.likelihoods_arr.shape());
    }

    #[test]
    #[should_panic(expected = "Converge checker not initialized")]
    fn test_get_early_stop_model_panics_without_converge_checker() {
        let mut em = make_em_builder_last();
        em.get_early_stop_model();
    }
}
