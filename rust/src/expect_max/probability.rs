use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;

#[derive(Debug)]
pub struct ProbabilityError(f64);

impl fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Input {} is not a valid probability. Must be between 0 and 1!",
            self.0
        )
    }
}

impl From<ProbabilityError> for PyErr {
    fn from(err: ProbabilityError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Probability {
    probability: f64,
}

impl Probability {
    pub fn new(probability: f64) -> Result<Probability, ProbabilityError> {
        if is_valid_probability(probability) {
            Ok(Probability { probability })
        } else {
            Err(ProbabilityError(probability))
        }
    }

    pub fn value(&self) -> f64 {
        self.probability
    }

    pub fn probability(&mut self, value: f64) -> Result<f64, ProbabilityError> {
        if is_valid_probability(value) {
            self.probability = value;
            Ok(value)
        } else {
            Err(ProbabilityError(value))
        }
    }
}

fn is_valid_probability(probability: f64) -> bool {
    (0.0..=1.0).contains(&probability)
    // probability >= 0.0 && probability <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability() {
        let prob = Probability { probability: 0.5 };
        assert_eq!(prob.value(), 0.5);
    }

    #[test]
    fn test_probability_setter() {
        let mut prob = Probability { probability: 0.5 };
        assert_eq!(prob.value(), 0.5);
        let success = prob.probability(0.0);
        assert!(success.is_ok());
        assert_eq!(prob.value(), 0.0);
        let success = prob.probability(1.0);
        assert!(success.is_ok());
        assert_eq!(prob.value(), 1.0);
        let failure = prob.probability(-0.1);
        assert!(failure.is_err());
        let failure = prob.probability(1.1);
        assert!(failure.is_err());
        let failure = prob.probability(f64::NAN);
        assert!(failure.is_err());
        let failure = prob.probability(f64::INFINITY);
        assert!(failure.is_err());
    }
}
