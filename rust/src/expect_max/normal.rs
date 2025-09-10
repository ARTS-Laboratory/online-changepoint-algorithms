use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;
#[derive(Debug, Clone)]
pub struct InvalidFloatError(f64);

impl fmt::Display for InvalidFloatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} is not a valid floating point input.", self.0)
    }
}

// impl error::Error for InvalidFloatError {
//     // fn source(&self) -> Option<&(dyn error::Error + 'static)> {
//     //     Some(&self.0)
//     // }
// }

impl From<InvalidFloatError> for PyErr {
    fn from(err: InvalidFloatError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}

#[derive(Debug)]
pub enum NormalError {
    BadMean(f64),
    BadStandardDeviation(f64),
}

impl fmt::Display for NormalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NormalError::BadMean(ref err) => write!(f, "Bad mean: {}", err),
            NormalError::BadStandardDeviation(ref err) => {
                write!(f, "Bad standard deviation: {}", err)
            }
        }
    }
}

// impl error::Error for NormalError {
//     fn source(&self) -> Option<&(dyn error::Error + 'static)> {
//         match *self {
//             NormalError::BadMean(ref err) => None, //Some(err),
//             NormalError::BadStandardDeviation(ref err) => Some(err),
//         }
//     }
// }

// impl From<InvalidFloatError> for NormalError {
//     fn from(err: InvalidFloatError) -> NormalError {
//
//     }
// }

// enum NormalKind {
//     Normal {mean: f64, stddev: f64},
//     DiracNormal {mean: f64}
// }
//
// impl NormalKind {
//     pub(crate) fn phi(&self, point: f64) -> f64 {
//         match self {
//             NormalKind::Normal => NormalKind::Normal.phi(point),
//             NormalKind::DiracNormal => NormalKind::DiracNormal.phi(point)
//         }
//     }
// }

#[derive(Copy, Clone, Debug)]
pub struct Normal {
    mean: f64,
    stddev: f64,
}

impl Normal {
    pub fn new(mean: f64, stddev: f64) -> Result<Self, NormalError> {
        if mean.is_finite() && stddev.is_finite() && stddev >= 0.0 {
            Ok(Self { mean, stddev })
        } else {
            Err(NormalError::BadStandardDeviation(stddev))
        }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn stddev(&self) -> f64 {
        self.stddev
    }

    pub fn set_mean(&mut self, mean: f64) -> Result<f64, NormalError> {
        if mean.is_finite() {
            self.mean = mean;
            Ok(mean)
        } else {
            Err(NormalError::BadMean(mean))
        }
    }

    pub fn set_stddev(&mut self, stddev: f64) -> Result<f64, NormalError> {
        if stddev.is_finite() && stddev >= 0.0 {
            self.stddev = stddev;
            Ok(stddev)
        } else {
            Err(NormalError::BadStandardDeviation(stddev))
        }
    }

    pub fn phi(&self, point: f64) -> f64 {
        match self.stddev {
            0.0 => {
                if point == self.mean {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                let denom: f64 = self.stddev * (2.0 * std::f64::consts::PI).sqrt();
                let ex = -(0.5 * (point - self.mean).powi(2) / (self.stddev.powi(2)));
                ex.exp() / denom
            }
        }
    }

    pub fn update_params(&mut self, mean: f64, stddev: f64) -> Result<(), NormalError> {
        self.set_mean(mean)?;
        self.set_stddev(stddev)?;
        Ok(())
    }
}

// Dirac delta distribution with variance 0
// pub struct DiracNormal {
//     mean: f64
// }
//
// impl DiracNormal {
//     pub fn phi(&self, point: f64) -> f64 {
//        if point == self.mean {1.0} else {0.0}
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_standard() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        assert_eq!(normal.mean, 0.0);
        assert_eq!(normal.stddev, 1.0);
    }

    #[test]
    fn test_normal_negative_mean() {
        let normal = Normal::new(-1.0, 1.0).unwrap();
        assert_eq!(normal.mean, -1.0);
        assert_eq!(normal.stddev, 1.0);
    }

    #[test]
    fn test_normal_negative_stddev() {
        let normal = Normal::new(0.0, -1.0);
        assert!(normal.is_err());
    }

    fn get_standard_normal() -> Normal {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal
    }

    #[test]
    fn test_set_mean() {
        let mut normal = get_standard_normal();
        assert_eq!(normal.mean(), 0.0);
        let res_success = normal.set_mean(23.5);
        assert!(res_success.is_ok());
        assert_eq!(normal.mean(), 23.5);
        let res_failure = normal.set_mean(f64::INFINITY);
        assert!(res_failure.is_err());
    }

    #[test]
    fn test_set_stddev() {
        let mut normal = get_standard_normal();
        assert_eq!(normal.stddev(), 1.0);
        let res_success = normal.set_stddev(36.2);
        assert!(res_success.is_ok());
        assert_eq!(normal.stddev(), 36.2);
        let fail_one = -3.14;
        let res_failure = normal.set_stddev(fail_one);
        assert!(res_failure.is_err());
        let res_failure = normal.set_stddev(0.0);
        assert!(res_failure.is_ok());
    }

    #[test]
    fn test_phi() {
        let normal = get_standard_normal();
        let answer: f64 = (2.0 * std::f64::consts::PI).sqrt().recip();
        assert_eq!(normal.phi(0.0), answer);
    }
}
