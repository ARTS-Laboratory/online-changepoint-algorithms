use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;
use std::num::NonZero;

#[derive(Copy, Clone, Debug)]
pub struct PositiveError(u32);

impl fmt::Display for PositiveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} is not a positive number.", self.0)
    }
}

impl From<PositiveError> for PyErr {
    fn from(err: PositiveError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PositiveInteger(u32);

impl PositiveInteger {
    pub fn new(value: u32) -> Result<PositiveInteger, PositiveError> {
        if is_positive(value) {
            Ok(PositiveInteger(value))
        } else {
            Err(PositiveError(value))
        }
    }

    pub fn value(&self) -> u32 {
        self.0
    }

    pub fn set(&mut self, value: u32) -> Result<u32, PositiveError> {
        if is_positive(value) {
            self.0 = value;
            Ok(value)
        } else {
            Err(PositiveError(value))
        }
    }
}

impl Default for PositiveInteger {
    fn default() -> PositiveInteger {
        PositiveInteger(1)
    }
}

fn is_positive(value: u32) -> bool {
    // only need to check if 0 since unsinged int cannot be negative
    value != 0
}

pub struct PositiveInteger2(NonZero<u32>);

impl PositiveInteger2 {
    pub fn new(value: u32) -> Result<Self, PositiveError> {
        match NonZero::new(value) {
            Some(x) => Ok(PositiveInteger2(x)),
            None => Err(PositiveError(value)),
        }
    }

    pub fn value(&self) -> u32 {
        self.0.get()
    }

    pub fn set(&mut self, value: u32) -> Result<u32, PositiveError> {
        NonZero::new(value)
            .and_then(|x| {
                self.0 = x;
                Some(value)
            })
            .ok_or(PositiveError(value))
        // match NonZero::new(value) {
        //     Some(x) => {
        //         self.0 = x;
        //         Ok(self)
        //     },
        //     None => Err(PositiveError(value))
        // }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_integer_default() {
        let x = PositiveInteger::default();
        assert_eq!(x.value(), 1);
    }

    #[test]
    fn test_positive_integer_new() {
        let x = PositiveInteger::new(10).unwrap();
        assert_eq!(x.value(), 10);
    }

    #[test]
    fn test_positive_integer_new_fail() {
        let x = PositiveInteger::new(0);
        assert!(x.is_err());
    }

    // Positive Integer 2

    #[test]
    fn test_positive_integer2_constructor() {
        let x = PositiveInteger2::new(10);
        assert!(x.is_ok());
        assert_eq!(x.unwrap().value(), 10);
    }

    #[test]
    fn test_positive_integer2_new_fail() {
        let x = PositiveInteger2::new(0);
        assert!(x.is_err());
    }

    #[test]
    fn test_positive_integer2_set_value() {
        let x = PositiveInteger2::new(10);
        assert!(x.is_ok());
        let mut pos_int = x.unwrap();
        assert_eq!(pos_int.value(), 10);
        let res = pos_int.set(2);
        assert!(res.is_ok());
        assert_eq!(pos_int.value(), 2);
    }
}
