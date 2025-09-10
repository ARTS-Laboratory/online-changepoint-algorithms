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

#[derive(Copy, Clone, Debug)]
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
