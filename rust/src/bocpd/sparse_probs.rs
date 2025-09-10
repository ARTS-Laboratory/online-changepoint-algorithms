use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, PyResult};
use std::collections::VecDeque;
use std::iter::zip;
use std::ops::{Deref, DerefMut};

#[pyclass]
pub struct SparseProb {
    #[pyo3(get)]
    pos: i64,
    #[pyo3(get, set)]
    value: f64,
}

#[pymethods]
impl SparseProb {
    #[new]
    fn new_py(run_length: i64, value: f64) -> PyResult<Self> {
        match run_length {
            0.. => Ok(Self {
                pos: run_length,
                value,
            }),
            ..0 => Err(PyValueError::new_err("run length must be nonnegative")),
        }
        // if run_length.is_positive() {
        //     Ok(SparseProb {pos: run_length, value})
        // } else {
        //     Err(PyValueError::new_err("run length must be positive"))
        // }
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }

    #[setter]
    fn set_pos(&mut self, pos: i64) -> PyResult<()> {
        if !self.pos.is_negative() {
            self.pos = pos;
            Ok(())
        } else {
            Err(PyValueError::new_err("run length must be nonnegative"))
        }
    }

    fn increment(&mut self) -> i64 {
        self.pos += 1;
        self.pos
    }
}

#[pyclass]
pub struct SparseProbs {
    probs: VecDeque<SparseProb>,
}

#[pymethods]
impl SparseProbs {
    #[new]
    pub fn new_py() -> Self {
        Self {
            probs: VecDeque::new(),
        }
    }

    pub fn reset(&mut self) {
        self.probs.clear();
        self.probs.push_back(SparseProb { pos: 0, value: 1.0 });
    }

    pub fn normalize(&mut self) {
        let total: f64 = self.probs.iter().map(|prob| prob.value).sum();
        if total.is_normal() {
            // not zero,
            for prob in self.probs.iter_mut() {
                prob.value /= total;
            }
        }
    }

    pub fn new_entry(&mut self, run_length: i64, value: f64) -> PyResult<()> {
        // todo make setter method that does the check.
        SparseProb::new_py(run_length, value).and_then(|item| {
            self.probs.push_front(item);
            Ok(())
        })
    }

    pub fn update_probs(&mut self, priors: Vec<f64>, hazard: f64) -> PyResult<()> {
        let mut head = 0.0;
        let negative_hazard = 1.0 - hazard;
        for (sparse_prob, prior) in zip(self.probs.iter_mut(), priors.iter()) {
            let val = sparse_prob.value * prior;
            head += val;
            sparse_prob.value = val * negative_hazard;
            sparse_prob.increment();
        }
        head *= hazard;
        self.new_entry(0, head)
    }

    pub fn max_prob(&self) -> (i64, f64) {
        let mut max_value = -f64::INFINITY;
        let mut max_idx = 0;
        for prob in self.probs.iter() {
            if prob.value > max_value {
                max_value = prob.value;
                max_idx = prob.pos;
            }
        }
        (max_idx, max_value)
    }
}

impl Deref for SparseProbs {
    type Target = VecDeque<SparseProb>;

    fn deref(&self) -> &Self::Target {
        &self.probs
    }
}

impl DerefMut for SparseProbs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.probs
    }
}
