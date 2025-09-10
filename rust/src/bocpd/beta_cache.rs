use pyo3::{pyclass, pymethods};
use statrs::function::beta::beta;
use std::collections::HashMap;

/// Cache for beta function with a fixed value
#[pyclass]
pub struct BetaCache {
    #[pyo3(get)]
    fixed_value: f64,
    cache: HashMap<u64, f64>,
}

#[pymethods]
impl BetaCache {
    #[new]
    pub fn new_py(fixed_value: f64) -> Self {
        Self {
            fixed_value,
            cache: HashMap::new(),
        }
    }

    pub fn get_fixed_value(&self) -> f64 {
        self.fixed_value
    }

    pub fn get_value(&mut self, value: f64) -> f64 {
        let key = value.to_bits();
        // if self.cache.contains_key(&key) {
        //     let value = self
        //         .cache
        //         .get(&key)
        //         .expect("Cache should always have key here since we checked just before.");
        //     *value
        // } else {
        //     // base case
        //     let fixed = self.fixed_value;
        //     let res = match value {
        //         ..0.0 => todo!("does not expect negative numbers. What should we do in this case?"), // beta(steady_x, increase_y), // technically, should throw error
        //         0.0..=1.0 => beta(fixed, value),
        //         1.0.. => (value / (fixed + value)) * beta(fixed, value - 1.0),
        //         _ => todo!("add functionality for other float types."),
        //     };
        //     self.cache.insert(key, res.clone());
        //     res
        // }
        // New attempt
        if let std::collections::hash_map::Entry::Vacant(e) = self.cache.entry(key) {
            let fixed = self.fixed_value;
            let res = match value {
                ..0.0 => todo!("does not expect negative numbers. What should we do in this case?"), // beta(steady_x, increase_y), // technically, should throw error
                0.0..=1.0 => beta(fixed, value),
                1.0.. => (value / (fixed + value)) * beta(fixed, value - 1.0),
                _ => todo!("add functionality for other float types."),
            };
            e.insert(res);
            res
        } else {
            let &value = self.cache.get(&key).expect("Cache should always have key here since we checked just before.");
            value
        }
    }
}
