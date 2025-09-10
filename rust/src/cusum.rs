use pyo3::{pyclass, pymethods};

/// A class that implements a version of Cumulative Summation.
#[pyclass]
pub struct CusumV0 {
    mean: f64,
    // variance: f64,
    mu: LastTwo<f64>,
    cp: LastTwo<f64>,
    cn: LastTwo<f64>,
    d: f64,
    alpha: f64,
    threshold: f64,
    // these are calculated
    scalar: f64,
    weight_no_diff: f64,
}

#[pymethods]
impl CusumV0 {
    #[new]
    pub fn new(mean: f64, variance: f64, alpha: f64, threshold: f64) -> Self {
        let d = 0.0;
        let scalar = 1.0 + alpha * 0.5;
        let weight_no_diff = alpha / variance;
        let mu = LastTwo { prev: 0.0, curr: mean}; // LastTwo::default();
        let cp = LastTwo::default();
        let cn = LastTwo::default();
        Self {
            mean, mu, cp, cn, d, alpha, threshold, scalar, weight_no_diff
        }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn set_d(&mut self, d: f64) {
        self.d = d;
    }

    pub fn update(&mut self, point: f64) {
        let weight = self.d * self.weight_no_diff;
        self.update_cp(point, weight);
        self.update_cn(point, weight);
        self.set_d(self.mu.curr() - self.mean());
        self.mu.append((1.0 - self.alpha) * self.mu.prev() + self.alpha * point);
    }

    pub fn predict(&mut self, _point: f64) -> f64 {
        let out: f64 = self.cp.curr().max(self.cn.curr().abs());
        if out > self.threshold {
            self.reset_current_shifts()
        }
        out
    }

    fn reset_current_shifts(&mut self) {
        self.cp.set_curr(0.0);
        self.cn.set_curr(0.0);
    }

    fn update_cp(&mut self, point: f64, weight: f64) {
        let value = 0.0_f64.max(self.cp.curr() + weight * (point - self.d * self.scalar));
        self.cp.append(value);
    }

    fn update_cn(&mut self, point: f64, weight: f64) {
        let value = 0.0_f64.min(self.cn.curr() - weight * (point + self.d * self.scalar));
        self.cn.append(value);
    }
}

/// A class that implements a version of Cumulative Summation.
#[pyclass]
pub struct CusumV1 {
    mean: f64,
    variance: f64,
    mu: LastTwo<f64>,
    cp: LastTwo<f64>,
    cn: LastTwo<f64>,
    alpha: f64,
    threshold: f64,
}

#[pymethods]
impl CusumV1 {
    #[new]
    pub fn new(mean: f64, std_dev: f64, h: f64, alpha: f64) -> Self {
        let cp = LastTwo::new(0.0, 0.0);
        let cn = LastTwo::new(0.0, 0.0);
        // todo might need to change this to last value.
        let mu = LastTwo::new(0.0, 0.0);
        let threshold = std_dev * h;
        let variance = std_dev.powi(2);
        Self {
            mean,
            variance,
            mu,
            cp,
            cn,
            alpha,
            threshold,
        }
    }

    /// Update model parameters using given input value.
    pub fn update(&mut self, point: f64) {
        let dev_shift = (self.mu.prev() - self.mean) / self.variance;
        let mean_mean = (self.alpha * self.mu.prev + self.mean) * 0.5;
        let target = self.mu.prev() + mean_mean;
        self.update_mu(point);
        self.update_cp(point, dev_shift, target);
        self.update_cn(point, dev_shift, target);
    }

    pub fn predict(&mut self, _point: f64) -> f64 {
        let out = self.cp.curr().max(self.cn.curr().abs());
        if out > self.threshold {
            self.reset_current_shifts()
        }
        out
    }

    fn reset_current_shifts(&mut self) {
        self.cp.set_curr(0.0);
        self.cn.set_curr(0.0);
    }

    fn update_mu(&mut self, point: f64) {
        let value = self.alpha * self.mu.prev() - (1.0 - self.alpha) * point;
        self.mu.append(value);
    }

    fn update_cp(&mut self, point: f64, dev_shift: f64, target: f64) {
        let value = 0.0_f64.max(self.cp.curr() + dev_shift * (point - target));
        self.cp.append(value);
    }

    fn update_cn(&mut self, point: f64, dev_shift: f64, target: f64) {
        let value = 0.0_f64.min(self.cn.curr() - dev_shift * (point + target));
        self.cn.append(value);
    }
}

struct LastTwo<T> {
    prev: T,
    curr: T,
}

impl<T> LastTwo<T> {
    pub fn new(prev: T, curr: T) -> Self {
        Self { prev, curr }
    }

    pub fn prev(&self) -> &T {
        &self.prev
    }

    pub fn set_prev(&mut self, prev: T) {
        self.prev = prev;
    }

    pub fn curr(&self) -> &T {
        &self.curr
    }

    pub fn set_curr(&mut self, curr: T) {
        self.curr = curr;
    }
}

impl LastTwo<f64> {
    pub fn append(&mut self, item: f64) -> f64 {
        let out = self.prev;
        self.prev = self.curr;
        self.curr = item;
        out
    }
}

impl Default for LastTwo<f64> {
    fn default() -> Self {
        LastTwo { prev: 0.0, curr: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Last Two Tests
    #[test]
    fn test_last_two_constructor() {
        let lt = LastTwo::new(0.0, 0.0);
        assert_eq!(lt.prev, 0.0);
        assert_eq!(lt.curr, 0.0);
    }

    #[test]
    fn test_last_two_append() {
        let mut lt = LastTwo::default();
        lt.append(24.0);
        assert_eq!(lt.curr, 24.0);
        lt.append(77.42);
        assert_eq!(lt.prev, 24.0);
        assert_eq!(lt.curr, 77.42);
    }

    // Test CusumV0
    #[test]
    fn test_cusum_v0() {
        let mean = 0.0;
        let variance = 1.0;
        let alpha = 0.5;
        let threshold = 5.0;
        let model = CusumV0::new(mean, variance, alpha, threshold);
    }

    fn make_cusum_v0() -> CusumV0 {
        let mean = 0.0;
        let variance = 1.0;
        let alpha = 0.5;
        let threshold = 5.0;
        CusumV0::new(mean, variance, alpha, threshold)
    }
    #[test]
    fn test_cusum_v0_normal_point() {
        let mut model = make_cusum_v0();
        let normal_point = 0.01;
        for _idx in 0..5 {
            model.update(normal_point);
            model.predict(normal_point);
        }
        model.update(normal_point);
        let prob = model.predict(normal_point);
        assert!(prob < model.variance * model.threshold);
    }

    #[test]
    fn test_cusum_v0_abnormal_point() {
        let mut model = make_cusum_v0();
        let abnormal_point = 20.0;
        for _idx in 0..5 {
            model.update(abnormal_point);
            model.predict(abnormal_point);
        }
        model.update(abnormal_point);
        dbg!(model.d);
        model.predict(abnormal_point);
        model.update(abnormal_point);
        let prob = model.predict(abnormal_point);
        dbg!(prob);
        assert!(prob > model.variance * model.threshold);
    }

    // Test CusumV1
    #[test]
    fn test_cusum_v1() {
        let mean = 0.0;
        let variance = 1.0;
        let alpha = 0.5;
        let threshold = 5.0;
        let model = CusumV1::new(mean, variance, alpha, threshold);
    }

    fn make_cusum_v1() -> CusumV1 {
        let mean = 0.0;
        let variance = 1.0;
        let alpha = 0.5;
        let threshold = 5.0;
        CusumV1::new(mean, variance, alpha, threshold)
    }

    #[test]
    fn test_cusum_v1_normal_point() {
        let mut model = make_cusum_v1();
        let normal_point = 0.01;
        for _idx in 0..5 {
            model.update(normal_point);
            model.predict(normal_point);
        }
        model.update(normal_point);
        let prob = model.predict(normal_point);
        assert!(prob < model.variance * model.threshold);
    }

    #[test]
    fn test_cusum_v1_abnormal_point() {
        let mut model = make_cusum_v1();
        let abnormal_point = 20.0;
        for idx in 0..5 {
            model.update(abnormal_point);
            model.predict(abnormal_point);
        }
        model.update(abnormal_point);
        model.predict(abnormal_point);
        model.update(abnormal_point);
        let prob = model.predict(abnormal_point);
        dbg!(prob);
        assert!(prob > model.variance * model.threshold);
    }
}
