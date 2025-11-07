pub struct NormalInverseGamma {
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub kappa: f64,
}

impl Default for NormalInverseGamma {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            mu: 0.0,
            kappa: 1.0,
        }
    }
}