pub trait Element {
    fn get_data(&self) -> f64;
}

impl Element for &f64 {
    fn get_data(&self) -> f64 {
        **self
    }
}

impl Element for f64 {
    fn get_data(&self) -> f64 {
        *self
    }
}
