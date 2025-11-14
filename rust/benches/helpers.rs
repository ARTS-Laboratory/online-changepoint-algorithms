use rand::distr::Distribution;
use rand_distr::{Normal};

pub fn generate_normal_data(mean: f64, std_dev: f64, num: u32) -> Vec<f64> {
    let rng = rand::rng();
    let dist = Normal::new(mean, std_dev)
        .unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("Standard normal distribution."));
    dist.sample_iter(rng).take(num as usize).collect()
}