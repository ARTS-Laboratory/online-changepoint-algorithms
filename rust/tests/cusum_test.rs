use _change_point_algorithms::cusum::{CusumV0, CusumV1};
use helpers::generate_normal_data;

mod helpers;

fn generate_data() -> Vec<f64> {
    let mean = 0.0;
    let std_dev = 1.0;
    let num = 1_000;
    generate_normal_data(mean, std_dev, num)
}

fn generate_abnormal_data() -> Vec<f64> {
    let mean = 50.0;
    let std_dev = 1.0;
    let num = 1_000;
    generate_normal_data(mean, std_dev, num)
}

// CusumV0 tests
#[test]
fn test_cusum_v0_all_normal() {
    let data = generate_data();
    let mean = 0.0;
    let std_dev: f64 = 1.0;
    let alpha = 0.5;
    let threshold = 3.0;
    let mut model = CusumV0::new(mean, std_dev.powi(2), alpha, threshold);
    let boundary = threshold * std_dev;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event);
        let prediction = model.predict(event);
        if prediction < boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_safe >= count_unsafe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}

#[test]
fn test_cusum_v0_all_abnormal() {
    let data = generate_abnormal_data();
    let mean = 0.0;
    let std_dev: f64 = 1.0;
    let alpha = 0.5;
    let threshold = 3.0;
    let mut model = CusumV0::new(mean, std_dev.powi(2), alpha, threshold);
    let boundary = threshold * std_dev;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event);
        let prediction = model.predict(event);
        if prediction < boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_unsafe >= count_safe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}

// CusumV1 tests
#[test]
fn test_cusum_v1_all_normal() {
    let data = generate_data();
    let mean = 0.0;
    let std_dev: f64 = 1.0;
    let alpha = 0.5;
    let threshold = 3.0;
    let mut model = CusumV1::new(mean, std_dev, alpha, threshold);
    let boundary = threshold * std_dev;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event);
        let prediction = model.predict(event);
        if prediction < boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_safe >= count_unsafe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}

#[test]
fn test_cusum_v1_all_abnormal() {
    let data = generate_abnormal_data();
    let mean = 0.0;
    let std_dev: f64 = 1.0;
    let alpha = 0.5;
    let threshold = 3.0;
    let mut model = CusumV1::new(mean, std_dev, alpha, threshold);
    let boundary = threshold * std_dev;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event);
        let prediction = model.predict(event);
        if prediction < boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_unsafe >= count_safe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}