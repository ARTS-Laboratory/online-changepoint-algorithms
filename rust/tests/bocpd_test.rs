use _change_point_algorithms::bocpd::bocpd_model::BocpdModel;
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

#[test]
fn test_bocpd_all_normal() {
    let data = generate_data();
    let mut model = BocpdModel::default();
    let lamb = 0.5;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event, lamb).expect("Should not fail to update model");
        let prediction = model.predict(event);
        if prediction > 0.05 {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_safe >= count_unsafe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}

#[test]
fn test_bocpd_all_abnormal() {
    let data = generate_abnormal_data();
    let mut model = BocpdModel::default();
    let lamb = 0.5;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update(event, lamb).expect("Should not fail to update model");
        let prediction = model.predict(event);
        if prediction > 0.05 {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_unsafe >= count_safe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}
