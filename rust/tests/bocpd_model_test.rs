use helpers::generate_normal_data;
use _change_point_algorithms::bocpd::bocpd_model::BocpdModel;

use std::cmp::max;

mod helpers;

#[test]
fn test_bocpd_model_all_normal() {
    let mean = 0.0;
    let std_dev = 1.0;
    let num_unknowns = 100;
    let data = generate_normal_data(mean, std_dev, num_unknowns);
    let lambda = 2.0;
    let mut model = BocpdModel::default();
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for &item in &data {
        model.update(item, lambda).expect("The model should not fail to update.");
        let prediction = model.predict(item);
        if prediction > 0.05 {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_safe > count_unsafe);
}

#[test]
fn test_bocpd_model_all_normal_param_size() {
    let mean = 0.0;
    let std_dev = 1.0;
    let num_unknowns = 500_000;
    let data = generate_normal_data(mean, std_dev, num_unknowns);
    let lambda = 20.0;
    let mut model = BocpdModel::default();
    let mut max_length = 0;
    for &item in &data {
        model.update(item, lambda).expect("The model should not fail to update.");
        let size = model.params_length();
        // println!("Size: {:?}", size);
        max_length = max(max_length, size);
        let _prediction = model.predict(item);
    }
    println!("Max size of params was {:?}", max_length);
}