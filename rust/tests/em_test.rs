use _change_point_algorithms::expect_max::{em_model::EmModel, em_early_stop_model::EarlyStopEmModel};
use _change_point_algorithms::expect_max::em_model_builder;
use _change_point_algorithms::expect_max::em_early_stop_model::EmLikelihoodCheck;
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
    let std_dev = 2.0;
    let num = 1_000;
    generate_normal_data(mean, std_dev, num)
}

// #[test]
// fn test_build_em_early_stop_model() {
//     let model = build_em_early_stop_model
// }

#[test]
fn test_em_all_normal() {
    let data = generate_data();
    let mut builder = em_model_builder::EmBuilderOne::new();
    let early_stop_builder = builder
        .build_normal(0.0, 1.0, 0.7).unwrap()
        .build_abnormal_from_tuples(&[(50.0, 2.0, 0.3)]).unwrap()
        .build_samples_from_slice(&[0.0, -0.2, 0.2, -1.0, 1.0, -0.5, 0.5, 50.0, 49.0, 51.0])
        .next_builder().unwrap()
        .build_likelihoods()
        .next_builder().unwrap()
        .build_likelihood_converge_checker()
        .get_early_stop_model();
    let mut model: EmLikelihoodCheck = EmLikelihoodCheck::from_early_stop_model(early_stop_builder);
    let threshold = 1e-8;
    let boundary = 0.5;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update_check_convergence(event, threshold).unwrap();
        let prediction = model.predict(event);
        if prediction >= boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_safe >= count_unsafe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}

#[test]
fn test_em_all_abnormal() {
    let data = generate_abnormal_data();
    let mut builder = em_model_builder::EmBuilderOne::new();
    let early_stop_builder = builder
        .build_normal(0.0, 1.0, 0.7).unwrap()
        .build_abnormal_from_tuples(&[(50.0, 2.0, 0.3)]).unwrap()
        .build_samples_from_slice(&[0.0, -0.2, 0.2, -1.0, 1.0, -0.5, 0.5, 50.0, 49.0, 51.0])
        .next_builder().unwrap()
        .build_likelihoods()
        .next_builder().unwrap()
        .build_likelihood_converge_checker()
        .get_early_stop_model();
    let mut model: EmLikelihoodCheck = EmLikelihoodCheck::from_early_stop_model(early_stop_builder);
    let threshold = 1e-8;
    let boundary = 0.5;
    let mut count_safe = 0;
    let mut count_unsafe = 0;
    for event in data {
        model.update_check_convergence(event, threshold).unwrap();
        let prediction = model.predict(event);
        if prediction >= boundary {
            count_safe += 1;
        } else {
            count_unsafe += 1;
        }
    }
    assert!(count_unsafe >= count_safe, "count_safe: {}, count_unsafe: {}", count_safe, count_unsafe);
}