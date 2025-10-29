use criterion::{criterion_group, criterion_main, Criterion};
use _change_point_algorithms::cusum::{CusumV0, CusumV1};
use rand_distr::Distribution;
use std::hint::black_box;
use helpers::generate_normal_data;
mod helpers;

fn get_params() -> (f64, f64, f64, f64) {
    let mean = 0.0;
    let std_dev = 10.0;
    let alpha = 0.95;
    let threshold = 3.0;
    (mean, std_dev, alpha, threshold)
}

pub fn cusumv0_benchmark(c: &mut Criterion) {
    let (mean, std_dev, alpha, threshold) = get_params();
    let data_size = 400_000;
    let unknowns = generate_normal_data(mean, std_dev, data_size);
    c.bench_function("Cusum v0", |b| {
        b.iter(|| {
            let mut model = CusumV0::new(mean, std_dev.powi(2), alpha, threshold);
            for &point in black_box(&unknowns) {
                let _ = black_box(model.update(black_box(point)));
                let _prediction = black_box(model.predict(point));
            }
        })
    });
}

pub fn cusumv1_benchmark(c: &mut Criterion) {
    let (mean, std_dev, alpha, threshold) = get_params();
    let data_size = 400_000;
    let unknowns = generate_normal_data(mean, std_dev, data_size);
    c.bench_function("Cusum v1", |b| {
        b.iter(|| {
            let mut model = CusumV1::new(mean, std_dev.powi(2), alpha, threshold);
            for &point in black_box(&unknowns) {
                let _ = black_box(model.update(black_box(point)));
                let _prediction = black_box(model.predict(point));
            }
        })
    });
}

criterion_group!(benches, cusumv0_benchmark, cusumv1_benchmark);
criterion_main!(benches);