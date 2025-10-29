use criterion::{criterion_group, criterion_main, Criterion};
use rand_distr::StandardNormal;
use _change_point_algorithms::bocpd::bocpd_model::BocpdModel;
use rand_distr::Distribution;
use std::hint::black_box;

pub fn bocpd_model_benchmark(c: &mut Criterion) {
    let data_size = 400_000;
    let mu = 0.0;
    let kappa = 1.0;
    let alpha = 0.5;
    let beta = 1.0;
    let lambda = 0.5;
    let rng = rand::rng();
    let unknown_data: Vec<f64> = StandardNormal.sample_iter(rng).take(data_size).collect();
    c.bench_function("bocpd naive model", |b| {
        b.iter(|| {
            let mut model = BocpdModel::new_py(alpha, beta, mu, kappa, true, None).expect("Should work because this is a benchmark");
            for &point in black_box(&unknown_data) {
                let _ = black_box(model.update(black_box(point), lambda));
                let _prediction = black_box(model.predict(point));
            }
        })
    });
}

criterion_group!(benches, bocpd_model_benchmark);
criterion_main!(benches);