use change_point_algorithms::bocpd::bocpd;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand;
use rand::distr::Distribution;
use rand_distr::StandardNormal;

pub fn bocpd_benchmark(c: &mut Criterion) {
    let data_size = 100_000;
    let mu = 0.0;
    let kappa = 1.0;
    let alpha = 0.5;
    let beta = 1.0;
    let lambda = 0.5;
    let rng = rand::rng();
    let unknown_data: Vec<f64> = StandardNormal.sample_iter(rng).take(data_size).collect();
    c.bench_function("bocpd naive vec", |b| {
        b.iter(|| {
            bocpd(
                black_box(unknown_data.iter()),
                mu,
                kappa,
                alpha,
                beta,
                lambda,
            )
        })
    });
}

criterion_group!(benches, bocpd_benchmark);
criterion_main!(benches);
