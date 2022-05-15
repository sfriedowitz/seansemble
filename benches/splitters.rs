use criterion::{black_box, criterion_group, Criterion};
use rand::prelude::{Rng, SeedableRng, StdRng};

use seansemble::{
    core::TrainingRow,
    trees::splits::{RegressionSplitter, Splitter},
};

pub fn regression_splitter(c: &mut Criterion) {
    let mut rng: StdRng = SeedableRng::seed_from_u64(0);

    let (nr, nc) = (10, 10);
    let data: Vec<_> = (0..100)
        .map(|_| {
            let reals = (0..nr).map(|_| rng.gen_range(0.0..10.0)).collect();
            let categoricals = (0..nc).map(|_| rng.gen_range(0..5)).collect();
            let label: f64 = rng.gen_range(0.0..100.0);
            let weight: f64 = rng.gen();
            TrainingRow::new(reals, categoricals, label, weight)
        })
        .collect();

    let mut splitter = RegressionSplitter::new(true, Some(&mut rng));

    c.bench_function("Regression Splitter", move |b| {
        b.iter(|| splitter.find_best_split(black_box(&data), 10, 2))
    });
}

criterion_group!(splitters, regression_splitter);
