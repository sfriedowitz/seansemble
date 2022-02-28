use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::core::{Label, TrainingRow};

pub fn check_dimensions<X, Y, W>(X: &Array2<X>, y: &Array1<Y>, w: &Array1<W>) {
    let (nx, ny, nw) = (X.nrows(), y.ndim(), w.ndim());
    if (nx, ny) != (ny, nw) {
        panic!["Input dimensions do not match: X = {}, y = {}, w = {}", nx, ny, nw];
    }
}

pub fn build_training_data<L: Label>(
    X: &Array2<f64>,
    y: &Array1<L>,
    w: Option<&Array1<f64>>,
) -> Vec<TrainingRow<L>> {
    // Unit weights if not provided
    let w = match w {
        Some(weights) => weights.to_owned(),
        None => Array::from_elem(y.len(), 1.0),
    };

    // Panic if mismatched dimensions
    check_dimensions(X, y, &w);

    (0..X.nrows())
        .map(|idx| {
            let reals = X.row(idx).iter().copied().collect();
            let yi = y[idx];
            let wi = w[idx];

            TrainingRow::new(reals, vec![], yi, wi)
        })
        .collect()
}

pub fn random_training_data<L: Label>(ns: usize, nr: usize, nc: usize) -> Vec<TrainingRow<L>> {
    let real_range = Uniform::new(-10.0, 10.0);
    let cat_range = Uniform::new(0, 5);

    let mut data = Vec::with_capacity(ns);
    for _ in 0..ns {
        let reals = Array::random(nr, real_range).to_vec();
        let cats = Array::random(nc, cat_range).to_vec();
        let y = L::gen_range(L::from(0).unwrap()..L::from(5).unwrap());
        let w = rand::random();
        data.push(TrainingRow::new(reals, cats, y, w));
    }

    data
}

pub fn linear_training_data(ns: usize, coeffs: &[f64], intercept: f64) -> Vec<TrainingRow<f64>> {
    let real_range = Uniform::new(-10.0, 10.0);

    let mut data = Vec::with_capacity(ns);
    for _ in 0..ns {
        let reals = Array::random(coeffs.len(), real_range).to_vec();
        let y =
            intercept + coeffs.iter().zip(reals.iter()).fold(0.0, |state, (c, x)| state + c * x);
        let w = rand::random();
        data.push(TrainingRow::new(reals, vec![], y, w));
    }

    data
}
