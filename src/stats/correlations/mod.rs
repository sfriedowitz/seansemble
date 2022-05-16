use nalgebra::{DMatrix, DVector};

pub fn distance_matrix<T>(x: &[T], y: &[T], metric: fn(&T, &T) -> f64) -> DMatrix<f64> {
    let n = x.len();
    let mut result = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let d = metric(&x[i], &y[j]);
            result[(i, j)] = d;
            result[(j, i)] = d;
        }
    }
    result
}

pub fn centered_distance<T>(x: &[T], metric: fn(&T, &T) -> f64) -> DMatrix<f64> {
    let n = x.len();
    let pair_distances = distance_matrix(x, x, metric);

    let grand_mean = pair_distances.sum() / ((n * n) as f64);
    let row_means = DVector::from_fn(n, |i, _| pair_distances.row(i).sum() / (n as f64));
    DMatrix::from_fn(n, n, |i, j| pair_distances[(i, j)] - row_means[i] - row_means[j] + grand_mean)
}

pub fn distance_covariance<T>(x: &[T], y: &[T], metric: fn(&T, &T) -> f64) -> f64 {
    let n = x.len();
    let A = centered_distance(x, metric);
    let B = centered_distance(y, metric);

    ((A * B).sum() / (n * n) as f64).sqrt()
}

pub fn distance_correlation<T>(x: &[T], y: &[T], metric: fn(&T, &T) -> f64) -> f64 {
    let cov = distance_covariance(x, y, metric);
    let varx = distance_covariance(x, x, metric);
    let vary = distance_covariance(y, y, metric);
    let varxy = (varx * vary).sqrt();

    if cov == 0.0 && varxy == 0.0 {
        0.0
    } else {
        cov / varxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix() {
        let x = Vec::from_iter((0..5).map(|i| i as f64));

        let dists = distance_matrix(&x, &x, |x, y| (x - y).abs());
        dists.diagonal().iter().for_each(|x| assert_eq!(*x, 0.0));
    }
}
