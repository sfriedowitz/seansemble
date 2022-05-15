use nalgebra::{DMatrix, DVector};

pub fn distance_matrix<T>(x: &[T], y: &[T], distance: fn(&T, &T) -> f64) -> DMatrix<f64> {
    let n = x.len();
    let mut result = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let d = distance(&x[i], &y[j]);
            result[(i, j)] = d;
            result[(j, i)] = d;
        }
    }
    result
}

pub fn centered_distance<T>(x: &[T], distance: fn(&T, &T) -> f64) -> Array2<f64> {
    let n = x.len();
    let pair_distances = distance_matrix(x, x, distance);

    let grand_mean = pair_distances.sum() / ((n * n) as f64);
    let row_means: Array1<f64> =
        pair_distances.rows().into_iter().map(|row| row.sum() / (n as f64)).collect();

    Array2::from_shape_fn((n, n), |(i, j)| {
        pair_distances[(i, j)] - row_means[i] - row_means[j] + grand_mean
    })
}

pub fn distance_covariance<T>(x: &[T], y: &[T], distance: fn(&T, &T) -> f64) -> f64 {
    let n = x.len();
    let A = centered_distance(x, distance);
    let B = centered_distance(y, distance);

    ((A * B).sum() / (n * n) as f64).sqrt()
}

pub fn distance_correlation<T>(x: &[T], y: &[T], distance: fn(&T, &T) -> f64) -> f64 {
    let cov = distance_covariance(x, y, distance);
    let varx = distance_covariance(x, x, distance);
    let vary = distance_covariance(y, y, distance);

    cov / (varx * vary).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix() {
        let x = Vec::from_iter((0..3).map(|i| i as f64));

        let dists = distance_matrix(&x, &x, |x, y| (x - y).abs());
        println!("{}", dists);
    }
}
