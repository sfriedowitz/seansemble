use ndarray::{arr1, concatenate, s, Array, Array1, Array2, Axis};
use ndarray_linalg::Solve;

use crate::core::{FeatureRow, Learner, Model, ModelingError, Prediction, TrainingRow};

#[derive(Clone, Copy, Debug)]
pub struct LinearRegressionLearner {
    fit_intercept: bool,
    alpha: f64,
}

impl LinearRegressionLearner {
    pub fn new(fit_intercept: bool, alpha: Option<f64>) -> Self {
        LinearRegressionLearner { fit_intercept, alpha: alpha.unwrap_or(0.0).max(0.0) }
    }

    fn solve_normal_equation(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
    ) -> Result<Array1<f64>, ModelingError> {
        // Multiply by weight matrix by scaling X columns (after transpose)
        let mut Xw = X.t().to_owned();
        for (i, mut col) in Xw.columns_mut().into_iter().enumerate() {
            col *= w[i];
        }

        let rhs = Xw.dot(y);
        let mut lhs = Xw.dot(X);

        // Add reg param if present
        if self.alpha != 0.0 {
            for i in 0..lhs.nrows() {
                lhs[(i, i)] += self.alpha;
            }
        }

        match lhs.solve_into(rhs) {
            Ok(beta) => Ok(beta),
            Err(_) => Err(ModelingError::FitError(
                "Failure while inverting linear operator in normal equations.".into(),
            )),
        }
    }
}

impl Learner<f64> for LinearRegressionLearner {
    fn fit(&mut self, data: &[TrainingRow<f64>]) -> Box<dyn Model<f64>> {
        // Get indices that are (1) non-constant and (2) non-NaN
        let indices: Vec<usize> = data[0]
            .features
            .real_indices()
            .filter(|&idx| {
                let has_nans = data.iter().any(|row| row.features.real_at(idx).is_nan());
                let mut is_constant = true;
                let head_value = data[0].features.real_at(idx);
                for row in data {
                    if row.features.real_at(idx) != head_value {
                        is_constant = false;
                        break;
                    }
                }
                !has_nans && !is_constant
            })
            .collect();

        // Collect data into matrices for fitting
        // First collect data in vectors, then assemble into nalgebra matrices
        let (ns, nf) = (data.len(), indices.len());

        let mut x_data = Vec::with_capacity(ns * nf);
        let mut y_data = Vec::with_capacity(ns);
        let mut w_data = Vec::with_capacity(ns);
        for row in data {
            y_data.push(row.label);
            w_data.push(row.weight);
            for &idx in indices.iter() {
                x_data.push(row.features.real_at(idx));
            }
        }

        // If there are not enough data rows, return early with mean
        if ns <= nf {
            let intercept = y_data.iter().sum::<f64>() / (ns as f64);
            let coeffs = vec![0.0; nf];
            return Box::new(LinearRegessionModel { intercept, coeffs, indices });
        }

        let yvec = arr1(&y_data);
        let wvec = arr1(&w_data);

        let Xmat = Array2::from_shape_vec((ns, nf), x_data).expect("X has correct dimensions");
        let Xmat = match self.fit_intercept {
            true => {
                let intercept_col: Array<f64, _> = Array::ones((ns, 1));
                concatenate![Axis(1), intercept_col, Xmat]
            }
            false => Xmat,
        };

        let (intercept, coeffs) = match self.solve_normal_equation(&Xmat, &yvec, &wvec) {
            Ok(beta) => {
                if self.fit_intercept {
                    (beta[0], beta.slice(s![1..]).to_vec())
                } else {
                    (0.0, beta.to_vec())
                }
            }
            Err(_) => {
                let mean = yvec.mean().expect("Label vector has a valid mean.");
                let zeros = vec![0.0; nf];
                (mean, zeros)
            }
        };

        Box::new(LinearRegessionModel { intercept, coeffs, indices })
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegessionModel {
    intercept: f64,
    coeffs: Vec<f64>,
    indices: Vec<usize>,
}

impl Model<f64> for LinearRegessionModel {
    fn transform(&self, inputs: &[FeatureRow]) -> Box<dyn Prediction<f64>> {
        let result = inputs
            .iter()
            .map(|row| {
                self.intercept
                    + self
                        .indices
                        .iter()
                        .fold(0.0, |state, &idx| state + self.coeffs[idx] * row.real_at(idx))
            })
            .collect();

        Box::new(LinearRegressionPrediction { result })
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegressionPrediction {
    result: Vec<f64>,
}

impl Prediction<f64> for LinearRegressionPrediction {
    fn expected(&self) -> Vec<f64> {
        self.result.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::linear_training_data;

    use super::*;

    #[test]
    fn test_regression() {
        let ns = 10;
        let coeffs = &[1.0, 2.0, 3.0, 4.0];
        let data = linear_training_data(ns, coeffs, 5.0);

        let features: Vec<FeatureRow> = data.iter().map(|row| row.features.clone()).collect();
        let labels: Vec<_> = data.iter().map(|row| row.label).collect();

        let mut learner = LinearRegressionLearner::new(true, None);
        let model = learner.fit(&data);
        let output = model.transform(&features);
        let predicted = output.expected();

        let error: f64 = predicted.iter().zip(labels.iter()).map(|(p, y)| (p - y).abs()).sum();

        assert!(error < 1e-9, "Predictions are inaccurate");
    }

    #[test]
    fn test_underconstrained() {
        let ns = 3; // Only 3 samples
        let coeffs = &[1.0, 2.0, 3.0, 4.0]; // But 4 features + intercept
        let data = linear_training_data(ns, coeffs, 0.0);

        let features: Vec<_> = data.iter().map(|row| row.features.clone()).collect();
        let mean = data.iter().map(|row| row.label).sum::<f64>() / (ns as f64);

        let mut learner = LinearRegressionLearner::new(true, None);
        let model = learner.fit(&data);
        let output = model.transform(&features);
        let predicted = output.expected();

        predicted.iter().for_each(|p| assert!((p - mean).abs() < 1e-9))
    }
}
