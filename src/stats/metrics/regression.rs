use super::{EvaluationMetric, PVA};

fn evaluate_error(pva: &PVA<f64>, metric: impl Fn(f64, f64) -> f64) -> f64 {
    match pva.len() {
        0 => 0.0,
        n => pva.iter().fold(0.0, |sum, (pred, actual)| sum + metric(*pred, *actual)) / (n as f64),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MSE {}

impl EvaluationMetric<f64> for MSE {
    fn evaluate(&self, pva: &PVA<f64>) -> f64 {
        evaluate_error(pva, |x, y| (x - y).powf(2.0))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MAE {}

impl EvaluationMetric<f64> for MAE {
    fn evaluate(&self, pva: &PVA<f64>) -> f64 {
        evaluate_error(pva, |x, y| (x - y).abs())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct R2 {}

impl EvaluationMetric<f64> for R2 {
    fn evaluate(&self, pva: &PVA<f64>) -> f64 {
        let n = pva.len();
        if n == 0 {
            return 0.0;
        }

        let mean = pva.actual.iter().sum::<f64>() / (n as f64);
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for (pred, actual) in pva {
            ss_tot += (actual - mean).powf(2.0);
            ss_res += (actual - pred).powf(2.0);
        }

        1.0 - ss_res / ss_tot
    }
}

/// Enumeration of metrics on real-valued PVA data.
#[derive(Clone, Copy, Debug)]
pub enum RegressionMetric {
    MSE(MSE),
    MAE(MAE),
    R2(R2),
}

impl EvaluationMetric<f64> for RegressionMetric {
    fn evaluate(&self, pva: &PVA<f64>) -> f64 {
        match self {
            Self::MSE(metric) => metric.evaluate(pva),
            Self::MAE(metric) => metric.evaluate(pva),
            Self::R2(metric) => metric.evaluate(pva),
        }
    }
}

impl From<MSE> for RegressionMetric {
    fn from(metric: MSE) -> Self {
        Self::MSE(metric)
    }
}

impl From<MAE> for RegressionMetric {
    fn from(metric: MAE) -> Self {
        Self::MAE(metric)
    }
}

impl From<R2> for RegressionMetric {
    fn from(metric: R2) -> Self {
        Self::R2(metric)
    }
}
