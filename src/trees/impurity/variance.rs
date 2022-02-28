use super::ImpurityCalculator;
use crate::core::TrainingRow;

#[derive(Clone, Debug)]
pub struct VarianceCalculator {
    total_sum: f64,
    total_sq_sum: f64,
    total_weight: f64,

    left_sum: f64,
    left_weight: f64,
}

impl VarianceCalculator {
    pub fn new(ts: f64, tsq: f64, tw: f64) -> Self {
        Self { total_sum: ts, total_sq_sum: tsq, total_weight: tw, left_sum: 0.0, left_weight: 0.0 }
    }

    pub fn from_labels(labels: &[f64], weights: &[f64]) -> Self {
        if !(labels.len() == weights.len()) {
            panic!("Labels and weights are not the same size.")
        }

        let mut ts = 0.0;
        let mut tsq = 0.0;
        let mut tw = 0.0;
        for (y, w) in labels.iter().zip(weights.iter()) {
            ts += w * y;
            tsq += w * y * y;
            tw += w;
        }

        Self::new(ts, tsq, tw)
    }

    pub fn from_training_data(data: &[TrainingRow<f64>]) -> Self {
        let (labels, weights): (Vec<_>, Vec<_>) =
            data.iter().map(|row| (row.label, row.weight)).unzip();
        Self::from_labels(labels.as_slice(), weights.as_slice())
    }
}

impl ImpurityCalculator<f64> for VarianceCalculator {
    fn add(&mut self, value: f64, weight: f64) {
        if !value.is_nan() && !weight.is_nan() {
            self.left_sum += weight * value;
            self.left_weight += weight;
        }
    }

    fn remove(&mut self, value: f64, weight: f64) {
        if !value.is_nan() && !weight.is_nan() {
            self.left_sum -= weight * value;
            self.left_weight -= weight;
        }
    }

    fn reset(&mut self) {
        self.left_sum = 0.0;
        self.left_weight = 0.0;
    }

    fn impurity(&self) -> f64 {
        let rs = self.total_sum - self.left_sum;
        let rw = self.total_weight - self.left_weight;
        if self.total_weight == 0.0 {
            0.0
        } else if rw == 0.0 || self.left_weight == 0.0 {
            self.total_sq_sum - self.total_sum * self.total_sum / self.total_weight
        } else {
            let (ls, lw) = (self.left_sum, self.left_weight);
            self.total_sq_sum - ls * ls / lw - rs * rs / rw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let calc = VarianceCalculator::from_labels(&[], &[]);
        assert!(calc.impurity() == 0.0);
    }
}
