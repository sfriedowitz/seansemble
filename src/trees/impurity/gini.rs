use std::collections::HashMap;

use super::ImpurityCalculator;
use crate::core::TrainingRow;

#[derive(Clone, Debug)]
pub struct GiniCalculator {
    total_categories: Vec<f64>,
    total_sq_sum: f64,
    total_weight: f64,

    left_categories: Vec<f64>,
    left_weight: f64,
    left_sq_sum: f64,
    right_sq_sum: f64,
}

impl GiniCalculator {
    pub fn new(tcw: Vec<f64>, tsq: f64, tw: f64) -> Self {
        let lcw = vec![0.0; tcw.len()];
        Self {
            total_categories: tcw,
            total_sq_sum: tsq,
            total_weight: tw,
            left_categories: lcw,
            left_weight: 0.0,
            left_sq_sum: 0.0,
            right_sq_sum: tsq,
        }
    }

    pub fn from_labels(labels: &[usize], weights: &[f64]) -> Self {
        if !(labels.len() == weights.len()) {
            panic!("Labels and weights are not the same size.")
        }

        let mut category_weights: HashMap<usize, f64> = HashMap::new();
        for (label, weight) in labels.iter().zip(weights.iter()) {
            *category_weights.entry(*label).or_insert(0.0) += weight;
        }

        if category_weights.is_empty() {
            return GiniCalculator::new(vec![], 0.0, 0.0);
        }

        let max_category = *category_weights.keys().max().unwrap();

        let mut weight_vec = Vec::with_capacity(max_category);
        let mut total_weight = 0.0;
        let mut total_sq_sum = 0.0;
        for (cat, weight) in category_weights {
            weight_vec[cat] = weight;
            total_weight += weight;
            total_sq_sum += weight * weight;
        }

        GiniCalculator::new(weight_vec, total_sq_sum, total_weight)
    }

    pub fn from_training_data(data: &Vec<TrainingRow<usize>>) -> GiniCalculator {
        let (labels, weights): (Vec<_>, Vec<_>) =
            data.iter().map(|row| (row.label, row.weight)).unzip();
        GiniCalculator::from_labels(&labels, &weights)
    }
}

impl ImpurityCalculator<usize> for GiniCalculator {
    fn add(&mut self, value: usize, weight: f64) {
        if value > 0 {
            let wl = self.left_categories[value];
            self.left_categories[value] = wl + weight;
            self.left_sq_sum += weight * (weight + 2.0 * wl);
            self.left_weight += weight;

            let wr = self.total_categories[value] - wl;
            self.right_sq_sum += weight * (weight - 2.0 * wr);
        }
    }

    fn remove(&mut self, value: usize, weight: f64) {
        if value > 0 {
            let wl = self.left_categories[value];
            self.left_categories[value] = wl - weight;
            self.left_sq_sum += weight * (weight - 2.0 * wl);
            self.left_weight -= weight;

            let wr = self.total_categories[value] - wl;
            self.right_sq_sum += weight * (weight + 2.0 * wr);
        }
    }

    fn reset(&mut self) {
        self.left_categories.fill(0.0);
        self.left_weight = 0.0;
        self.left_sq_sum = 0.0;
        self.right_sq_sum = self.total_sq_sum;
    }

    fn impurity(&self) -> f64 {
        if self.total_weight == 0.0 {
            0.0
        } else if self.left_sq_sum == 0.0 && self.right_sq_sum == 0.0 {
            self.total_weight - self.total_sq_sum / self.total_weight
        } else {
            self.total_weight
                - self.left_sq_sum / self.left_weight
                - self.right_sq_sum / (self.total_weight - self.left_weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let calc = GiniCalculator::from_labels(&[], &[]);
        assert!(calc.impurity() == 0.0);
    }
}
