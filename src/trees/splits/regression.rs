use std::collections::HashSet;

use float_cmp::approx_eq;
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::{Split, Splitter};
use crate::core::TrainingRow;
use crate::trees::impurity::{ImpurityCalculator, VarianceCalculator};

#[derive(Clone, Debug)]
pub struct RegressionSplitter {
    randomize_pivot: bool,
    rng: StdRng,
}

impl RegressionSplitter {
    pub fn new(randomize_pivot: bool, rng: Option<&mut StdRng>) -> Self {
        let new_rng = match rng {
            Some(r) => SeedableRng::from_rng(r).expect("Seeding RNG failed."),
            None => SeedableRng::from_entropy(),
        };
        Self { randomize_pivot, rng: new_rng }
    }

    /// Find the best split on a continuous feature.
    fn best_real_split(
        &mut self,
        data: &[TrainingRow<f64>],
        calc: &mut VarianceCalculator,
        idx: usize,
        min_count: usize,
    ) -> (Split, f64) {
        // Pull out the feature to consider and sort by it
        let mut thin_data: Vec<(f64, f64, f64)> =
            data.iter().map(|row| (row.features.real_at(idx), row.label, row.weight)).collect();
        thin_data.sort_by(|(a, _, _), (b, _, _)| a.partial_cmp(b).unwrap());

        // Best cases for iteration
        let mut best_variance = f64::INFINITY;
        let mut best_pivot = f64::INFINITY;

        // Move the data from right to left partition one value at a time
        calc.reset();
        let jmax = data.len() - min_count;
        for j in 0..jmax {
            calc.add(thin_data[j].1, thin_data[j].2);
            let total_variance = calc.impurity();

            // Keep track of the best split, avoiding splits in the middle of constant features
            let left = thin_data[j + 1].0;
            let right = thin_data[j].0;
            let lr_equal = approx_eq!(f64, left, right, epsilon = 1e-10);
            if total_variance < best_variance && j + 1 >= min_count && !lr_equal {
                best_variance = total_variance;
                best_pivot = match self.randomize_pivot {
                    true => right + (left - right) * self.rng.gen::<f64>(),
                    false => 0.5 * (left + right),
                }
            }
        }

        (Split::Real(idx, best_pivot), best_variance)
    }

    /// Find the best split on a categorical variable.
    fn best_categorical_split(
        &mut self,
        data: &[TrainingRow<f64>],
        calc: &mut VarianceCalculator,
        idx: usize,
        min_count: usize,
    ) -> (Split, f64) {
        let thin_data: Vec<(usize, f64, f64)> = data
            .iter()
            .map(|row| (row.features.categorical_at(idx), row.label, row.weight))
            .collect();
        let total_weight: f64 = thin_data.iter().fold(0.0, |state, (_, _, w)| state + w);

        // Group the data by categorical feature
        struct CategoryAvg {
            category: usize,
            label_avg: f64,
            weight: f64,
            size: usize,
        }

        // TODO: This is incorrect b/c group_by is consecutive. Rewrite with manual loop and group
        let mut category_averages: Vec<CategoryAvg> = thin_data
            .iter()
            .group_by(|(category, _, _)| category)
            .into_iter()
            .map(|(&category, groups)| {
                let mut label_sum = 0.0;
                let mut weight = 0.0;
                let mut size = 0;
                for g in groups {
                    label_sum += g.1 * g.2;
                    weight += g.2;
                    size += 1;
                }
                CategoryAvg { category, label_avg: label_sum / weight, weight, size }
            })
            .collect();

        // If too many values are trivial, return an empty split
        let non_trivial: f64 =
            category_averages.iter().map(|avg| if avg.size > 1 { avg.weight } else { 0.0 }).sum();
        if non_trivial / total_weight < 0.5 {
            return (Split::Categorical(idx, HashSet::new()), f64::INFINITY);
        }

        // Best cases for iteration
        let mut left_num: usize = 0;
        let mut best_variance = f64::INFINITY;
        let mut best_set: HashSet<usize> = HashSet::new();

        // Sort by ascending label avg per category
        category_averages.sort_by(|c1, c2| c1.label_avg.partial_cmp(&c2.label_avg).unwrap());

        // Add categories one at a time in order of avg label
        calc.reset();
        for j in 0..(category_averages.len() - 1) {
            let avg = &category_averages[j];
            left_num += avg.size;

            calc.add(avg.label_avg, avg.weight);
            let total_variance = calc.impurity();

            if total_variance < best_variance
                && left_num >= min_count
                && (thin_data.len() - left_num) >= min_count
            {
                best_variance = total_variance;
                best_set = category_averages[..(j + 1)].iter().map(|avg| avg.category).collect();
            }
        }

        (Split::Categorical(idx, best_set), best_variance)
    }
}

impl Splitter<f64> for RegressionSplitter {
    fn find_best_split(
        &mut self,
        data: &[TrainingRow<f64>],
        nfeatures: usize,
        min_count: usize,
    ) -> (Split, f64) {
        let mut calc = VarianceCalculator::from_training_data(data);
        let init_variance = calc.impurity();

        let mut best_split = Split::None;
        let mut best_variance = f64::INFINITY;

        let rep = &data[0];
        let (nf, nr) = (rep.features.num_features(), rep.features.num_reals());
        let mut indices: Vec<usize> = (0..nf).collect();
        indices.shuffle(&mut self.rng);

        for index in indices.into_iter().take(nfeatures) {
            let (trial_split, trial_variance): (Split, f64) = match index {
                idx if idx < nr => self.best_real_split(data, &mut calc, idx, min_count),
                idx => self.best_categorical_split(data, &mut calc, idx - nr, min_count),
            };

            if trial_variance < best_variance {
                best_variance = trial_variance;
                best_split = trial_split;
            }
        }

        if best_variance.is_infinite() {
            (Split::None, 0.0)
        } else {
            let delta_impurity = init_variance - best_variance;
            (best_split, delta_impurity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FeatureRow;

    #[test]
    fn split_real() {
        let data = vec![
            TrainingRow::new(vec![1.0], vec![], 1.0, 1.0),
            TrainingRow::new(vec![2.0], vec![], 2.0, 1.0),
        ];

        let mut splitter = RegressionSplitter::new(false, None);
        let (split, _) = splitter.find_best_split(&data, 10, 1);

        let row1 = FeatureRow::new(vec![1.49], vec![]);
        assert!(split.turn_left(&row1));

        let row2 = FeatureRow::new(vec![1.51], vec![]);
        assert!(!split.turn_left(&row2));
    }
}
