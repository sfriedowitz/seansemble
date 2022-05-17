use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

use crate::core::Result;
use crate::core::{FeatureRow, Learner, Model, Prediction, TrainingRow};

/// A learner that calculates the mean of the labels
#[derive(Clone, Copy, Debug, Default)]
pub struct GuessTheMeanLearner {}

impl Learner<f64> for GuessTheMeanLearner {
    fn fit(&self, data: &[TrainingRow<f64>], _rng: &mut impl Rng) -> Result<Box<dyn Model<f64>>> {
        let sums = data.iter().fold((0.0, 0.0), |(sum, weight), row| {
            let row_weight = row.weight.unwrap_or(1.0);
            (sum + row_weight * row.label, weight + row_weight)
        });
        Ok(Box::new(GuessTheMeanModel { mean: sums.0 / sums.1 }))
    }
}

impl Learner<usize> for GuessTheMeanLearner {
    fn fit(
        &self,
        data: &[TrainingRow<usize>],
        rng: &mut impl Rng,
    ) -> Result<Box<dyn Model<usize>>> {
        let mut labels: Vec<(usize, f64)> =
            data.iter().map(|row| (row.label, row.weight.unwrap_or(1.0))).collect();
        labels.shuffle(rng);

        let mut weight_sums = HashMap::new();
        for (label, weight) in labels {
            *weight_sums.entry(label).or_insert(0.0) += weight;
        }

        let mean_label = weight_sums
            .into_iter()
            .max_by(|(_, wa), (_, wb)| wa.partial_cmp(wb).unwrap())
            .map(|(k, _)| k)
            .unwrap();

        Ok(Box::new(GuessTheMeanModel { mean: mean_label }))
    }
}

/// A model produced by a GuessTheMean learner
#[derive(Clone, Copy, Debug)]
pub struct GuessTheMeanModel<T> {
    mean: T,
}

impl Model<f64> for GuessTheMeanModel<f64> {
    fn transform(&self, inputs: &[FeatureRow]) -> Result<Box<dyn Prediction<f64>>> {
        Ok(Box::new(GuessTheMeanPrediction { result: vec![self.mean; inputs.len()] }))
    }

    fn loss(&self) -> Option<f64> {
        None
    }
}

impl Model<usize> for GuessTheMeanModel<usize> {
    fn transform(&self, inputs: &[FeatureRow]) -> Result<Box<dyn Prediction<usize>>> {
        Ok(Box::new(GuessTheMeanPrediction { result: vec![self.mean; inputs.len()] }))
    }

    fn loss(&self) -> Option<f64> {
        None
    }
}

/// A prediction result for a GuessTheMean learner
#[derive(Clone, Debug)]
pub struct GuessTheMeanPrediction<T> {
    result: Vec<T>,
}

impl Prediction<f64> for GuessTheMeanPrediction<f64> {
    fn expected(&self) -> Vec<f64> {
        self.result.clone()
    }
}

impl Prediction<usize> for GuessTheMeanPrediction<usize> {
    fn expected(&self) -> Vec<usize> {
        self.result.clone()
    }
}

#[cfg(test)]
mod tests {
    // use crate::utils::random_training_data;

    // use super::*;

    // #[test]
    // fn test_categorical() {
    //     let ns = 5;
    //     let mut data = random_training_data::<usize>(ns, 1, 1);
    //     let features: Vec<FeatureRow> = data.iter().map(|row| row.features.clone()).collect();

    //     // Modify data so class of first element is largest weight
    //     let weight_sum: f64 = data.iter().map(|row| row.weight).sum();

    //     let head = &data[0];
    //     let head_label = head.label;
    //     let adjusted_weight = head.weight + weight_sum;
    //     data[0] = TrainingRow::from_features(head.features.clone(), head.label, adjusted_weight);

    //     let mut learner = GuessTheMeanLearner::new(None);
    //     let model = learner.fit(&data);
    //     let output = model.transform(&features);
    //     let predicted = output.expected();

    //     predicted.into_iter().for_each(|p| assert!(p == head_label));
    // }
}
