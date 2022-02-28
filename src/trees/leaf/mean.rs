use rand::prelude::{SeedableRng, SliceRandom, StdRng};
use std::collections::HashMap;

use crate::core::{FeatureRow, Label, Learner, Model, Prediction, TrainingRow};

/// A learner that calculates the mean of the labels
#[derive(Clone, Debug)]
pub struct GuessTheMeanLearner {
    rng: StdRng,
}

impl GuessTheMeanLearner {
    pub fn new(seed_rng: Option<&mut StdRng>) -> Self {
        let rng = match seed_rng {
            Some(r) => SeedableRng::from_rng(r).expect("Seeding RNG failed."),
            None => SeedableRng::from_entropy(),
        };
        Self { rng }
    }
}

impl Learner<f64> for GuessTheMeanLearner {
    fn fit(&mut self, data: &[TrainingRow<f64>]) -> Box<dyn Model<f64>> {
        let sums = data.iter().fold((0.0, 0.0), |(sum, weight), row| {
            (sum + row.weight * row.label, weight + row.weight)
        });
        Box::new(GuessTheMeanModel { mean: sums.0 / sums.1 })
    }
}

impl Learner<usize> for GuessTheMeanLearner {
    fn fit(&mut self, data: &[TrainingRow<usize>]) -> Box<dyn Model<usize>> {
        let mut labels: Vec<(usize, f64)> =
            data.iter().map(|row| (row.label, row.weight)).collect();
        labels.shuffle(&mut self.rng);

        let mut weight_sums = HashMap::new();
        for (label, weight) in labels {
            *weight_sums.entry(label).or_insert(0.0) += weight;
        }

        let mean_label = weight_sums
            .into_iter()
            .max_by(|(_, wa), (_, wb)| wa.partial_cmp(wb).unwrap())
            .map(|(k, _)| k)
            .unwrap();

        Box::new(GuessTheMeanModel { mean: mean_label })
    }
}

/// A model produced by a GuessTheMean learner
#[derive(Clone, Copy, Debug)]
pub struct GuessTheMeanModel<L: Label> {
    mean: L,
}

impl<L: Label> Model<L> for GuessTheMeanModel<L> {
    fn transform(&self, inputs: &[FeatureRow]) -> Box<dyn Prediction<L>> {
        Box::new(GuessTheMeanPrediction { result: vec![self.mean; inputs.len()] })
    }
}

/// A prediction result for a GuessTheMean learner
#[derive(Clone, Debug)]
pub struct GuessTheMeanPrediction<L: Label> {
    result: Vec<L>,
}

impl<L: Label> Prediction<L> for GuessTheMeanPrediction<L> {
    fn expected(&self) -> Vec<L> {
        self.result.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::random_training_data;

    use super::*;

    #[test]
    fn test_categorical() {
        let ns = 5;
        let mut data = random_training_data::<usize>(ns, 1, 1);
        let features: Vec<FeatureRow> = data.iter().map(|row| row.features.clone()).collect();

        // Modify data so class of first element is largest weight
        let weight_sum: f64 = data.iter().map(|row| row.weight).sum();

        let head = &data[0];
        let head_label = head.label;
        let adjusted_weight = head.weight + weight_sum;
        data[0] = TrainingRow::from_features(head.features.clone(), head.label, adjusted_weight);

        let mut learner = GuessTheMeanLearner::new(None);
        let model = learner.fit(&data);
        let output = model.transform(&features);
        let predicted = output.expected();

        predicted.into_iter().for_each(|p| assert!(p == head_label));
    }
}
