use super::{values::Label, AnyValue};

#[derive(Clone, Debug)]
pub struct TrainingRow<T: Label> {
    pub features: Vec<AnyValue>,
    pub label: T,
    pub weight: Option<f64>,
}

impl<T: Label> TrainingRow<T> {
    pub fn new(features: Vec<AnyValue>, label: T, weight: Option<f64>) -> Self {
        Self { features, label, weight }
    }
}

pub trait TrainingData {
    fn num_rows(&self) -> usize;

    fn total_weight(&self) -> f64;
}

pub struct RegressionTrainingData {
    rows: Vec<TrainingRow<f64>>,
}

impl TrainingData for RegressionTrainingData {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn total_weight(&self) -> f64 {
        self.rows.iter().map(|r| r.weight.unwrap_or(1.0)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_indices() {
        let mut features: Vec<AnyValue> = vec![1.0.into(), 2.0.into(), 3.0.into()];
        features.extend(&vec![1.into(), 2.into(), 3.into()]);

        assert!(features[2].is_real());
        assert!(features[3].is_categorcial());
    }
}
