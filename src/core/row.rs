use std::os::linux::raw;

use super::AnyValue;

#[derive(Clone, Debug)]
pub struct TrainingRow<T> {
    pub features: Vec<AnyValue>,
    pub label: T,
    pub weight: Option<f64>,
}

impl<T> TrainingRow<T> {
    pub fn new(raw_features: Vec<impl Into<AnyValue>>, label: T, weight: Option<f64>) -> Self {
        let features = raw_features.into_iter().map(|x| x.into()).collect();
        Self { features, label, weight }
    }

    pub fn real_features(&self) -> Vec<usize> {
        self.features
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_real())
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn categorical_features(&self) -> Vec<usize> {
        self.features
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_categorcial())
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_feature_indices() {
        let mut features: Vec<AnyValue> = vec![1.0.into(), 2.0.into(), 3.0.into()];
        features.extend(&vec![1.into(), 2.into(), 3.into()]);
        let row = TrainingRow::new(features, 1.0, Some(5.0));

        assert!(row.real_features() == vec![0, 1, 2]);
        assert!(row.categorical_features() == vec![3, 4, 5]);

        let other = vec![1.0, 2.0, 3.0];
        let this = TrainingRow::new(other, 1.0, None);
        dbg!(this);
    }
}
