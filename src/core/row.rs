use super::AnyValue;

#[derive(Clone, Debug)]
pub struct FeatureRow {
    pub data: Vec<AnyValue>,
}

impl FeatureRow {
    pub fn new(data: Vec<AnyValue>) -> Self {
        FeatureRow { data }
    }

    pub fn real_indices(&self) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_real())
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn categorical_indices(&self) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_categorcial())
            .map(|(idx, _)| idx)
            .collect()
    }
}

impl From<Vec<AnyValue>> for FeatureRow {
    fn from(data: Vec<AnyValue>) -> Self {
        FeatureRow::new(data)
    }
}

impl From<Vec<f64>> for FeatureRow {
    fn from(data: Vec<f64>) -> Self {
        FeatureRow::new(data.into_iter().map(|x| x.into()).collect())
    }
}

impl From<Vec<usize>> for FeatureRow {
    fn from(data: Vec<usize>) -> Self {
        FeatureRow::new(data.into_iter().map(|x| x.into()).collect())
    }
}

impl From<&[f64]> for FeatureRow {
    fn from(data: &[f64]) -> Self {
        FeatureRow::new(data.iter().map(|x| (*x).into()).collect())
    }
}

impl From<&[usize]> for FeatureRow {
    fn from(data: &[usize]) -> Self {
        FeatureRow::new(data.iter().map(|x| (*x).into()).collect())
    }
}

#[derive(Clone, Debug)]
pub struct TrainingRow<T> {
    pub features: FeatureRow,
    pub label: T,
    pub weight: Option<f64>,
}

impl<T> TrainingRow<T> {
    pub fn new(features: impl Into<FeatureRow>, label: T, weight: Option<f64>) -> Self {
        Self { features: features.into(), label, weight }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_feature_indices() {
        let mut data: Vec<AnyValue> = vec![1.0.into(), 2.0.into(), 3.0.into()];
        data.extend(&vec![1.into(), 2.into(), 3.into()]);
        let features: FeatureRow = data.into();

        assert!(features.real_indices() == vec![0, 1, 2]);
        assert!(features.categorical_indices() == vec![3, 4, 5]);
    }
}
