use std::ops::Range;

use super::Label;

#[derive(Clone, Debug)]
pub struct FeatureRow {
    reals: Vec<f64>,
    categoricals: Vec<usize>,
}

impl FeatureRow {
    pub fn new(reals: Vec<f64>, categoricals: Vec<usize>) -> Self {
        Self { reals, categoricals }
    }

    pub fn reals(&self) -> &[f64] {
        &self.reals
    }

    pub fn categoricals(&self) -> &[usize] {
        &self.categoricals
    }

    pub fn num_reals(&self) -> usize {
        self.reals().len()
    }

    pub fn num_categoricals(&self) -> usize {
        self.categoricals().len()
    }

    pub fn num_features(&self) -> usize {
        self.num_reals() + self.num_categoricals()
    }

    pub fn real_indices(&self) -> Range<usize> {
        0..self.num_reals()
    }

    pub fn categorical_indices(&self) -> Range<usize> {
        0..self.num_categoricals()
    }

    pub fn real_at(&self, idx: usize) -> f64 {
        if !(self.real_indices().contains(&idx)) {
            panic!("Invalid index '{}' for real feature.", idx);
        };
        self.reals[idx]
    }

    pub fn categorical_at(&self, idx: usize) -> usize {
        if !(self.categorical_indices().contains(&idx)) {
            panic!("Invalid index '{}' for categorical feature.", idx);
        };
        self.categoricals[idx]
    }
}

#[derive(Clone, Debug)]
pub struct TrainingRow<L: Label> {
    pub features: FeatureRow,
    pub label: L,
    pub weight: f64,
}

impl<L: Label> TrainingRow<L> {
    pub fn new(reals: Vec<f64>, categoricals: Vec<usize>, label: L, weight: f64) -> Self {
        Self { features: FeatureRow::new(reals, categoricals), label, weight }
    }

    pub fn from_features(features: FeatureRow, label: L, weight: f64) -> Self {
        Self { features, label, weight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_indices() {
        let reals = vec![1.0, 2.0, 3.0];
        let cats = vec![1, 2, 3];
        let row = FeatureRow::new(reals, cats);

        assert!(row.real_at(2) == 3.0);
        assert!(row.categorical_at(2) == 3);

        // Suppress panic output below
        std::panic::set_hook(Box::new(|_| {}));

        let real_panic = std::panic::catch_unwind(|| row.real_at(3));
        assert!(real_panic.is_err());

        let cat_panic = std::panic::catch_unwind(|| row.categorical_at(3));
        assert!(cat_panic.is_err());
    }
}
