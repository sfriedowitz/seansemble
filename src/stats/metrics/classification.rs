use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::DMatrix;

use super::{EvaluationMetric, PVA};

/// Construct a confusion matrix for the PVA data.
///
/// The confusion matrix is defined to have entry i, j equal to the number of observations
/// known to be in group i but predicted to be in group j.
pub fn confusion_matrix(pva: &PVA<usize>) -> DMatrix<usize> {
    // Get all unique labels
    let labels: Vec<usize> =
        pva.predicted.iter().chain(pva.actual.iter()).unique().copied().collect();

    let n = labels.len();
    let index: HashMap<usize, usize> = labels.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // Increment matrix elements for each pair in PVA
    // Sum down rows per column -> actual count
    // Sum across columns per row -> predicted count
    let mut confusion = DMatrix::zeros(n, n);
    for (pred, actual) in pva {
        let i = index[&pred];
        let j = index[&actual];
        confusion[(i, j)] += 1;
    }
    confusion
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Accuracy {}

impl EvaluationMetric<usize> for Accuracy {
    fn evaluate(&self, pva: &PVA<usize>) -> f64 {
        let mut correct = 0;
        for (pred, actual) in pva {
            if pred == actual {
                correct += 1;
            }
        }
        (correct as f64) / (pva.len() as f64)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Precision {}

impl EvaluationMetric<usize> for Precision {
    fn evaluate(&self, pva: &PVA<usize>) -> f64 {
        let confusion = confusion_matrix(pva);
        let predicted_counts: Vec<usize> = confusion.row_iter().map(|r| r.sum()).collect();

        let total: f64 = predicted_counts
            .iter()
            .enumerate()
            .map(
                |(i, &count)| {
                    if count > 0 {
                        confusion[(i, i)] as f64 / count as f64
                    } else {
                        1.0
                    }
                },
            )
            .sum();

        total / predicted_counts.len() as f64
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Recall {}

impl EvaluationMetric<usize> for Recall {
    fn evaluate(&self, pva: &PVA<usize>) -> f64 {
        let confusion = confusion_matrix(pva);
        let actual_counts: Vec<usize> = confusion.column_iter().map(|c| c.sum()).collect();

        let total: f64 = actual_counts
            .iter()
            .enumerate()
            .map(
                |(i, &count)| {
                    if count > 0 {
                        confusion[(i, i)] as f64 / count as f64
                    } else {
                        1.0
                    }
                },
            )
            .sum();

        total / actual_counts.len() as f64
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MacroF1 {}

impl EvaluationMetric<usize> for MacroF1 {
    fn evaluate(&self, pva: &PVA<usize>) -> f64 {
        let confusion = confusion_matrix(pva);

        let score_sum: f64 = (0..confusion.nrows())
            .map(|i| {
                let predicted_count = confusion.row(i).sum();
                let actual_count = confusion.column(i).sum();

                let precision = match predicted_count {
                    count if count > 0 => confusion[(i, i)] as f64 / count as f64,
                    _ => 1.0,
                };
                let recall = match actual_count {
                    count if count > 0 => confusion[(i, i)] as f64 / count as f64,
                    _ => 1.0,
                };

                if precision > 0.0 && recall > 0.0 {
                    (actual_count as f64) * 2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                }
            })
            .sum();

        score_sum / pva.len() as f64
    }
}

/// Enumeration of metrics on categorical-valued PVA data.
#[derive(Clone, Copy, Debug)]
pub enum ClassificationMetric {
    Accuracy(Accuracy),
    Precision(Precision),
    Recall(Recall),
    MacroF1(MacroF1),
}

impl EvaluationMetric<usize> for ClassificationMetric {
    fn evaluate(&self, pva: &PVA<usize>) -> f64 {
        match self {
            Self::Accuracy(metric) => metric.evaluate(pva),
            Self::Precision(metric) => metric.evaluate(pva),
            Self::Recall(metric) => metric.evaluate(pva),
            Self::MacroF1(metric) => metric.evaluate(pva),
        }
    }
}

impl From<Accuracy> for ClassificationMetric {
    fn from(metric: Accuracy) -> Self {
        Self::Accuracy(metric)
    }
}

impl From<Precision> for ClassificationMetric {
    fn from(metric: Precision) -> Self {
        Self::Precision(metric)
    }
}

impl From<Recall> for ClassificationMetric {
    fn from(metric: Recall) -> Self {
        Self::Recall(metric)
    }
}

impl From<MacroF1> for ClassificationMetric {
    fn from(metric: MacroF1) -> Self {
        Self::MacroF1(metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let y1 = vec![0, 1, 0, 1, 2, 3, 0, 1, 2, 3];
        let y2 = vec![0, 1, 4, 1, 2, 1, 0, 1, 4, 3];

        let pva_equal = PVA::new(y1.clone(), y1.clone());
        let confusion_equal = confusion_matrix(&pva_equal);
        assert!(confusion_equal.nrows() == 4); // Only 4 labels from [0..3]
        assert!(confusion_equal.diagonal().sum() == y1.len()); // Only diagonal elements present

        let pva_diff = PVA::new(y1.clone(), y2.clone());
        let confusion_diff = confusion_matrix(&pva_diff);
        assert!(confusion_diff.nrows() == 5); // Now 5 labels from [0..4]
        assert!(confusion_diff.diagonal().sum() < y1.len()); // Off diagonals -> lesser diagonal sum
    }
}
