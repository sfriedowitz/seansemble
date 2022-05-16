use super::{confusion_matrix, PVA};

#[derive(Clone, Copy, Debug)]
pub enum ClassificationMetric {
    Recall,
    Precision,
    Accuracy,
    MacroF1,
}

impl ClassificationMetric {
    pub fn calculate(&self, pva: &PVA<usize>) -> f64 {
        match self {
            Self::Accuracy => Self::calculate_accuracy(pva),
            Self::Precision => Self::calculate_precision(pva),
            Self::Recall => Self::calculate_recall(pva),
            Self::MacroF1 => Self::calculate_f1(pva),
        }
    }

    fn calculate_accuracy(pva: &PVA<usize>) -> f64 {
        let mut correct = 0;
        for (pred, actual) in pva {
            if pred == actual {
                correct += 1;
            }
        }
        (correct as f64) / (pva.len() as f64)
    }

    fn calculate_precision(pva: &PVA<usize>) -> f64 {
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

    fn calculate_recall(pva: &PVA<usize>) -> f64 {
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

    fn calculate_f1(pva: &PVA<usize>) -> f64 {
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
