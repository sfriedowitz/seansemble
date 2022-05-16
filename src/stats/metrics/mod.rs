mod classification;
mod pva;
mod regression;

pub use self::classification::{Accuracy, ClassificationMetric, Precision, Recall};
pub use self::pva::PVA;
pub use self::regression::RegressionMetric;

/// A method that evaluates performance of the PVA data.
pub trait EvaluationMetric<T> {
    fn evaluate(&self, pva: &PVA<T>) -> f64;
}
