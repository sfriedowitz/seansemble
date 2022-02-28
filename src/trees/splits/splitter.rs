use std::fmt::Debug;

use super::Split;
use crate::core::{Label, TrainingRow};

pub trait Splitter<L: Label>: Debug {
    ///  Get the best split, considering num_features random features (w/o replacement)
    fn find_best_split(
        &mut self,
        data: &[TrainingRow<L>],
        num_features: usize,
        min_count: usize,
    ) -> (Split, f64);
}
