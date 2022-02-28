use std::rc::Rc;

use super::{Label, TrainingRow};

pub struct TrainingData<L: Label> {
    rows: Vec<TrainingRow<L>>,
    total_weight: f64,
}

pub struct TrainingSlice<L: Label> {
    data: Rc<TrainingData<L>>,
    indices: Vec<usize>,
}
