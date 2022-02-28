use std::rc::Rc;

use super::{Label, TrainingRow};

#[derive(Clone, Debug)]
pub struct TrainingData<L: Label> {
    rows: Vec<TrainingRow<L>>,
    total_weight: f64,
}

impl<L: Label> TrainingData<L> {
    pub fn new(rows: Vec<TrainingRow<L>>) -> Self {
        let total_weight = rows.iter().fold(0.0, |sum, row| sum + row.weight);
        Self { rows, total_weight }
    }

    pub fn slice_at(&self, indices: Vec<usize>) -> TrainingSlice<L> {
        let max_idx = indices.iter().max().cloned().unwrap_or(0);
        if max_idx >= self.len() {
            panic!("Index out of bounds for training slice.");
        }

        todo!()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
}

pub struct TrainingSlice<L: Label> {
    data: Rc<TrainingData<L>>,
    indices: Vec<usize>,
}

impl<L: Label> TrainingSlice<L> {
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}
