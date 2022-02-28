use std::fmt::Debug;

use crate::core::Label;

/// The predicted vs. actual response values
#[derive(Clone, Debug)]
pub struct PVA<L: Label> {
    pub predicted: Vec<L>,
    pub actual: Vec<L>,
}

impl<L: Label> PVA<L> {
    pub fn new(predicted: Vec<L>, actual: Vec<L>) -> Self {
        if predicted.len() != actual.len() {
            panic!("The vector sizes don't match: {} != {}", predicted.len(), actual.len(),)
        }
        Self { predicted, actual }
    }

    pub fn len(&self) -> usize {
        self.predicted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&'_ self) -> PVAIterator<'_, L> {
        PVAIterator { pva: self, index: 0 }
    }
}

/// An iterator over (pred, actual) values in a PVA container
pub struct PVAIterator<'a, L: Label> {
    pva: &'a PVA<L>,
    index: usize,
}

impl<'a, L: Label> Iterator for PVAIterator<'a, L> {
    type Item = (L, L);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.pva.len() {
            None
        } else {
            let predicted = self.pva.predicted[self.index];
            let actual = self.pva.actual[self.index];
            self.index += 1;
            Some((predicted, actual))
        }
    }
}

impl<'a, L: Label> IntoIterator for &'a PVA<L> {
    type Item = (L, L);
    type IntoIter = PVAIterator<'a, L>;

    fn into_iter(self) -> Self::IntoIter {
        PVAIterator { pva: self, index: 0 }
    }
}
