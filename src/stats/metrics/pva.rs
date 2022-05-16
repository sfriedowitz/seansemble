use std::fmt::Debug;

/// The predicted vs. actual response values
#[derive(Clone, Debug)]
pub struct PVA<T> {
    pub predicted: Vec<T>,
    pub actual: Vec<T>,
}

impl<T> PVA<T> {
    pub fn new(predicted: Vec<T>, actual: Vec<T>) -> Self {
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

    pub fn iter(&'_ self) -> PVAIterator<'_, T> {
        PVAIterator { pva: self, index: 0 }
    }
}

/// An iterator over (pred, actual) values in a PVA container
pub struct PVAIterator<'a, T> {
    pva: &'a PVA<T>,
    index: usize,
}

impl<'a, T> Iterator for PVAIterator<'a, T> {
    type Item = (&'a T, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.pva.len() {
            None
        } else {
            let predicted = &self.pva.predicted[self.index];
            let actual = &self.pva.actual[self.index];
            self.index += 1;
            Some((predicted, actual))
        }
    }
}

impl<'a, T> IntoIterator for &'a PVA<T> {
    type Item = (&'a T, &'a T);
    type IntoIter = PVAIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        PVAIterator { pva: self, index: 0 }
    }
}
