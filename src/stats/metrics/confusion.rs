use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::DMatrix;

use super::PVA;

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
