use crate::core::FeatureRow;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub enum Split {
    None,
    Real(usize, f64),
    Categorical(usize, HashSet<usize>),
}

impl Split {
    /// The index of the data row used to obtain this split
    pub fn index(&self) -> usize {
        match self {
            Self::Real(index, _) => *index,
            Self::Categorical(index, _) => *index,
            Self::None => usize::MAX,
        }
    }

    /// Should a new row turn left at this split?
    pub fn turn_left(&self, input: &FeatureRow) -> bool {
        if let Self::Real(index, pivot) = self {
            let rv = input.real_at(*index);
            if pivot.is_nan() {
                !rv.is_nan()
            } else {
                rv <= *pivot
            }
        } else if let Self::Categorical(index, included) = self {
            included.contains(&input.categorical_at(*index))
        } else {
            false
        }
    }
}
