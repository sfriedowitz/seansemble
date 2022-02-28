use std::{
    fmt::{Debug, Display},
    ops::Range,
};

use num_traits::NumCast;
use rand::Rng;

/// Trait bound for the two types used as labels.
///
/// `f64` for real-valued labels, and `usize` for categorical labels.
pub trait Label: 'static + Debug + Display + Copy + NumCast {
    fn gen_range(range: Range<Self>) -> Self;
}

impl Label for usize {
    fn gen_range(range: Range<usize>) -> usize {
        rand::thread_rng().gen_range(range)
    }
}

impl Label for f64 {
    fn gen_range(range: Range<f64>) -> f64 {
        rand::thread_rng().gen_range(range)
    }
}
