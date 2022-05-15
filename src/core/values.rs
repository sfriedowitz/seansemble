use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, Sub},
};

use num::{Bounded, FromPrimitive, Num, NumCast, Zero};

pub trait Label:
    Copy
    + Num
    + NumCast
    + Zero
    + std::iter::Sum<Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + PartialOrd
    + AddAssign
    + Bounded
    + FromPrimitive
{
}

impl Label for f64 {}
impl Label for usize {}

#[derive(Clone, Copy, Debug)]
pub enum AnyValue {
    Null,
    Real(f64),
    Categorical(usize),
}

impl AnyValue {
    pub fn is_real(&self) -> bool {
        use AnyValue::*;
        match self {
            Real(_) => true,
            _ => false,
        }
    }

    pub fn is_categorcial(&self) -> bool {
        use AnyValue::*;
        match self {
            Categorical(_) => true,
            _ => false,
        }
    }

    pub fn as_real(&self) -> Option<f64> {
        use AnyValue::*;
        match self {
            Real(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_categorical(&self) -> Option<usize> {
        use AnyValue::*;
        match self {
            Categorical(x) => Some(*x),
            _ => None,
        }
    }
}

impl From<f64> for AnyValue {
    fn from(x: f64) -> Self {
        AnyValue::Real(x)
    }
}

impl From<usize> for AnyValue {
    fn from(x: usize) -> Self {
        AnyValue::Categorical(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_types() {
        assert!(AnyValue::from(1.0).is_real());
        assert!(AnyValue::from(3).is_categorcial());

        let x = AnyValue::from(1.0);
        assert!(x.as_categorical().is_none());

        let y = AnyValue::from(5);
        assert!(y.as_real().is_none());
    }
}
