use std::fmt::Debug;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnyValue {
    Real(f64),
    Categorical(usize),
}

impl AnyValue {
    pub fn as_real(&self) -> Option<f64> {
        if let AnyValue::Real(x) = self {
            Some(*x)
        } else {
            None
        }
    }

    pub fn as_categorical(&self) -> Option<usize> {
        if let AnyValue::Categorical(x) = self {
            Some(*x)
        } else {
            None
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self, AnyValue::Real(_))
    }

    pub fn is_categorcial(&self) -> bool {
        matches!(self, AnyValue::Categorical(_))
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
