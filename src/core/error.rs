use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug)]
pub enum FailureType {
    Fit,
    Predict,
    Transform,
}

#[derive(Clone, Debug)]
pub struct ModelingError {
    pub err: FailureType,
    pub msg: String,
}

impl ModelingError {
    pub fn from_fit<S: Into<String>>(msg: S) -> Self {
        Self { err: FailureType::Fit, msg: msg.into() }
    }

    pub fn from_predict<S: Into<String>>(msg: S) -> Self {
        Self { err: FailureType::Predict, msg: msg.into() }
    }

    pub fn from_transform<S: Into<String>>(msg: S) -> Self {
        Self { err: FailureType::Transform, msg: msg.into() }
    }

    pub fn because<S: Into<String>>(err: FailureType, msg: S) -> Self {
        Self { err, msg: msg.into() }
    }
}

impl fmt::Display for ModelingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.err {
            FailureType::Fit => "Fitting Error",
            FailureType::Predict => "Prediction Error",
            FailureType::Transform => "Transform Error",
        };
        write!(f, "{}: {}", prefix, self.msg)
    }
}

impl Error for ModelingError {}
