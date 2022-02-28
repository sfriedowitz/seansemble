use std::borrow::Cow;

use thiserror::Error as ThisError;

type ErrString = Cow<'static, str>;
pub type Result<T> = ::std::result::Result<T, ModelingError>;

#[derive(Clone, Debug, ThisError)]
#[non_exhaustive]
pub enum ModelingError {
    #[error("FitError: {0}")]
    FitError(ErrString),
    #[error("PredictError: {0}")]
    PredictError(ErrString),
    #[error("TransformError: {0}")]
    TransformError(ErrString),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_state() {
        let e = ModelingError::FitError("Not enough data to train model.".into());
        println!("{}", e);
    }
}
