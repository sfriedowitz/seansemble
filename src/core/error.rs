use std::borrow::Cow;

use thiserror::Error as ThisError;

pub type Result<T> = ::std::result::Result<T, ModelingError>;

type ErrString = Cow<'static, str>;

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
