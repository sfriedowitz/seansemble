use std::borrow::Cow;

use thiserror::Error;

pub type Result<T> = ::std::result::Result<T, ModelingError>;

type ErrString = Cow<'static, str>;

#[non_exhaustive]
#[derive(Clone, Debug, Error)]
pub enum ModelingError {
    #[error("FitError: {0}")]
    FitError(ErrString),
    #[error("PredictError: {0}")]
    PredictError(ErrString),
    #[error("TransformError: {0}")]
    TransformError(ErrString),
    #[error("SolutionError: {0}")]
    SolutionError(ErrString),
    #[error("GenericError: {0}")]
    Generic(ErrString),
}
