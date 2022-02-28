mod api;
mod data;
mod error;
mod row;
mod types;

pub use self::api::{Learner, Model, Prediction};
pub use self::data::{TrainingData, TrainingSlice};
pub use self::error::ModelingError;
pub use self::row::{FeatureRow, TrainingRow};
pub use self::types::Label;
