use crate::core::{Label, Model, TrainingRow};
use crate::trees::splits::Split;

#[derive(Clone, Debug)]
pub enum TrainingNode<L: Label> {
    Leaf {
        data: Vec<TrainingRow<L>>,
        depth: usize,
    },
    Internal {
        split: Split,
        left: Box<TrainingNode<L>>,
        right: Box<TrainingNode<L>>,
        delta: f64,
        depth: usize,
    },
}

impl<L: Label> TrainingNode<L> {
    pub fn leaf(data: Vec<TrainingRow<L>>, depth: usize) -> Self {
        TrainingNode::Leaf { data, depth }
    }

    pub fn internal(
        split: Split,
        left: Box<TrainingNode<L>>,
        right: Box<TrainingNode<L>>,
        delta: f64,
        depth: usize,
    ) -> Self {
        TrainingNode::Internal { split, left, right, delta, depth }
    }

    pub fn training_weight(&self) -> f64 {
        match self {
            Self::Leaf { data, .. } => data.iter().map(|row| row.weight).sum(),
            Self::Internal { left, right, .. } => left.training_weight() + right.training_weight(),
        }
    }

    pub fn build_model(&self, learner: &LeafLearner) -> ModelNode<L> {
        let weight = self.training_weight();
        match self {
            Self::Leaf { data, depth } => {
                todo!()
            }
            Self::Internal { split, left, right, depth, .. } => {
                let left_model = Box::new(left.build_model(learner));
                let right_model = Box::new(right.build_model(learner));

                ModelNode::internal(split.clone(), left_model, right_model, weight, *depth)
            }
        }
    }
}

#[derive(Debug)]
pub enum ModelNode<L: Label> {
    Leaf {
        model: Box<dyn Model<L>>,
        training_weight: f64,
        depth: usize,
    },
    Internal {
        split: Split,
        left: Box<ModelNode<L>>,
        right: Box<ModelNode<L>>,
        training_weight: f64,
        depth: usize,
    },
}

impl<L: Label> ModelNode<L> {
    pub fn leaf(model: Box<dyn Model<L>>, training_weight: f64, depth: usize) -> Self {
        ModelNode::Leaf { model, training_weight, depth }
    }

    pub fn internal(
        split: Split,
        left: Box<ModelNode<L>>,
        right: Box<ModelNode<L>>,
        training_weight: f64,
        depth: usize,
    ) -> Self {
        ModelNode::Internal { split, left, right, training_weight, depth }
    }
}
