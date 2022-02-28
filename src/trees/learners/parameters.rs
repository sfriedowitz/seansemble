/// Hyperparameters for a decision tree learner.
#[derive(Debug, Copy, Clone)]
pub struct DecisionTreeParameters {
    pub max_depth: usize,
    pub num_features: usize,
    pub min_leaf_instances: usize,
    pub min_impurity_decrease: f64,
}

impl DecisionTreeParameters {
    /// The maximum depth of the tree.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// The numbers of features to consider at each split.
    pub fn with_num_features(mut self, num_features: usize) -> Self {
        self.num_features = num_features;
        self
    }

    /// The minimum number of samples required to be at a leaf node.
    pub fn with_min_leaf_instances(mut self, min_leaf_instances: usize) -> Self {
        self.min_leaf_instances = min_leaf_instances;
        self
    }

    /// The minimum change in impurity for splitting an internal node.
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease;
        self
    }
}

impl Default for DecisionTreeParameters {
    fn default() -> Self {
        Self {
            max_depth: 30,
            num_features: usize::MAX,
            min_leaf_instances: 2,
            min_impurity_decrease: 0.0,
        }
    }
}
