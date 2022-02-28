/// Interface for calculating the impurity during a split
pub trait ImpurityCalculator<T> {
    fn add(&mut self, value: T, weight: f64);

    fn remove(&mut self, value: T, weight: f64);

    fn reset(&mut self);

    fn impurity(&self) -> f64;
}
