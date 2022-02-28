use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct CategoricalEncoder<T: Hash + Eq + Clone> {
    encoding: HashMap<T, usize>,
    decoding: HashMap<usize, T>,
}

impl<T: Hash + Eq + Clone> CategoricalEncoder<T> {
    pub fn new(values: &[T]) -> Self {
        let encoding: HashMap<T, usize> =
            values.iter().unique().enumerate().map(|(i, x)| (x.clone(), i + 1)).collect();

        let decoding: HashMap<usize, T> = encoding.iter().map(|(x, i)| (*i, x.clone())).collect();

        CategoricalEncoder { encoding, decoding }
    }

    /// Encode an input to a usize value, returning 0 if the input is not present.
    pub fn encode(&self, input: &T) -> usize {
        self.encoding.get(input).cloned().unwrap_or(0)
    }

    /// Decode a usize value to return the (optional) input it corresponds to.
    pub fn decode(&self, code: usize) -> Option<&T> {
        self.decoding.get(&code)
    }
}

impl<T: Hash + Eq + Clone> FromIterator<T> for CategoricalEncoder<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values: Vec<T> = iter.into_iter().collect();
        CategoricalEncoder::new(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::CategoricalEncoder;

    #[test]
    fn encoding_round_trip() {
        let values = vec!["dog", "cat", "pig"];
        let encoder = CategoricalEncoder::new(&values);

        // Encoding unknown value should return zero
        assert_eq!(encoder.encode(&"chicken"), 0);
        assert_eq!(encoder.encode(&"mew"), 0);

        // Test round trip
        for (i, val) in values.iter().enumerate() {
            let code = i + 1;
            let e = encoder.encode(&val);
            assert_eq!(e, code);

            let d = encoder.decode(code).unwrap();
            assert_eq!(val, d);
        }
    }
}
