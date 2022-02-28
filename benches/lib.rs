#[macro_use]
extern crate criterion;
extern crate loro;

pub mod splitters;

criterion_main!(splitters::splitters);
