#[macro_use]
extern crate criterion;
extern crate toro;

pub mod splitters;

criterion_main!(splitters::splitters);
