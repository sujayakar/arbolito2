#![allow(unused)]
#![feature(portable_simd, stdsimd)]

#[cfg(test)]
mod model;

#[cfg(test)]
mod scalar;

#[cfg(all(test, target_arch = "aarch64"))]
mod simd;

mod transpose;

#[cfg(test)]
mod simd_loop;
