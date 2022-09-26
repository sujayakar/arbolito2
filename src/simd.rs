use std::simd::u8x16;
use std::simd::{SimdPartialEq, ToBitMask};
use std::arch::aarch64::{
    uint8x8_t,
    uint8x8x2_t,
    vtbl2_u8,
};
use crate::model::{LeafType, Tree16, Tree16Node, QueryResult, U4};
use crate::scalar::{NodeType, ScalarTree16};
use proptest::prelude::*;

pub struct SimdTree16 {
    // 0000 is an invalid bit pattern (leaves must either have values or pointers)
    // [ under_root ] [ is_branch ] [ has_value ] [ is_ptr ] [ 4 bit parent ]
    nodes: u8x16,
    // [ 4 bits unused ] [ 4-bit label ]
    labels: u8x16,
}

impl From<ScalarTree16> for SimdTree16 {
    fn from(t: ScalarTree16) -> Self {
        let mut nodes = [0u8; 16];
        let mut labels = [0u8; 16];

        assert!(t.nodes.len() <= 16);
        for (i, node) in t.nodes.into_iter().enumerate() {
            if let Some(parent) = node.parent {
                nodes[i] = parent.into();
            }
            if let NodeType::Leaf(LeafType::Pointer { .. }) = node.node_type {
                nodes[i] |= 1 << 4;
            }
            if node.node_type.has_value() {
                nodes[i] |= 1 << 5;
            }
            if let NodeType::Branch { .. } = node.node_type {
                nodes[i] |= 1 << 6;
            }
            if node.parent.is_none() {
                nodes[i] |= 1 << 7;
            }
            labels[i] = node.label.into();
        }
        Self {
            nodes: nodes.into(),
            labels: labels.into(),
        }
    }
}

// Hmm, PSHUFB/TBL aren't supported in `std::simd` yet.
// https://github.com/rust-lang/portable-simd/issues/11
// https://github.com/rust-lang/portable-simd/issues/226
// https://github.com/rust-lang/portable-simd/issues/242
//
// agner fog of apple M1: https://dougallj.github.io/applecpu/firestorm-simd.html
impl SimdTree16 {
    fn query(&self, key: &[U4]) -> QueryResult {
        if key.is_empty() {
            return QueryResult::Reject;
        }
        assert!(key.len() <= 8);

        let zero = u8x16::splat(0);

        let mut match_table = [0u8; 16];
        for (i, k) in key.into_iter().enumerate() {
            match_table[usize::from(*k)] |= 1 << i;
        }
        let match_table = u8x16::from(match_table);
        let match_bitset = arm_shuffle(match_table, self.labels);

        let mut matches = [zero; 8];

        // Start with the matches for nodes under the root.
        let under_root = (u8x16::splat(0b1000_000) & self.nodes).simd_ne(zero);
        matches[0] = under_root.select(match_bitset, zero);

        for i in 0..7 {
            let prev_matches = matches[i];
            let shifted = prev_matches >> u8x16::splat(1);
            matches[i + 1] = match_bitset & shifted;
        }

        // TODO: Use reduce sum here? can this be totally branch free?
        for (i, node_matches) in matches[..(key.len() - 1)].iter().enumerate() {
            let key_matches = (u8x16::splat(1 << i) & node_matches).simd_ne(zero);
            let is_ptr = (u8x16::splat(1 << 4) & self.nodes).simd_ne(zero);
            let prefix_matches = (key_matches & is_ptr).to_bitmask();
            if prefix_matches != 0 {
                // let i = prefix_matches.count_trailing_zeros();
                return QueryResult::PartialMatch { consumed: i + 1 };
            }
        }
        let exact_match = (u8x16::splat(1 << (key.len() - 1)) & matches[key.len() - 1]).simd_ne(zero);
        if exact_match.any() {
            let exact_match = exact_match.to_array();
            debug_assert_eq!(exact_match.into_iter().filter(|b| *b).count(), 1);
            for (i, is_match) in exact_match.iter().enumerate() {
                if *is_match {
                    let node = self.nodes.as_array()[i];
                    let is_branch = (node & (1 << 6)) != 0;
                    let has_value = (node & (1 << 5)) != 0;
                    if is_branch {
                        return QueryResult::FullMatchInternal { has_value };
                    } else {
                        let is_ptr = (node & (1 << 4)) != 0;
                        let leaf_type = if is_ptr {
                            LeafType::Pointer { has_value }
                        } else {
                            LeafType::Value
                        };
                        return QueryResult::FullMatchLeaf(leaf_type);
                    }
                }
            }
        }
        QueryResult::Reject
    }
}

fn to_std_arch(v: u8x16) -> uint8x8x2_t {
    unsafe { std::mem::transmute(v) }
}

fn from_std_arch(v: uint8x8x2_t) -> u8x16 {
    unsafe { std::mem::transmute(v) }
}

fn arm_shuffle(table: u8x16, indexes: u8x16) -> u8x16 {
    let table = to_std_arch(table);
    let indexes = to_std_arch(indexes);
    let result = unsafe {
        uint8x8x2_t(
            vtbl2_u8(table, indexes.0),
            vtbl2_u8(table, indexes.1),
        )
    };
    from_std_arch(result)
}

proptest! {
    #![proptest_config(ProptestConfig { failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_simd_matches(t in any::<Tree16>(), key in prop::collection::vec(any::<U4>(), 0..=8)) {
        let scalar = ScalarTree16::from(t.clone());
        let simd = SimdTree16::from(scalar.clone());
        assert_eq!(t.query(&key), simd.query(&key));
    }
}
