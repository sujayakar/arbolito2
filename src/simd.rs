use std::simd::u8x16;
use std::simd::{SimdPartialEq, SimdPartialOrd, ToBitMask, SimdUint};
use std::arch::aarch64::{
    uint8x8_t,
    uint8x8x2_t,
    vtbl2_u8,
};
use crate::model::{LeafType, Tree16, Tree16Node, QueryResult, QueryResult2, U4};
use crate::scalar::{NodeType, ScalarTree16};
use proptest::prelude::*;

pub struct SimdTree16 {
    // 0000 is an invalid bit pattern (leaves must either have values or pointers)
    // [ under_root ] [ is_branch ] [ has_value ] [ is_ptr ] [ 4 bit parent ]
    nodes: u8x16,
    // [ 4 bits unused ] [ 4-bit label ]
    labels: u8x16,

    is_ptr: u16,
    has_value: u16,
}

impl From<ScalarTree16> for SimdTree16 {
    fn from(t: ScalarTree16) -> Self {
        let mut nodes = [0u8; 16];
        let mut labels = [0u8; 16];
        let mut is_ptr = 0;
        let mut has_value = 0;

        assert!(t.nodes.len() <= 16);
        for (i, node) in t.nodes.into_iter().enumerate() {
            if let Some(parent) = node.parent {
                nodes[i] = parent.into();
            }
            if let NodeType::Leaf(LeafType::Pointer { .. }) = node.node_type {
                nodes[i] |= 1 << 4;
                is_ptr |= 1 << i;
            }
            if node.node_type.has_value() {
                nodes[i] |= 1 << 5;
                has_value |= 1 << i;
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
            is_ptr,
            has_value,
        }
    }
}

// Hmm, PSHUFB/TBL aren't supported in `std::simd` yet.
// https://github.com/rust-lang/portable-simd/issues/11
// https://github.com/rust-lang/portable-simd/issues/226
// https://github.com/rust-lang/portable-simd/issues/242
//
// agner fog of apple M1: https://dougallj.github.io/applecpu/firestorm-simd.html
// movemask aarch64: https://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
impl SimdTree16 {
    fn query(&self, key: &[U4]) -> QueryResult {
        if key.is_empty() {
            return QueryResult::Reject;
        }
        debug_assert!(key.len() <= 8);

        let mut key_buf = [U4::try_from(0).unwrap(); 8];
        for (i, k) in key.iter().enumerate() {
            key_buf[i] = *k;
        }
        self._query_simd(key_buf, key.len())
    }

    fn query2(&self, key: &[U4]) -> Option<QueryResult2> {
        if key.is_empty() {
            return None;
        }
        debug_assert!(key.len() <= 8);

        let mut key_buf = [U4::try_from(0).unwrap(); 8];
        for (i, k) in key.iter().enumerate() {
            key_buf[i] = *k;
        }
        self._query2(key_buf, key.len())
    }

    fn query3(&self, from: Option<U4>, key: &[U4]) -> Option<QueryResult2> {
        if key.is_empty() {
            return None;
        }
        debug_assert!(key.len() <= 8);

        let mut key_buf = [U4::try_from(0).unwrap(); 8];
        for (i, k) in key.iter().enumerate() {
            key_buf[i] = *k;
        }
        self._query3(from, key_buf, key.len())
    }

    #[inline(never)]
    fn match_table(&self, key: [U4; 8], key_len: usize) -> u8x16 {
        let mut match_table = [0u8; 16];
        for (i, k) in key.into_iter().enumerate() {
            if i < key_len {
                match_table[usize::from(k)] |= 1 << i;
            }
        }
        u8x16::from(match_table)
    }

    // #[inline(never)]
    fn match_bitset(&self, match_table: u8x16) -> u8x16 {
        let zero = u8x16::splat(0);
        let valid_nodes = (u8x16::splat(0b1111_0000) & self.nodes).simd_ne(zero);
        let match_lookups = arm_shuffle(match_table, self.labels);
        valid_nodes.select(match_lookups, zero)
    }

    #[inline(never)]
    fn matches(&self, match_bitset: u8x16) -> [u8x16; 8] {
        let zero = u8x16::splat(0);
        let mut matches = [zero; 8];

        // Start with the matches for nodes under the root.
        let under_root = (u8x16::splat(0b1000_0000) & self.nodes).simd_ne(zero);
        let matches_char = u8x16::splat(1) & match_bitset;
        matches[0] = under_root.select(matches_char, zero);

        for i in 0..7 {
            let prev_matches = matches[i];
            let has_parent = (u8x16::splat(0b1000_0000) & self.nodes).simd_eq(zero);
            let parent_ix = (u8x16::splat(0b0000_1111) & self.nodes);
            let copied = arm_shuffle(prev_matches, parent_ix);
            let next_matches = (copied << u8x16::splat(1)) & match_bitset;
            matches[i + 1] = has_parent.select(next_matches, zero);
        }

        matches
    }

    #[inline(always)]
    fn matches2(&self, match_bitset: u8x16, key_len: usize) -> QueryResult {
        let zero = u8x16::splat(0);
        let one = u8x16::splat(1);
        let mut step = [0u8; 16];
        for i in 0..16 {
            step[i] = i as u8 + 1;
        }
        let step = u8x16::from(step);

        let under_root = (u8x16::splat(0b1000_0000) & self.nodes).simd_ne(zero);
        let has_parent = (u8x16::splat(0b1000_0000) & self.nodes).simd_eq(zero);
        let is_ptr = (u8x16::splat(0b0001_0000) & self.nodes).simd_ne(zero);
        let parent_ix = (u8x16::splat(0b0000_1111) & self.nodes);

        let matches_char = u8x16::splat(1) & match_bitset;
        let mut matches = under_root.select(matches_char, zero);
        let mut path = matches;

        // Interestingly, this doesn't seem to get unrolled.
        for i in 0..7 {
            let from_parent = arm_shuffle(matches, parent_ix);
            let next_matches = (from_parent << one) & match_bitset;
            let next_matches = has_parent.select(next_matches, zero);

            path |= matches;
            matches = next_matches;
        }

        let mut result = QueryResult::Reject;

        let prefix_match = is_ptr.select(path, zero).reduce_sum();
        if prefix_match > 0 {
            let consumed = prefix_match.trailing_zeros() as usize + 1;
            result = QueryResult::PartialMatch { consumed };
        }

        let exact_match = u8x16::splat(1 << (key_len - 1))
            .simd_eq(path)
            .select(step, zero)
            .reduce_sum();
        if exact_match > 0 {
            debug_assert!(1 <= exact_match && exact_match < 17);
            let exact_match = exact_match - 1;
            let node = unsafe { self.nodes.as_array().get_unchecked(exact_match as usize) };
            let is_ptr = node & (1 << 4) != 0;
            let has_value = node & (1 << 5) != 0;
            let is_branch = node & (1 << 6) != 0;
            if is_branch {
                result = QueryResult::FullMatchInternal { has_value };
            } else {
                let leaf_type = if is_ptr {
                    LeafType::Pointer { has_value }
                } else {
                    LeafType::Value
                };
                result = QueryResult::FullMatchLeaf(leaf_type);
            }
        }
        result
    }

    #[inline(never)]
    fn find_prefix(&self, matches: [u8x16; 8], key_len: usize) -> Option<QueryResult> {
        let zero = u8x16::splat(0);
        for (i, node_matches) in matches[..key_len - 1].iter().enumerate() {
            let key_matches = (u8x16::splat(1 << i) & node_matches).simd_ne(zero);
            let is_ptr = (u8x16::splat(1 << 4) & self.nodes).simd_ne(zero);
            let prefix_matches = (key_matches & is_ptr).to_bitmask();
            if prefix_matches != 0 {
                // let i = prefix_matches.count_trailing_zeros();
                return Some(QueryResult::PartialMatch { consumed: i + 1 });
            }
        }
        None
    }

    #[inline(never)]
    fn find_exact(&self, matches: [u8x16; 8], key_len: usize) -> Option<QueryResult> {
        let zero = u8x16::splat(0);
        let exact_match = (u8x16::splat(1 << (key_len - 1)) & matches[key_len - 1])
            .simd_ne(zero)
            .to_bitmask();
        if exact_match != 0 {
            debug_assert_eq!(exact_match.count_ones(), 1);
            let i = exact_match.trailing_zeros();
            let node = self.nodes.as_array()[i as usize];
            let is_branch = (node & (1 << 6)) != 0;
            let has_value = (node & (1 << 5)) != 0;
            if is_branch {
                return Some(QueryResult::FullMatchInternal { has_value });
            } else {
                let is_ptr = (node & (1 << 4)) != 0;
                let leaf_type = if is_ptr {
                    LeafType::Pointer { has_value }
                } else {
                    LeafType::Value
                };
                return Some(QueryResult::FullMatchLeaf(leaf_type));
            }
        }
        None
    }

    #[inline(never)]
    fn _query_simd(&self, key: [U4; 8], key_len: usize) -> QueryResult {
        use crate::transpose::match_table_simd;
        let mut k2 = [0u8; 8];
        for i in 0..8 {
            k2[i] = key[i].into();
        }
        let match_table = match_table_simd(k2, key_len);
        // let match_table = self.match_table(key, key_len);
        let match_bitset = self.match_bitset(match_table);
        // println!("match bitset: {:?}", match_bitset);

        // let matches = self.matches(match_bitset);
        self.matches2(match_bitset, key_len)
    }

    #[inline(never)]
    fn _query2(&self, key: [U4; 8], key_len: usize) -> Option<QueryResult2> {
        use crate::transpose::match_table_simd;
        let mut k2 = [0u8; 8];
        for i in 0..8 {
            k2[i] = key[i].into();
        }
        let match_table = match_table_simd(k2, key_len);
        let match_bitset = self.match_bitset(match_table);

        let zero = u8x16::splat(0);
        let one = u8x16::splat(1);
        let mut step = [0u8; 16];
        for i in 0..16 {
            step[i] = i as u8;
        }
        let step = u8x16::from(step);

        let under_root = (u8x16::splat(0b1000_0000) & self.nodes).simd_ne(zero);
        let has_parent = (u8x16::splat(0b1000_0000) & self.nodes).simd_eq(zero);
        let is_ptr = (u8x16::splat(0b0001_0000) & self.nodes).simd_ne(zero);
        let has_value = (u8x16::splat(0b0010_0000) & self.nodes).simd_ne(zero);
        let parent_ix = (u8x16::splat(0b0000_1111) & self.nodes);

        let matches_char = u8x16::splat(1) & match_bitset;
        let mut matches = under_root.select(matches_char, zero);
        let mut path = matches;

        // Interestingly, this doesn't seem to get unrolled.
        for i in 0..7 {
            let from_parent = arm_shuffle(matches, parent_ix);
            let next_matches = (from_parent << one) & match_bitset;
            let next_matches = has_parent.select(next_matches, zero);

            path |= matches;
            matches = next_matches;
        }

        let mut result = None;

        // The path hits at most one pointer leaf.
        // Intersect pointers with the path the key took.
        //
        // We want to count where is_ptr and node is to the left.
        //
        // TODO: this codegens pretty poorly.

        let full_match_bitset = u8x16::splat(1 << (key_len - 1));
        let partial_match = path.simd_ne(full_match_bitset) & path.simd_gt(zero) & is_ptr;
        let full_match = path.simd_eq(full_match_bitset);
        let is_match = partial_match | full_match;

        let match_bitset = is_match.select(path, zero).reduce_sum();
        if match_bitset > 0 {
            let i = is_match.select(step, zero).reduce_sum();
            let node = unsafe { self.nodes.as_array().get_unchecked(i as usize) };

            let consumed = match_bitset.trailing_zeros() as u8 + 1;

            let node_is_ptr = node & (1 << 4) != 0;
            let node_has_value = node & (1 << 5) != 0;
            let is_branch = node & (1 << 6) != 0;

            let prev_mask = !(0b1111_1111_1111_1111 << i);
            let ptr_rank = (prev_mask & self.is_ptr).count_ones();
            let value_rank = (prev_mask & self.has_value).count_ones();

            result = Some(QueryResult2 {
                consumed,
                node: i,
                has_value: if node_has_value { Some(value_rank as usize) } else { None },
                has_ptr: if node_is_ptr { Some(ptr_rank as usize) } else { None },
                is_branch,
            });
        }

        result
    }

    #[inline(never)]
    fn _query3(&self, from: Option<U4>, key: [U4; 8], key_len: usize) -> Option<QueryResult2> {
        use crate::transpose::match_table_simd;
        let mut k2 = [0u8; 8];
        for i in 0..8 {
            k2[i] = key[i].into();
        }
        let match_table = match_table_simd(k2, key_len);
        let match_bitset = self.match_bitset(match_table);

        let zero = u8x16::splat(0);
        let one = u8x16::splat(1);
        let mut step = [0u8; 16];
        for i in 0..16 {
            step[i] = i as u8;
        }
        let step = u8x16::from(step);

        let has_parent = (u8x16::splat(0b1000_0000) & self.nodes).simd_eq(zero);
        let is_ptr = (u8x16::splat(0b0001_0000) & self.nodes).simd_ne(zero);
        let has_value = (u8x16::splat(0b0010_0000) & self.nodes).simd_ne(zero);
        let parent_ix = (u8x16::splat(0b0000_1111) & self.nodes);

        let start_u8 = match from {
            None => 0b1000_0000,
            Some(i) => u8::from(i),
        };
        let under_start = (u8x16::splat(0b1000_1111) & self.nodes).simd_eq(u8x16::splat(start_u8));

        let matches_char = u8x16::splat(1) & match_bitset;
        let mut matches = under_start.select(matches_char, zero);
        let mut path = matches;

        // Interestingly, this doesn't seem to get unrolled.
        for i in 0..7 {
            let from_parent = arm_shuffle(matches, parent_ix);
            let next_matches = (from_parent << one) & match_bitset;
            let next_matches = has_parent.select(next_matches, zero);

            path |= matches;
            matches = next_matches;
        }

        let mut result = None;

        // The path hits at most one pointer leaf.
        // Intersect pointers with the path the key took.
        //
        // We want to count where is_ptr and node is to the left.
        //
        // TODO: this codegens pretty poorly.

        let full_match_bitset = u8x16::splat(1 << (key_len - 1));
        let partial_match = path.simd_ne(full_match_bitset) & path.simd_gt(zero) & is_ptr;
        let full_match = path.simd_eq(full_match_bitset);
        let is_match = partial_match | full_match;

        let match_bitset = is_match.select(path, zero).reduce_sum();
        if match_bitset > 0 {
            let i = is_match.select(step, zero).reduce_sum();
            let node = unsafe { self.nodes.as_array().get_unchecked(i as usize) };

            let consumed = match_bitset.trailing_zeros() as u8 + 1;

            let node_is_ptr = node & (1 << 4) != 0;
            let node_has_value = node & (1 << 5) != 0;
            let is_branch = node & (1 << 6) != 0;

            let prev_mask = !(0b1111_1111_1111_1111 << i);
            let ptr_rank = (prev_mask & self.is_ptr).count_ones();
            let value_rank = (prev_mask & self.has_value).count_ones();

            result = Some(QueryResult2 {
                consumed,
                node: i,
                has_value: if node_has_value { Some(value_rank as usize) } else { None },
                has_ptr: if node_is_ptr { Some(ptr_rank as usize) } else { None },
                is_branch,
            });
        }

        result
    }
}

fn to_std_arch(v: u8x16) -> uint8x8x2_t {
    unsafe { std::mem::transmute(v) }
}

fn from_std_arch(v: uint8x8x2_t) -> u8x16 {
    unsafe { std::mem::transmute(v) }
}

pub fn arm_shuffle(table: u8x16, indexes: u8x16) -> u8x16 {
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



#[test]
fn test_simd_trophy() -> anyhow::Result<()> {
    let t = Tree16 {
        children: maplit::btreemap!(),
    };
    let key = vec![
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
        U4::try_from(0)?,
    ];

    let scalar = ScalarTree16::from(t.clone());
    let simd = SimdTree16::from(scalar.clone());
    assert_eq!(t.query(&key), simd.query(&key));

    Ok(())
}

#[test]
fn test_simd_trophy2() -> anyhow::Result<()> {
    let t = Tree16 {
        children: maplit::btreemap!(
            U4::try_from(0)? => Tree16Node::Leaf(LeafType::Pointer { has_value: false }),
            U4::try_from(4)? => Tree16Node::Leaf(LeafType::Value),
        ),
    };
    let key = vec![U4::try_from(4)?];

    let scalar = ScalarTree16::from(t.clone());
    let simd = SimdTree16::from(scalar.clone());
    assert_eq!(t.query2(&key), simd.query2(&key));

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 81920, failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_simd_matches(t in any::<Tree16>(), key in prop::collection::vec(any::<U4>(), 0..=8)) {
        let scalar = ScalarTree16::from(t.clone());
        let simd = SimdTree16::from(scalar.clone());
        assert_eq!(t.query(&key), simd.query(&key));

        for key in t.keys() {
            assert_eq!(t.query(&key), simd.query(&key));
        }
    }

    #[test]
    fn test_simd_matches2(t in any::<Tree16>(), key in prop::collection::vec(any::<U4>(), 0..=8)) {
        let scalar = ScalarTree16::from(t.clone());
        let simd = SimdTree16::from(scalar.clone());
        assert_eq!(t.query2(&key), simd.query2(&key));

        for key in t.keys() {
            assert_eq!(t.query2(&key), simd.query2(&key));
        }
    }

    #[test]
    fn test_simd_matches3(t in any::<Tree16>(), start in any::<Option<u8>>(), key in prop::collection::vec(any::<U4>(), 0..=8)) {
        let scalar = ScalarTree16::from(t.clone());
        let start = if scalar.nodes.is_empty() { None } else { start.map(|i| U4::try_from(i as usize % scalar.nodes.len()).unwrap()) };
        let simd = SimdTree16::from(scalar.clone());
        assert_eq!(t.query3(start, &key), simd.query3(start, &key));

        // for key in t.keys() {
        //     assert_eq!(t.query2(&key), simd.query2(&key));
        // }
    }
}
