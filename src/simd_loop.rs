use std::simd::{u8x16, SimdPartialEq, SimdPartialOrd, SimdUint};
use crate::transpose::match_table_simd;
use crate::model::QueryResult2;
use crate::simd::arm_shuffle;
use crate::model::U4;
use std::collections::{BTreeMap, VecDeque};

const PARENT_MASK: u8 = 0b1000_1111;
const ROOT: u8 = 0b1000_0000;

#[derive(Debug)]
pub struct Subtree {
    // 0000 is an invalid bit pattern (leaves must either have values or pointers)
    // [ under_root ] [ is_branch ] [ has_value ] [ is_ptr ] [ 4 bit parent ]
    nodes: u8x16,
    // [ 4 bits unused ] [ 4-bit label ]
    labels: u8x16,

    is_ptr: u16,
    has_value: u16,
    ptr_rank: usize,
    value_rank: usize,
}

impl Subtree {
    fn empty(ptr_rank: usize, value_rank: usize) -> Self {
        Self {
            nodes: u8x16::splat(0),
            labels: u8x16::splat(0),

            is_ptr: 0,
            has_value: 0,
            ptr_rank,
            value_rank,
        }
    }

    fn query(&self, from: u8, match_mask: MatchMask) -> Option<QueryResult2> {
        let zero = u8x16::splat(0);
        let one = u8x16::splat(1);
        let step = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let parent_ix = (u8x16::splat(0b0000_1111) & self.nodes);
        let has_parent = (u8x16::splat(0b1000_0000) & self.nodes).simd_eq(zero);
        let is_ptr = (u8x16::splat(0b0001_0000) & self.nodes).simd_ne(zero);

        // Find all nodes that are underneath `from` and match the first character.
        let under_from = (u8x16::splat(PARENT_MASK) & self.nodes).simd_eq(u8x16::splat(from));
        let matches_first = one & match_mask.bitsets;
        let mut matches = under_from.select(matches_first, zero);
        let mut path = matches;

        for i in 0..7 {
            let from_parent = arm_shuffle(matches, parent_ix);
            let next_matches = (from_parent << one) & match_mask.bitsets;
            let next_matches = has_parent.select(next_matches, zero);

            path |= matches;
            matches = next_matches;
        }

        let full_match_bitset = u8x16::splat(1 << (match_mask.len - 1));
        let partial_match = path.simd_ne(full_match_bitset) & path.simd_gt(zero) & is_ptr;
        let full_match = path.simd_eq(full_match_bitset);
        let is_match = partial_match | full_match;

        let mut result = None;

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

#[derive(Debug)]
pub struct SimdTree<T> {
    // Root subtree is index zero.
    subtrees: Vec<Subtree>,

    root_value: Option<T>,
    values: Vec<T>,
}

#[derive(Clone, Copy, Debug)]
struct MatchMask {
    bitsets: u8x16,
    len: u8,
}

impl MatchMask {
    fn new(key: &[u8]) -> (Self, usize) {
        assert!(!key.is_empty());

        let mut buf_u8 = [0u8; 4];
        let num_u8 = std::cmp::min(4, key.len());
        buf_u8[..num_u8].copy_from_slice(&key[..num_u8]);

        let mut buf_u4 = [0u8; 8];
        let num_u4 = num_u8 * 2;
        for i in 0..4 {
            buf_u4[2 * i] = buf_u8[i] & 0b0000_1111;
            buf_u4[2 * i + 1] = buf_u8[i] >> 4;
        }

        let match_mask = Self {
            bitsets: match_table_simd(buf_u4, num_u4),
            len: num_u4 as u8,
        };
        (match_mask, num_u8)
    }

    fn advance(&mut self, n: u8) {
        self.bitsets = self.bitsets >> u8x16::splat(n);
        self.len -= n;
    }
}

struct TrieNode<T> {
    children: BTreeMap<U4, TrieNode<T>>,
    value: Option<T>,
}

impl<T> TrieNode<T> {
    fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            value: None,
        }
    }

    fn insert(&mut self, key: &[U4], value: T) {
        let (head, tail) = match key {
            &[] => {
                self.value = Some(value);
                return;
            },
            &[head, ref tail @ ..] => (head, tail)
        };
        let child = self.children.entry(head).or_insert_with(TrieNode::new);
        child.insert(tail, value);
    }
}

impl<T> FromIterator<(Vec<u8>, T)> for SimdTree<T> {
    fn from_iter<I: IntoIterator<Item = (Vec<u8>, T)>>(iter: I) -> Self {
        let mut root = TrieNode::new();
        let mut len = 0;
        for (key, value) in iter {
            let mut u4_key = Vec::with_capacity(key.len() * 2);
            for byte in key {
                u4_key.push(U4::try_from((byte & 0b0000_1111) as usize).unwrap());
                u4_key.push(U4::try_from((byte >> 4) as usize).unwrap());
            }
            root.insert(&u4_key, value);
            len += 1;
        }
        let root_value = root.value.take();

        let mut subtrees = vec![Subtree::empty(0, 0)];
        let mut values = vec![];

        let mut subtree_queue = VecDeque::new();
        subtree_queue.push_back((0, root));

        while let Some((subtree_ix, mut trie)) = subtree_queue.pop_front() {
            assert!(trie.value.is_none());

            let mut labels = [0u8; 16];
            let mut nodes = [0u8; 16];
            let mut is_ptr = 0u16;
            let mut has_value = 0u16;
            let mut i = 0;
            let mut j = trie.children.len();

            // Nodes in this queue already exist in the tree and space has been reserved in j.
            let mut tree_queue = VecDeque::new();
            tree_queue.push_back(trie);

            while let Some(mut trie) = tree_queue.pop_front() {
                assert!(i + trie.children.len() < 16);
                assert!(i <= j);
                assert!(j < 16);

                // Step 1: Fully expand `trie`'s immediate children into either leaves or pointers.
                for (label, child) in &mut trie.children {
                    let mut node = 0;
                    if !child.children.is_empty() {
                        node |= 1 << 4; // has ptr
                        is_ptr |= 1 << i;
                    }
                    if let Some(value) = child.value.take() {
                        values.push(value);
                        node |= 1 << 5; // has value
                        has_value |= 1 << i;
                    }
                    node |= 1 << 7; // under root;

                    labels[i] = u8::from(*label);
                    nodes[i] = node;
                    i += 1;
                }
                // Step 2: Try to expand children into internal nodes.
                for (_, child) in trie.children {
                    if j + child.children.len() < 16 {
                        nodes[i] &= !(1 << 4); // turn off has ptr
                        nodes[i] |= 1 << 7;    // turn on is branch
                        j += child.children.len();
                        tree_queue.push_back(child);
                    }
                    // Leave the node as a pointer and add a new subtree to process.
                    else {
                        let ptr_rank = subtrees.len();
                        subtrees.push(Subtree::empty(ptr_rank, values.len()));
                        subtree_queue.push_back((ptr_rank, child));
                    }
                }
            }

            let subtree = &mut subtrees[subtree_ix];
            subtree.labels = labels.into();
            subtree.nodes = nodes.into();
            subtree.is_ptr = is_ptr;
            subtree.has_value = has_value;
        }

        SimdTree {
            subtrees,
            root_value,
            values,
        }
    }
}

#[test]
fn double_bfs() {
    let mut keys = vec![
        (b"a", 0)
    ];
    let t: SimdTree<u32> = keys
        .into_iter()
        .map(|(k, v)| (k.to_vec(), v))
        .collect();
    println!("{t:?}");
}

impl<T> SimdTree<T> {
    pub fn query(&self, key: &[u8]) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        if key.is_empty() {
            return self.root_value.as_ref();
        }

        // Loop invariant: mask_len > 0
        let (mut match_mask, mut bytes_consumed) = MatchMask::new(key);

        let mut current_subtree = &self.subtrees[0];
        let mut current_node = ROOT;
        loop {
            let result = current_subtree.query(current_node, match_mask)?;

            match_mask.advance(result.consumed);


            // Case 1: Exhausted our input entirely.
            if match_mask.len == 0 && bytes_consumed >= key.len() {
                if let Some(subtree_rank) = result.has_value {
                    return Some(&self.values[current_subtree.value_rank + subtree_rank]);
                }
                return None;
            }

            // Refill if needed.
            if match_mask.len == 0 {
                let (next_mask, next_consumed) = MatchMask::new(&key[bytes_consumed..]);
                match_mask = next_mask;
                bytes_consumed += next_consumed;
            }

            // Case 2: Jump to the next subtree.
            if let Some(subtree_rank) = result.has_ptr {
                current_subtree = &self.subtrees[current_subtree.ptr_rank + subtree_rank];
                current_node = ROOT;
                continue;
            }

            // Case 3: We have more input => retry the existing tree.
            current_node = result.node;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// struct KeyBitset {
//     key: Vec<u8>,
//     start: usize,
//     len: usize,

//     bitsets: u8x16,
//     num_loaded: usize,
// }

// impl KeyBitset {
//     #[inline(never)]
//     fn new(key: &[u8]) -> Self {
//         let mut key = key.to_vec();
//         let len = key.len();
//         key.extend([0, 0, 0]);
//         let mut s = Self {
//             key,
//             start: 0,
//             len,

//             bitsets: u8x16::splat(0),
//             num_loaded: 0,
//         };
//         s.refill();
//         s
//     }

//     #[inline(never)]
//     fn refill(&mut self) {
//         debug_assert_eq!(self.num_loaded, 0);
//         if self.is_empty() {
//             return;
//         }
//         let mut buf_u8 = [0u8; 4];
//         let num_u8 = std::cmp::min(4, self.len - self.start);
//         buf_u8[..num_u8].copy_from_slice(&self.key[self.start..(self.start + num_u8)]);

//         let mut buf_u4 = [0u8; 8];
//         let num_u4 = num_u8 * 2;
//         for i in 0..4 {
//             buf_u4[2 * i] = buf_u8[i] & 0b0000_1111;
//             buf_u4[2 * i + 1] = buf_u8[i] >> 4;
//         }

//         self.start += num_u8;
//         self.bitsets = match_table_simd(buf_u4, num_u4);
//         self.num_loaded = num_u4;
//     }

//     fn advance(&mut self, n: usize) {
//         debug_assert!(0 < n && n <= self.num_loaded);
//         self.bitsets = self.bitsets >> u8x16::splat(n as u8);
//         self.num_loaded -= n;

//         if self.num_loaded == 0 {
//             self.refill();
//         }
//     }

//     fn is_empty(&self) -> bool {
//         self.start >= self.key.len()
//     }
// }

// #[test]
// fn test_bitset() {
//     let mut b = KeyBitset::new(b"12345678");
//     b.advance(8);
//     b.advance(8);
//     assert!(b.is_empty());
// }

// // impl<T> ScalarLoop<T> {
// //     fn get(&self, key: &[u8]) -> Option<&T> {
// //         if key.is_empty() {
// //             return None;
// //         }

// //         // (current tree16, current node)
// //         // feed bytes
// //         // -> full match internal -> retry
// //         // -> full match leaf + key empty -> return has value?
// //         // -> full match leaf + key remaining -> try load pointer + retry
// //         // -> partial match -> consume bytes + retry
// //         // -> reject -> return

// //         let mut key_bitsets = todo!();
// //         let mut num_consumed = 4 * 2;

// //         let mut current_node = &self.subtrees[0];
// //         loop {
// //             let query_result = current_node.query(key_bitsets)?;
// //             key_bitsets.advance(query_result.consumed);

// //             if key_bitsets.is_empty() {
// //                 if query_result.node.has_value {
// //                     return Some(&self.values[todo!()]);
// //                 }
// //                 return None;
// //             }
// //             // feed more values into the current node
// //             if query_result.node.is_branch {
// //                 continue;
// //             }
// //             // prefix match: our key is a prefix of a value
// //             if query_result.node.has_value {
// //                 return None;
// //             }
// //             assert!(query_result.node.is_ptr);
// //             // load next node
// //             current_node = todo!();
// //         }
// //     }
// // }
