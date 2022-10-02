use std::simd::u8x16;
use crate::transpose::match_table_simd;
use crate::simd::SimdTree16;

struct SimdTree<T> {
    tree: SimdTree16,
    values: Vec<T>,
    ptrs: Vec<usize>,
}

pub struct SimdLoop<T> {
    subtrees: Vec<SimdTree<T>>,
    root_value: Option<T>,
    len: usize,
}

impl<T> SimdLoop<T> {
    pub fn query(&self, key: &[u8]) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        if key.is_empty() {
            return self.root_value.as_ref();
        }

        // let mut key_bitset = KeyBitset::new(key);

        // let mut current_tree = 0;
        // let mut current_node = None;
        // loop {
        //     let node = &self.subtrees[i];
        //     match node.tree.query2(key_bitse

        // }

        todo!()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

struct KeyBitset {
    key: Vec<u8>,
    start: usize,
    len: usize,

    bitsets: u8x16,
    num_loaded: usize,
}

impl KeyBitset {
    #[inline(never)]
    fn new(key: &[u8]) -> Self {
        let mut key = key.to_vec();
        let len = key.len();
        key.extend([0, 0, 0]);
        let mut s = Self {
            key,
            start: 0,
            len,

            bitsets: u8x16::splat(0),
            num_loaded: 0,
        };
        s.refill();
        s
    }

    #[inline(never)]
    fn refill(&mut self) {
        debug_assert_eq!(self.num_loaded, 0);
        if self.is_empty() {
            return;
        }
        let mut buf_u8 = [0u8; 4];
        let num_u8 = std::cmp::min(4, self.len - self.start);
        buf_u8[..num_u8].copy_from_slice(&self.key[self.start..(self.start + num_u8)]);

        let mut buf_u4 = [0u8; 8];
        let num_u4 = num_u8 * 2;
        for i in 0..4 {
            buf_u4[2 * i] = buf_u8[i] & 0b0000_1111;
            buf_u4[2 * i + 1] = buf_u8[i] >> 4;
        }

        self.start += num_u8;
        self.bitsets = match_table_simd(buf_u4, num_u4);
        self.num_loaded = num_u4;
    }

    fn advance(&mut self, n: usize) {
        debug_assert!(0 < n && n <= self.num_loaded);
        self.bitsets = self.bitsets >> u8x16::splat(n as u8);
        self.num_loaded -= n;

        if self.num_loaded == 0 {
            self.refill();
        }
    }

    fn is_empty(&self) -> bool {
        self.start >= self.key.len()
    }
}

#[test]
fn test_bitset() {
    let mut b = KeyBitset::new(b"12345678");
    b.advance(8);
    b.advance(8);
    assert!(b.is_empty());
}

// impl<T> ScalarLoop<T> {
//     fn get(&self, key: &[u8]) -> Option<&T> {
//         if key.is_empty() {
//             return None;
//         }

//         // (current tree16, current node)
//         // feed bytes
//         // -> full match internal -> retry
//         // -> full match leaf + key empty -> return has value?
//         // -> full match leaf + key remaining -> try load pointer + retry
//         // -> partial match -> consume bytes + retry
//         // -> reject -> return

//         let mut key_bitsets = todo!();
//         let mut num_consumed = 4 * 2;

//         let mut current_node = &self.subtrees[0];
//         loop {
//             let query_result = current_node.query(key_bitsets)?;
//             key_bitsets.advance(query_result.consumed);

//             if key_bitsets.is_empty() {
//                 if query_result.node.has_value {
//                     return Some(&self.values[todo!()]);
//                 }
//                 return None;
//             }
//             // feed more values into the current node
//             if query_result.node.is_branch {
//                 continue;
//             }
//             // prefix match: our key is a prefix of a value
//             if query_result.node.has_value {
//                 return None;
//             }
//             assert!(query_result.node.is_ptr);
//             // load next node
//             current_node = todo!();
//         }
//     }
// }
