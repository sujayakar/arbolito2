use std::collections::BTreeMap;
use proptest_derive::Arbitrary;
use proptest::prelude::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct U4(u8);

impl Arbitrary for U4 {
    type Parameters = ();
    type Strategy = BoxedStrategy<U4>;

    fn arbitrary_with(args: ()) -> Self::Strategy {
        (0u8..16).prop_map(U4).boxed()
    }
}

impl TryFrom<usize> for U4 {
    type Error = anyhow::Error;

    fn try_from(n: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(n < 16);
        Ok(U4(n as u8))
    }
}

impl From<U4> for usize {
    fn from(n: U4) -> usize {
        n.0 as usize
    }
}

#[derive(Clone, Debug)]
pub struct Tree16 {
    pub children: BTreeMap<U4, Tree16Node>,
}

impl Tree16 {
    pub fn query(&self, key: &[U4]) -> QueryResult {
        match key {
            &[] => QueryResult::Mismatch,
            &[head, ref tail @ ..] => {
                match self.children.get(&head) {
                    Some(tree) => tree.query(tail, 1),
                    None => QueryResult::Mismatch,
                }
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum Tree16Node {
    Branch {
        children: BTreeMap<U4, Tree16Node>,
        has_value: bool,
    },
    Leaf {
        is_ptr: bool,
        has_value: bool,
    },
}

impl Arbitrary for Tree16Node {
    type Parameters = ();
    type Strategy = BoxedStrategy<Tree16Node>;

    fn arbitrary_with(args: ()) -> Self::Strategy {
        let leaf = any::<(bool, bool)>()
            .prop_map(|(is_ptr, has_value)| Tree16Node::Leaf { is_ptr, has_value });
        let tree = leaf.prop_recursive(
            15, // At most 15 internal nodes
            16, // At most 16 nodes
            8,  // Expected 8 branches,
            |element| {
                let tree = |has_value| {
                    prop::collection::btree_map(any::<U4>(), element.clone(), 0..16)
                        .prop_map(move |mut children| {
                            let mut size = 1;
                            children.retain(|_, child| {
                                size += child.size();
                                size <= 16
                            });
                            Tree16Node::Branch { children, has_value }
                        })
                };
                prop_oneof! [
                    tree(true),
                    tree(false),
                ]
            });
        tree.boxed()
    }
}

impl Arbitrary for Tree16 {
    type Parameters = ();
    type Strategy = BoxedStrategy<Tree16>;

    fn arbitrary_with(args: ()) -> Self::Strategy {
        any::<Tree16Node>()
            .prop_filter_map("rootless", |node| {
                match node {
                    Tree16Node::Branch { children, .. } => Some(Self { children }),
                    Tree16Node::Leaf { .. } => None,
                }
            })
            .boxed()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum QueryResult {
    Partial,
    Value,
    Pointer { consumed: usize },
    Mismatch
}

impl Tree16Node {
    fn has_value(&self) -> bool {
        match self {
            Tree16Node::Branch { has_value, .. } => *has_value,
            Tree16Node::Leaf { has_value, .. } => *has_value,
        }
    }

    fn size(&self) -> usize {
        match self {
            Tree16Node::Branch { children, .. } => 1 + children.values().map(|c| c.size()).sum::<usize>(),
            Tree16Node::Leaf { .. } => 1,
        }
    }

    fn query(&self, key: &[U4], consumed: usize) -> QueryResult {
        match key {
            &[] => {
                if self.has_value() {
                    QueryResult::Value
                } else {
                    QueryResult::Partial
                }
            },
            &[head, ref tail @ ..] => {
                match self {
                    Tree16Node::Branch { ref children, .. } => {
                        match children.get(&head) {
                            Some(tree) => tree.query(tail, consumed + 1),
                            None => QueryResult::Mismatch,
                        }
                    },
                    Tree16Node::Leaf { is_ptr: true, .. } => QueryResult::Pointer { consumed },
                    Tree16Node::Leaf { is_ptr: false, .. } => QueryResult::Mismatch,
                }
            },
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_query(t in any::<Tree16>(), key in any::<Vec<U4>>()) {
        let result = t.query(&key);
        println!("{t:?} {key:?} => {result:?}");
    }
}
