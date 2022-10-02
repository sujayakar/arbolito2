use std::collections::{BTreeMap, VecDeque};
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

impl From<U4> for u8 {
    fn from(n: U4) -> u8 {
        n.0
    }
}

#[derive(Debug)]
struct BfsOrder<'a> {
    tree: &'a Tree16,
    order: Vec<&'a Tree16Node>,
    by_path: BTreeMap<Vec<U4>, usize>,
}

impl<'a> BfsOrder<'a> {
    fn node_ix(&self, key: &[U4]) -> u8 {
        self.by_path[key] as u8
    }

    fn ptr_rank(&self, key: &[U4]) -> Option<usize> {
        let i = self.by_path.get(key)?;
        if matches!(self.order[*i], Tree16Node::Leaf(LeafType::Pointer { .. })) {
            let rank = self.order[0..*i]
                .into_iter()
                .filter(|n| matches!(n, Tree16Node::Leaf(LeafType::Pointer { .. })))
                .count();
            return Some(rank);
        }
        None
    }

    fn value_rank(&self, key: &[U4]) -> Option<usize> {
        let i = self.by_path.get(key)?;
        if self.order[*i].has_value() {
            let rank = self.order[0..*i]
                .into_iter()
                .filter(|n| n.has_value())
                .count();
            return Some(rank);
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct Tree16 {
    pub children: BTreeMap<U4, Tree16Node>,
}

impl Tree16 {
    fn bfs_order(&self) -> BfsOrder<'_> {
        let mut queue = VecDeque::new();
        for (label, node) in &self.children {
            queue.push_back((vec![*label], node));
        }
        let mut order = vec![];
        let mut by_path = BTreeMap::new();

        while let Some((path, node)) = queue.pop_front() {
            let i = order.len();

            order.push(node);
            by_path.insert(path.clone(), i);

            if let Tree16Node::Branch { ref children, .. } = node {
                for (label, child) in children {
                    let mut child_path = path.clone();
                    child_path.push(*label);
                    queue.push_back((child_path, child));
                }
            }
        }
        BfsOrder { tree: self, order, by_path }
    }

    pub fn query(&self, key: &[U4]) -> QueryResult {
        match key {
            &[] => QueryResult::Reject,
            &[head, ref tail @ ..] => {
                match self.children.get(&head) {
                    Some(tree) => tree.query(tail, 1),
                    None => QueryResult::Reject,
                }
            },
        }
    }

    pub fn query2(&self, key: &[U4]) -> Option<QueryResult2> {
        let bfs_order = self.bfs_order();
        match self.query(key) {
            QueryResult::FullMatchInternal { has_value } => {
                let value_rank = bfs_order.value_rank(&key);
                assert_eq!(value_rank.is_some(), has_value);
                assert!(bfs_order.ptr_rank(&key).is_none());
                Some(QueryResult2 {
                    consumed: key.len() as u8,
                    node: bfs_order.node_ix(&key),
                    has_value: value_rank,
                    has_ptr: None,
                    is_branch: true,
                })
            },
            QueryResult::FullMatchLeaf(leaf_type) => {
                let value_rank = bfs_order.value_rank(&key);
                assert_eq!(value_rank.is_some(), leaf_type.has_value());
                let ptr_rank = bfs_order.ptr_rank(&key);
                assert_eq!(ptr_rank.is_some(), matches!(leaf_type, LeafType::Pointer { .. }));
                Some(QueryResult2 {
                    consumed: key.len() as u8,
                    node: bfs_order.node_ix(&key),
                    has_value: value_rank,
                    has_ptr: ptr_rank,
                    is_branch: false,
                })
            },
            QueryResult::PartialMatch { consumed } => {
                let value_rank = bfs_order.value_rank(&key[..consumed]);
                let ptr_rank = bfs_order.ptr_rank(&key[..consumed]);
                assert!(ptr_rank.is_some());
                Some(QueryResult2 {
                    consumed: consumed as u8,
                    node: bfs_order.node_ix(&key[..consumed]),
                    has_value: value_rank,
                    has_ptr: ptr_rank,
                    is_branch: false,
                })
            },
            QueryResult::Reject => None,
        }
    }

    pub fn query3(&self, from: Option<U4>, key: &[U4]) -> Option<QueryResult2> {
        if key.is_empty() {
            return None;
        }
        let bfs_order = self.bfs_order();
        let mut prefixed: Vec<U4> = vec![];
        let mut prefix_len = 0;
        if let Some(i) = from {
            // Illegal to start from a pointer node.
            if matches!(bfs_order.order[usize::from(i)], Tree16Node::Leaf(LeafType::Pointer { .. })) {
                return None;
            }
            let prefix = bfs_order
                .by_path
                .iter()
                .filter_map(|(key, j)| if usize::from(i) == *j { Some(key) } else { None })
                .next()
                .unwrap();
            prefix_len += prefix.len() as u8;
            prefixed.extend(prefix);
        }
        prefixed.extend(key);
        let mut result = self.query2(&prefixed);
        if let Some(ref mut r) = result {
            r.consumed -= prefix_len;
        }
        result
    }

    pub fn keys(&self) -> Vec<Vec<U4>> {
        let mut out = vec![];
        for (label, node) in &self.children {
            for mut key in node.keys_reversed() {
                key.push(*label);
                key.reverse();
                out.push(key);
            }
        }
        out
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LeafType {
    Pointer { has_value: bool },
    Value,
}

impl LeafType {
    pub fn has_value(&self) -> bool {
        match self {
            LeafType::Pointer { has_value } => *has_value,
            LeafType::Value => true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Tree16Node {
    Branch {
        children: BTreeMap<U4, Tree16Node>,
        has_value: bool,
    },
    Leaf(LeafType),
}

impl Arbitrary for Tree16Node {
    type Parameters = ();
    type Strategy = BoxedStrategy<Tree16Node>;

    fn arbitrary_with(args: ()) -> Self::Strategy {
        let leaf = any::<Option<bool>>()
            .prop_map(|has_value| {
                let leaf_type = match has_value {
                    Some(has_value) => LeafType::Pointer { has_value },
                    None => LeafType::Value,
                };
                Tree16Node::Leaf(leaf_type)
            });
        let tree = leaf.prop_recursive(
            15, // At most 15 internal nodes
            16, // At most 16 nodes
            8,  // Expected 8 branches
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
                    Tree16Node::Leaf(..) => None,
                }
            })
            .boxed()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct QueryResult2 {
    pub consumed: u8,
    pub node: u8,

    // Does the node we landed on have a value? If so, what's its rank within the tree in BFS order?
    pub has_value: Option<usize>,
    // Does the node we landed on have a pointer?
    pub has_ptr: Option<usize>,
    // Is the node we landed on a branch? (Note that is_branch => has_ptr.is_none())
    pub is_branch: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub enum QueryResult {
    // We consumed all of the input and landed on an internal node.
    FullMatchInternal { has_value: bool },
    // We consumed all of the input and landed on a leaf.
    FullMatchLeaf(LeafType),
    // We consumed some prefix of the input and landed on a pointer leaf.
    PartialMatch { consumed: usize },
    // The key does not exist in the tree.
    Reject,
}

impl Tree16Node {
    fn has_value(&self) -> bool {
        match self {
            Tree16Node::Branch { has_value, .. } => *has_value,
            Tree16Node::Leaf(leaf_type) => leaf_type.has_value(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Tree16Node::Branch { children, .. } => 1 + children.values().map(|c| c.size()).sum::<usize>(),
            Tree16Node::Leaf(..) => 1,
        }
    }

    fn query(&self, key: &[U4], consumed: usize) -> QueryResult {
        match key {
            &[] => {
                match self {
                    Tree16Node::Branch { has_value, .. } => QueryResult::FullMatchInternal { has_value: *has_value },
                    Tree16Node::Leaf(leaf_type) => QueryResult::FullMatchLeaf(*leaf_type),
                }
            },
            &[head, ref tail @ ..] => {
                match self {
                    Tree16Node::Branch { ref children, .. } => {
                        match children.get(&head) {
                            Some(tree) => tree.query(tail, consumed + 1),
                            None => QueryResult::Reject,
                        }
                    },
                    Tree16Node::Leaf(LeafType::Pointer { .. }) => QueryResult::PartialMatch { consumed },
                    Tree16Node::Leaf(LeafType::Value) => QueryResult::Reject,
                }
            },
        }
    }

    fn keys_reversed(&self) -> Vec<Vec<U4>> {
        let mut out = vec![];
        if self.has_value() {
            out.push(vec![]);
        }
        if let Tree16Node::Branch { ref children, .. } = self {
            for (label, child) in children {
                for mut key in child.keys_reversed() {
                    key.push(*label);
                    out.push(key);
                }
            }
        }
        out
    }

}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_query(t in any::<Tree16>(), mut keys in any::<Vec<Vec<U4>>>()) {
        keys.extend(t.keys());
        for key in keys {
            let result = t.query(&key);
            println!("{result:?}");
        }
    }
}
