use std::collections::VecDeque;
use maplit::btreemap;
use proptest::prelude::*;

use crate::model::{Tree16, Tree16Node, QueryResult, U4};

#[derive(Debug)]
struct ScalarNode {
    parent: Option<U4>,
    label: U4,

    // TODO: compress these flags.
    is_ptr: bool,
    has_value: bool,
    is_leaf: bool,
}

struct ScalarTree16 {
    nodes: Vec<ScalarNode>,
}

impl From<Tree16> for ScalarTree16 {
    fn from(tree: Tree16) -> Self {
        let mut queue = VecDeque::new();
        for (label, child) in tree.children {
            queue.push_back((None, label, child));
        }
        let mut nodes = Vec::with_capacity(16);
        while let Some((parent, label, child)) = queue.pop_front() {
            let i = U4::try_from(nodes.len()).unwrap();
            let (is_ptr, has_value, is_leaf) = match child {
                Tree16Node::Branch { children, has_value } => {
                    for (label, child) in children {
                        queue.push_back((Some(i), label, child));
                    }
                    (false, has_value, false)
                },
                Tree16Node::PtrLeaf { has_value } => (true, has_value, true),
                Tree16Node::ValueLeaf => (false, true, true),
            };
            let node = ScalarNode {
                parent,
                label,
                is_ptr,
                has_value,
                is_leaf,
            };
            nodes.push(node);
        }
        Self { nodes }

    }
}

impl ScalarTree16 {
    fn query(&self, key: &[U4]) -> QueryResult {
        if key.is_empty() {
            return QueryResult::Mismatch;
        }

        // Construct "bitsets" of length `key.len()` for each node.
        let empty = key.iter().map(|_| false).collect::<VecDeque<_>>();
        let label_matches = self
            .nodes
            .iter()
            .map(|node| {
                key.iter().map(|k| node.label == *k).collect::<VecDeque<_>>()
            })
            .collect::<Vec<_>>();

        // Start with the match sets for nodes under the root.
        let matches0 = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                if node.parent.is_none() {
                    label_matches[i].clone()
                } else {
                    empty.clone()
                }
            })
            .collect::<Vec<_>>();

        let mut matches = vec![matches0];

        for i in 0..(key.len() - 1) {
            let prev_matches = matches.last().unwrap();
            let next_matches = self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    if let Some(parent_ix) = node.parent {
                        let mut parent_matches = prev_matches[usize::from(parent_ix)].clone();

                        // Shift the matches vector forward one position.
                        parent_matches.push_front(false);
                        parent_matches.pop_back();

                        // AND the shifted parent vector with our matches.
                        label_matches[i]
                            .iter()
                            .zip(parent_matches.iter())
                            .map(|(node, parent)| *node && *parent)
                            .collect()
                    } else {
                        empty.clone()
                    }
                })
                .collect();
            matches.push(next_matches);
        }

        let mut exact_match = matches
            .last()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(i, matches)| {
                if matches[key.len() - 1] {
                    let node = &self.nodes[i];
                    if node.has_value {
                        return Some(QueryResult::Value);
                    }
                    // TODO: Resume feeding in more bytes.
                    // if !node.is_leaf {
                    //     return Some(QueryResult::Partial);
                    // }
                }
                None
            })
            .collect::<Vec<_>>();

        let mut prefix_match = vec![];
        for (len, node_matches) in matches.iter().enumerate() {
            for (i, matches) in node_matches.iter().enumerate() {
                let node = &self.nodes[i];
                if matches[len] && node.is_ptr && len + 1 < key.len() {
                    prefix_match.push(QueryResult::Pointer { consumed: len + 1 });
                }
            }
        }
        assert!(exact_match.len() + prefix_match.len() <= 1);
        if let Some(r) = exact_match.pop() {
            return r;
        }
        if let Some(r) = prefix_match.pop() {
            return r;
        }
        QueryResult::Mismatch
    }
}

#[test]
fn trophycase() -> anyhow::Result<()> {
    let t = Tree16 {
        children: btreemap!(
            U4::try_from(11)? => Tree16Node::PtrLeaf { has_value: false },
        ),
    };
    let key = vec![U4::try_from(11)?];

    let scalar = ScalarTree16::from(t.clone());
    assert_eq!(t.query(&key), scalar.query(&key));

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_matches_tree16(t in any::<Tree16>(), mut keys in any::<Vec<Vec<U4>>>()) {
        keys.extend(t.keys());
        let scalar = ScalarTree16::from(t.clone());
        for key in keys {
            assert_eq!(t.query(&key), scalar.query(&key));
        }
    }
}
