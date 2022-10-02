use std::collections::VecDeque;
use maplit::btreemap;
use proptest::prelude::*;

use crate::model::{LeafType, Tree16, Tree16Node, QueryResult, QueryResult2, U4};

#[derive(Copy, Clone, Debug)]
pub struct ScalarNode {
    pub parent: Option<U4>,
    pub label: U4,
    pub node_type: NodeType,
}

#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Branch { has_value: bool },
    Leaf(LeafType),
}

impl NodeType {
    pub fn has_value(&self) -> bool {
        match self {
            NodeType::Branch { has_value } => *has_value,
            NodeType::Leaf(leaf_type) => leaf_type.has_value(),
        }
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, NodeType::Leaf(LeafType::Pointer { .. }))
    }
}

#[derive(Clone)]
pub struct ScalarTree16 {
    pub nodes: Vec<ScalarNode>,
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
            let node_type = match child {
                Tree16Node::Branch { children, has_value } => {
                    for (label, child) in children {
                        queue.push_back((Some(i), label, child));
                    }
                    NodeType::Branch { has_value }
                },
                Tree16Node::Leaf(leaf_type) => NodeType::Leaf(leaf_type),
            };
            let node = ScalarNode {
                parent,
                label,
                node_type,
            };
            nodes.push(node);
        }
        Self { nodes }

    }
}

impl ScalarTree16 {
    fn query(&self, key: &[U4]) -> QueryResult {
        if key.is_empty() {
            return QueryResult::Reject;
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
                    let result = match self.nodes[i].node_type {
                        NodeType::Leaf(leaf_type) => QueryResult::FullMatchLeaf(leaf_type),
                        NodeType::Branch { has_value } => QueryResult::FullMatchInternal { has_value },
                    };
                    return Some(result);
                }
                None
            })
            .collect::<Vec<_>>();

        let mut prefix_match = vec![];
        for (len, node_matches) in matches.iter().enumerate() {
            for (i, matches) in node_matches.iter().enumerate() {
                if !matches[len] || len + 1 == key.len() {
                    continue;
                }
                if let NodeType::Leaf(LeafType::Pointer { .. }) = self.nodes[i].node_type {
                    prefix_match.push(QueryResult::PartialMatch { consumed: len + 1 });
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
        QueryResult::Reject
    }

    fn query2(&self, key: &[U4]) -> Option<QueryResult2> {
        if key.is_empty() {
            return None;
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
                    let n = self.nodes[i].node_type;
                    let result = QueryResult2 {
                        consumed: key.len() as u8,
                        node: i as u8,
                        has_value: self.value_rank(i),
                        has_ptr: self.ptr_rank(i),
                        is_branch: matches!(n, NodeType::Branch { .. }),
                    };
                    return Some(result)
                }
                None
            })
            .collect::<Vec<_>>();

        let mut prefix_match = vec![];
        for (len, node_matches) in matches.iter().enumerate() {
            for (i, matches) in node_matches.iter().enumerate() {
                if !matches[len] || len + 1 == key.len() {
                    continue;
                }
                if let Some(ptr_rank) = self.ptr_rank(i) {
                    let result = QueryResult2 {
                        consumed: len as u8 + 1,
                        node: i as u8,
                        has_value: self.value_rank(i),
                        has_ptr: Some(ptr_rank),
                        is_branch: false,
                    };
                    prefix_match.push(result);
                }
            }
        }
        assert!(exact_match.len() + prefix_match.len() <= 1);
        if let Some(r) = exact_match.pop() {
            return Some(r);
        }
        if let Some(r) = prefix_match.pop() {
            return Some(r);
        }

        None
    }

    fn query3(&self, from: Option<U4>, key: &[U4]) -> Option<QueryResult2> {
        if key.is_empty() {
            return None;
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

        // Start with the match sets for nodes under the start node.
        let matches0 = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                if node.parent == from {
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
                    let n = self.nodes[i].node_type;
                    let result = QueryResult2 {
                        consumed: key.len() as u8,
                        node: i as u8,
                        has_value: self.value_rank(i),
                        has_ptr: self.ptr_rank(i),
                        is_branch: matches!(n, NodeType::Branch { .. }),
                    };
                    return Some(result)
                }
                None
            })
            .collect::<Vec<_>>();

        let mut prefix_match = vec![];
        for (len, node_matches) in matches.iter().enumerate() {
            for (i, matches) in node_matches.iter().enumerate() {
                if !matches[len] || len + 1 == key.len() {
                    continue;
                }
                if let Some(ptr_rank) = self.ptr_rank(i) {
                    let result = QueryResult2 {
                        consumed: len as u8 + 1,
                        node: i as u8,
                        has_value: self.value_rank(i),
                        has_ptr: Some(ptr_rank),
                        is_branch: false,
                    };
                    prefix_match.push(result);
                }
            }
        }
        assert!(exact_match.len() + prefix_match.len() <= 1);
        if let Some(r) = exact_match.pop() {
            return Some(r);
        }
        if let Some(r) = prefix_match.pop() {
            return Some(r);
        }

        None
    }

    fn value_rank(&self, i: usize) -> Option<usize> {
        if !self.nodes[i].node_type.has_value() {
            return None;
        }
        let rank = self.nodes[..i]
            .iter()
            .filter(|n| n.node_type.has_value())
            .count();
        Some(rank)
    }

    fn ptr_rank(&self, i: usize) -> Option<usize> {
        if !self.nodes[i].node_type.is_ptr() {
            return None;
        }
        let rank = self.nodes[..i]
            .iter()
            .filter(|n| n.node_type.is_ptr())
            .count();
        Some(rank)
    }
}

#[test]
fn trophycase() -> anyhow::Result<()> {
    let t = Tree16 {
        children: btreemap!(
            U4::try_from(11)? => Tree16Node::Leaf(LeafType::Pointer { has_value: false }),
        ),
    };
    let key = vec![U4::try_from(11)?];

    let scalar = ScalarTree16::from(t.clone());
    assert_eq!(t.query(&key), scalar.query(&key));

    Ok(())
}

#[test]
fn trophycase2() -> anyhow::Result<()> {
    let t = Tree16 {
        children: btreemap!(
            U4::try_from(0)? => Tree16Node::Leaf(LeafType::Value),
            U4::try_from(13)? => Tree16Node::Leaf(LeafType::Pointer { has_value: false }),
        ),
    };
    let key = vec![U4::try_from(13)?, U4::try_from(0)?];

    let scalar = ScalarTree16::from(t.clone());
    assert_eq!(t.query2(&key), scalar.query2(&key));

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn test_matches_tree16(t in any::<Tree16>(), mut keys in any::<Vec<Vec<U4>>>()) {
        keys.extend(t.keys());
        let scalar = ScalarTree16::from(t.clone());
        for key in keys {
            assert_eq!(t.query(&key), scalar.query(&key));
        }
    }

    #[test]
    fn test_matches_tree162(t in any::<Tree16>(), mut keys in any::<Vec<Vec<U4>>>()) {
        keys.extend(t.keys());
        let scalar = ScalarTree16::from(t.clone());
        for key in keys {
            assert_eq!(t.query2(&key), scalar.query2(&key));
        }
    }

    #[test]
    fn test_matches_tree163(t in any::<Tree16>(), start in any::<Option<u8>>(), mut keys in any::<Vec<Vec<U4>>>()) {
        // keys.extend(t.keys());
        let scalar = ScalarTree16::from(t.clone());

        let start = if scalar.nodes.is_empty() { None } else { start.map(|i| U4::try_from(i as usize % scalar.nodes.len()).unwrap()) };
        for key in keys {
            assert_eq!(t.query3(start, &key), scalar.query3(start, &key));
        }
    }
}
