#![allow(unused)]
use std::collections::{
    HashMap,
    VecDeque,
};

#[derive(Debug)]
enum ReferenceTree {
    Branch {
        children: HashMap<u8, ReferenceTree>,
        has_value: bool,
    },
    Leaf(LeafType),
}

#[derive(Debug)]
enum LeafType {
    Value,
    TreePtr,
}

impl ReferenceTree {
    fn query(&self, bytes: &[u8]) -> &'static str {
        match bytes {
            &[] => {
                match self {
                    ReferenceTree::Branch { has_value, .. } => {
                        if *has_value {
                            "match"
                        } else {
                            "partial"
                        }
                    },
                    ReferenceTree::Leaf(LeafType::Value) => "match",
                    ReferenceTree::Leaf(LeafType::TreePtr) => "tree",
                }
            },
            &[head, ref tail @ ..] => {
                match self {
                    ReferenceTree::Branch { children, .. } => {
                        match children.get(&head) {
                            Some(child) => child.query(tail),
                            None => "reject",
                        }
                    },
                    ReferenceTree::Leaf(LeafType::Value) => "reject",
                    ReferenceTree::Leaf(LeafType::TreePtr) => "tree",
                }
            },
        }
    }
}

#[derive(Debug)]
enum Terminal {
    None,
    Value,
    Tree,
}

#[derive(Debug)]
struct Node {
    parent: Option<usize>,
    label: u8,
    terminal: Terminal,
}

type Tree = Vec<Node>;

#[derive(Debug)]
enum QueryResult {
    Partial { internal_node: usize },
    Value { value_node: usize },
    PrefixTree { tree_node: usize, bytes_consumed: usize },
    None,
}

fn query(tree: &Tree, bytes: &[u8]) -> QueryResult {
    let mut label_matches = vec![];
    for node in tree {
        let matches = bytes.iter().map(|b| node.label == *b).collect::<VecDeque<_>>();
        label_matches.push(matches);
    }
    println!("label matches: {label_matches:?}\n");

    let empty = bytes.iter().map(|_| false).collect::<VecDeque<_>>();

    // Only get the match sets for nodes under the root.
    let matches0: Vec<_> = tree
        .iter()
        .enumerate()
        .map(|(i, node)| {
            if node.parent.is_none() {
                label_matches[i].clone()
            } else {
                empty.clone()
            }
        })
        .collect();

    println!("0: {matches0:?}");
    let mut matches = vec![matches0];

    for i in 0..(bytes.len() - 1) {
        let prev_matches = matches.last().unwrap();
        let next_matches = tree
            .iter()
            .enumerate()
            .map(|(i, node)| {
                if let Some(parent_ix) = node.parent {
                    let mut parent_matches = prev_matches[parent_ix].clone();

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
        println!("{}: {next_matches:?}", matches.len());
        matches.push(next_matches);
    }
    println!("");

    let mut exact_match: Vec<_> = matches
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, matches)| {
            if matches[bytes.len() - 1] {
                let terminal = &tree[i].terminal;
                if matches!(terminal, Terminal::None) {
                    return Some(QueryResult::Partial { internal_node: i });
                } else if matches!(terminal, Terminal::Value) {
                    return Some(QueryResult::Value { value_node: i });
                }
            }
            None
        })
        .collect();

    let mut prefix_match = vec![];
    for (len, node_matches) in matches.iter().enumerate() {
        for (i, matches) in node_matches.iter().enumerate() {
            let node = &tree[i];
            if matches[len] && matches!(node.terminal, Terminal::Tree) {
                prefix_match.push(QueryResult::PrefixTree { tree_node: i, bytes_consumed: len + 1 });
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
    QueryResult::None
}

#[test]
fn test_query() {
    let tree = vec![
        Node {
            parent: None,
            label: 0,
            terminal: Terminal::None,
        },
        Node {
            parent: Some(0),
            label: 1,
            terminal: Terminal::None,
        },
        Node {
            parent: Some(1),
            label: 1,
            terminal: Terminal::Tree,
        },
        Node {
            parent: Some(1),
            label: 2,
            terminal: Terminal::Value,
        },
        Node {
            parent: None,
            label: 1,
            terminal: Terminal::None,
        },
    ];
    println!("{:?}", query(&tree, &[0, 1, 2]));
}
