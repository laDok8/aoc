use std::collections::{HashMap, HashSet};

// Oriented
struct Graph {
    // We don't really need nodes for this problem
    edges: Vec<Edge>,
}

struct Edge {
    from: String,
    to: String,
}

impl Graph {
    fn get_neighbors(&self, node_id: &str) -> Vec<&str> {
        let mut neighbors = vec![];
        for edge in &self.edges {
            if edge.from == node_id {
                neighbors.push(edge.to.as_str());
            }
        }
        neighbors
    }
}

fn dfs_traverse(
    node_id: &str,
    graph: &Graph,
    visited: &mut HashSet<String>,
    memo: &mut HashMap<(String, u64), i64>,
    must_visit: &Vec<String>,
    mut must_visit_mask: u64, // bitmask of which must_visit nodes we've seen
) -> i64 {
    // Update mask if current node is one of the must_visit nodes
    for (i, required) in must_visit.iter().enumerate() {
        if node_id == required {
            must_visit_mask |= 1u64 << i;
            break;
        }
    }

    // Check if we've reached the destination
    if node_id == "out" {
        // All bits must be set (we visited all required nodes)
        let all_visited_mask = (1u64 << must_visit.len()) - 1;
        if must_visit_mask == all_visited_mask {
            return 1;
        } else {
            return 0;
        }
    }

    // Memo key is (current_node, which must_visit nodes we've seen so far)
    let memo_key = (node_id.to_string(), must_visit_mask);
    if let Some(&result) = memo.get(&memo_key) {
        return result;
    }

    visited.insert(node_id.to_string());
    let mut total_paths = 0;

    for neighbor in graph.get_neighbors(node_id) {
        if visited.contains(neighbor){
            continue;
        }

        total_paths += dfs_traverse(neighbor, graph, visited, memo, must_visit, must_visit_mask);
    }

    visited.remove(node_id);
    memo.insert(memo_key, total_paths);
    total_paths
}

fn part1(graph: &Graph) {
    // all unique paths from a start node to an end node

    let start_node_id = "you";

    let mut memo: HashMap<(String, u64), i64> = HashMap::new();
    let mut visited: HashSet<String> = HashSet::from([start_node_id.to_string()]);
    let must_visit: Vec<String> = vec![];

    let paths = dfs_traverse(
        start_node_id,
        graph,
        &mut visited,
        &mut memo,
        &must_visit,
        0,
    );
    println!("Part 1: {}", paths);
}

fn part2(graph: &Graph) {
    let start_node_id = "svr";
    let must_visit: Vec<String> = vec!["fft".to_string(), "dac".to_string()];
    let mut memo: HashMap<(String, u64), i64> = HashMap::new();
    let mut visited: HashSet<String> = HashSet::from([start_node_id.to_string()]);

    let paths = dfs_traverse(
        start_node_id,
        graph,
        &mut visited,
        &mut memo,
        &must_visit,
        0,
    );
    println!("Part 2: {}", paths);
}

pub fn day11() {
    let input = include_str!("../inputs/day11.txt").trim();

    let mut graph = Graph {
        edges: vec![],
    };

    for line in input.lines() {
        let parts: Vec<&str> = line.split(":").collect();
        let node_id = parts[0].trim();
        let mut connections = vec![];

        for c in parts[1].split_whitespace() {
            let conn = c.trim();
            connections.push(conn.to_string());
        }

        for conn in connections {
            graph.edges.push(Edge {
                from: node_id.to_string(),
                to: conn,
            });
        }
    }

    part1(&graph);
    part2(&graph);
}
