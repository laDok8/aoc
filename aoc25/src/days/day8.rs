use std::collections::HashMap;

#[derive(PartialEq)]
struct Point(usize, usize, usize); // (x, y, z)

struct Node {
    point: Point,
    group_id: usize,
}

struct Edge {
    from: usize, // Index of the {`from`, `to`} nodes in the graph
    to: usize,
    weight: f64,
}

struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Graph {
    fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn dist(a: &Point, b: &Point) -> f64 {
        let dx = (a.0 as isize - b.0 as isize) as f64;
        let dy = (a.1 as isize - b.1 as isize) as f64;
        let dz = (a.2 as isize - b.2 as isize) as f64;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    // Adds a node and creates edges to all existing nodes (one way)
    fn add_node(&mut self, point: Point) {
        let new_node_idx = self.nodes.len();
        for (idx, other) in self.nodes.iter().enumerate() {
            let weight = Graph::dist(&point, &other.point);
            self.edges.push(Edge {
                from: new_node_idx,
                to: idx,
                weight,
            });
        }
        self.nodes.push(Node {
            point,
            group_id: new_node_idx,
        });
    }

    fn sort_edges(&mut self) {
        self.edges
            .sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());
    }

    /// Merges two nodes into the same group if they're in different groups.
    /// Returns true if a merge was performed.
    fn merge_groups(&mut self, from: usize, to: usize) -> bool {
        let from_group = self.nodes[from].group_id;
        let to_group = self.nodes[to].group_id;

        if from_group == to_group {
            return false;
        }

        // Update group IDs
        let new_group = from_group.min(to_group);
        for node in self.nodes.iter_mut() {
            if node.group_id == from_group || node.group_id == to_group {
                node.group_id = new_group;
            }
        }

        true
    }
}

fn part1(graph: &mut Graph, max_edges: usize) {
    let num_edges = graph.edges.len().min(max_edges);

    for i in 0..num_edges {
        let from = graph.edges[i].from;
        let to = graph.edges[i].to;

        graph.merge_groups(from, to);
    }

    // Count circuit sizes
    let mut circuit_sizes: HashMap<usize, usize> = HashMap::new();
    for node in &graph.nodes {
        *circuit_sizes.entry(node.group_id).or_insert(0) += 1;
    }

    // Three largest
    let mut sizes: Vec<usize> = circuit_sizes.values().cloned().collect();
    sizes.sort_by(|a, b| b.cmp(a));

    let result = sizes.iter().take(3).product::<usize>();
    println!("Part 1: {result}");
}

fn part2(graph: &mut Graph) {
    // Continue until all nodes are in the same group, marking result as if it was last connection point
    let mut res = 0;
    for i in 0..graph.edges.len() {
        let from = graph.edges[i].from;
        let to = graph.edges[i].to;
        if graph.merge_groups(from, to) {
            res = graph.nodes[from].point.0 * graph.nodes[to].point.0;
        }
        // Check if all nodes are in the same group
        let first_group = graph.nodes[0].group_id;
        if graph.nodes.iter().all(|n| n.group_id == first_group) {
            break;
        }
    }
    println!("Part 2: {res} ");
}

pub fn day8() {
    let input = include_str!("../inputs/day8.txt").trim();
    let mut graph = Graph::new();

    for line in input.lines() {
        let values: Vec<usize> = line
            .split(',')
            .map(|s| s.parse::<usize>())
            .collect::<Result<_, _>>()
            .unwrap();
        let point = Point(values[0], values[1], values[2]);
        graph.add_node(point);
    }

    graph.sort_edges();

    // 10 for test input
    let connections = if graph.nodes.len() > 20 { 1000 } else { 10 };
    part1(&mut graph, connections);
    // I can continue with mutated
    part2(&mut graph);
}
