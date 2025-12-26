use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

struct Manifold {
    start: Point,
    splits: HashSet<Point>,
}

fn part1(man: &Manifold, max_y: i32) {
    let mut hits = 0;
    //simulate flow from start to splits (downward only)
    let mut flow_points: HashSet<Point> = HashSet::new();
    flow_points.insert(man.start);
    for _ in 0..=max_y {
        let mut next_flow: HashSet<Point> = HashSet::new();
        //either increase Y or split and delete old one
        for cur_point in flow_points.iter() {
            let hit = man.splits.iter().filter(|p| p == &cur_point).count() != 0;
            if hit {
                next_flow.insert(Point {
                    x: cur_point.x - 1,
                    y: cur_point.y + 1,
                });
                next_flow.insert(Point {
                    x: cur_point.x + 1,
                    y: cur_point.y + 1,
                });
                hits += 1;
            } else {
                next_flow.insert(Point {
                    x: cur_point.x,
                    y: cur_point.y + 1,
                });
            }
        }
        // update for next Y
        flow_points = next_flow;
    }
    println!("Part 1: {hits}");
}

fn part2(man: &Manifold, max_y: i32) {
    let mut memo: HashMap<(i32, i32), i64> = HashMap::new();

    fn dp(point: Point, man: &Manifold, max_y: i32, memo: &mut HashMap<(i32, i32), i64>) -> i64 {
        // Base case: If the point is out of bounds, no timelines are added
        if point.y > max_y {
            return 0;
        }

        // Check if the result is already computed
        if let Some(&result) = memo.get(&(point.x, point.y)) {
            return result;
        }

        // Check if the current point hits a split
        let hit = man.splits.iter().any(|&p| p == point);

        let result = if hit {
            // If it hits, split into two timelines
            1 + dp(
                Point {
                    x: point.x - 1,
                    y: point.y + 1,
                },
                man,
                max_y,
                memo,
            ) + dp(
                Point {
                    x: point.x + 1,
                    y: point.y + 1,
                },
                man,
                max_y,
                memo,
            )
        } else {
            // Otherwise, continue downward
            dp(
                Point {
                    x: point.x,
                    y: point.y + 1,
                },
                man,
                max_y,
                memo,
            )
        };

        // Store the result in the memoization table
        memo.insert((point.x, point.y), result);
        result
    }

    let timelines = 1 + dp(man.start, man, max_y, &mut memo);
    println!("Part 2: {timelines}");
}

pub fn day7() {
    let input: &str = include_str!("../inputs/day7.txt").trim();
    let mut manifold = Manifold {
        start: Point { x: 0, y: 0 },
        splits: HashSet::new(),
    };

    for (y, line) in input.lines().enumerate() {
        for (x, c) in line.chars().enumerate() {
            match c {
                'S' => {
                    manifold.start = Point {
                        x: x as i32,
                        y: y as i32,
                    };
                }
                '^' => {
                    let split = Point {
                        x: x as i32,
                        y: y as i32,
                    };
                    manifold.splits.insert(split);
                }
                _ => {}
            }
        }
    }
    let max_y = manifold.splits.iter().map(|p| p.y).max().unwrap_or(0);

    part1(&manifold, max_y);
    part2(&manifold, max_y);
}
