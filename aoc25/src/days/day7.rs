use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct Point {
    x: usize,
    y: usize,
}

struct Manifold {
    start: Point,
    splits: HashSet<Point>,
    max_y: usize,
}

fn part1(man: &Manifold) {
    let mut hits = 0;
    let mut flow_points: HashSet<Point> = HashSet::from([man.start]);

    for _ in 0..=man.max_y {
        flow_points = flow_points
            .iter()
            .flat_map(|&cur_point| {
                if man.splits.contains(&cur_point) {
                    hits += 1;
                    vec![
                        Point {
                            x: cur_point.x - 1,
                            y: cur_point.y + 1,
                        },
                        Point {
                            x: cur_point.x + 1,
                            y: cur_point.y + 1,
                        },
                    ]
                } else {
                    vec![Point {
                        x: cur_point.x,
                        y: cur_point.y + 1,
                    }]
                }
            })
            .collect();
    }

    println!("Part 1: {hits}");
}

fn part2(man: &Manifold) {
    let mut memo: HashMap<Point, i64> = HashMap::new();

    fn flow(point: Point, man: &Manifold, memo: &mut HashMap<Point, i64>) -> i64 {
        if point.y > man.max_y {
            return 0;
        }

        if let Some(&result) = memo.get(&point) {
            return result;
        }

        let result = if man.splits.contains(&point) {
            1 + flow(
                Point {
                    x: point.x - 1,
                    y: point.y + 1,
                },
                man,
                memo,
            ) + flow(
                Point {
                    x: point.x + 1,
                    y: point.y + 1,
                },
                man,
                memo,
            )
        } else {
            flow(
                Point {
                    x: point.x,
                    y: point.y + 1,
                },
                man,
                memo,
            )
        };

        memo.insert(point, result);
        result
    }

    let timelines = 1 + flow(man.start, man, &mut memo);
    println!("Part 2: {timelines}");
}

pub fn day7() {
    let input = include_str!("../inputs/day7.txt").trim();
    let mut manifold = Manifold {
        start: Point { x: 0, y: 0 },
        splits: HashSet::new(),
        max_y: 0,
    };

    input.lines().enumerate().for_each(|(y, line)| {
        line.chars().enumerate().for_each(|(x, c)| match c {
            'S' => manifold.start = Point { x, y },
            '^' => {
                manifold.splits.insert(Point { x, y });
            }
            _ => {}
        });
    });

    manifold.max_y = manifold.splits.iter().map(|p| p.y).max().unwrap_or(0);

    part1(&manifold);
    part2(&manifold);
}
