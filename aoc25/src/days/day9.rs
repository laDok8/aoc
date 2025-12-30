#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn rect_area(&self, p: &Point) -> i64 {
        (p.x - self.x + 1 ).abs() as i64 * (p.y - self.y + 1).abs() as i64
    }
}

fn part1(points: &[Point]) {
    let mut max_area = 0;

    for i in 0..points.len() {
        for j in i + 1..points.len() {
            max_area = max_area.max(points[i].rect_area(&points[j]));
        }
    }

    println!("Part 1: {max_area}");
}

fn part2(_input: &str) {
    todo!("Part 2 not implemented yet");
}

pub fn day9() {
    let input = include_str!("../inputs/day9.txt").trim();

    let points: Vec<Point> = input
        .lines()
        .map(|line| {
            let (x, y) = line.split_once(',').unwrap();
            Point {
                x: x.parse().unwrap(),
                y: y.parse().unwrap(),
            }
        })
        .collect();

    part1(&points);
    part2(input);
}
