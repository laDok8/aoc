use std::vec;

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Point {
    x: i32,
    y: i32,
}

struct Grid {
    // Coordinate compression
    xs: Vec<i32>,
    ys: Vec<i32>,
    // outside polygon grid marker
    is_bad: Vec<Vec<bool>>,
    prefix_sum: Vec<Vec<i32>>,
}

impl Point {
    fn rect_area(&self, p: &Point) -> i64 {
        ((p.x - self.x).abs() + 1) as i64 * ((p.y - self.y).abs() + 1) as i64
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

// instead of full 2d grid, we can focus on interesting (x,y) ranges only
fn cordinate_compress(grid: &mut Grid, points: &[Point]) {
    grid.xs = points.iter().map(|p| p.x).collect();
    grid.ys = points.iter().map(|p| p.y).collect();

    grid.xs.sort_unstable(); // Faster
    grid.xs.dedup(); // Remove duplicates
    grid.ys.sort_unstable();
    grid.ys.dedup();
}

// ray-casting algorithm to determine if point is inside polygon
fn build_bad_grid(grid: &mut Grid, points: &[Point]) {
    grid.is_bad = vec![vec![false; grid.ys.len() - 1]; grid.xs.len() - 1];

    for x_id in 0..grid.xs.len() - 1 {
        for y_id in 0..grid.ys.len() - 1 {
            // Use midpoint of the cell range, not the corner
            let x = (grid.xs[x_id] + grid.xs[x_id + 1]) / 2;
            let y = (grid.ys[y_id] + grid.ys[y_id + 1]) / 2;

            let mut intersections = 0;

            for i in 0..points.len() {
                let j = (i + 1) % points.len();
                let p1 = &points[i];
                let p2 = &points[j];

                // Check if the edge intersects the ray to the right from (x, y)
                if (p1.y > y) != (p2.y > y) {
                    let at_x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
                    if at_x > x {
                        intersections += 1;
                    }
                }
            }

            // If intersections is odd, point is inside polygon
            if intersections % 2 == 0 {
                grid.is_bad[x_id][y_id] = true;
            }
        }
    }
}

// Build prefix sum for counting bad cells in any rectangular region
fn build_prefix_sum(grid: &mut Grid) {
    grid.prefix_sum = vec![vec![0; grid.ys.len() - 1]; grid.xs.len() - 1];

    for x_id in 0..grid.xs.len() - 1 {
        for y_id in 0..grid.ys.len() - 1 {
            let curr = grid.is_bad[x_id][y_id] as i32;
            let left = if x_id > 0 {
                grid.prefix_sum[x_id - 1][y_id]
            } else {
                0
            };
            let down = if y_id > 0 {
                grid.prefix_sum[x_id][y_id - 1]
            } else {
                0
            };
            let diag = if x_id > 0 && y_id > 0 {
                grid.prefix_sum[x_id - 1][y_id - 1]
            } else {
                0
            };

            grid.prefix_sum[x_id][y_id] = curr + left + down - diag;
        }
    }
}

// Query number of bad cells in rectangle defined by two corner points
fn query_bad_count(grid: &Grid, p1: &Point, p2: &Point) -> i32 {
    let (x1, x2) = if p1.x < p2.x {
        (p1.x, p2.x)
    } else {
        (p2.x, p1.x)
    };
    let (y1, y2) = if p1.y < p2.y {
        (p1.y, p2.y)
    } else {
        (p2.y, p1.y)
    };

    let x1_id = grid.xs.iter().position(|&x| x == x1).unwrap();
    let x2_id = grid.xs.iter().position(|&x| x == x2).unwrap();
    let y1_id = grid.ys.iter().position(|&y| y == y1).unwrap();
    let y2_id = grid.ys.iter().position(|&y| y == y2).unwrap();

    let get_sum = |x: i32, y: i32| -> i32 {
        if x < 0 || y < 0 {
            return 0;
        }
        grid.prefix_sum[x as usize][y as usize]
    };

    // inclusion-exclusion principle
    get_sum(x2_id as i32 - 1, y2_id as i32 - 1)
        - get_sum(x1_id as i32 - 1, y2_id as i32 - 1)
        - get_sum(x2_id as i32 - 1, y1_id as i32 - 1)
        + get_sum(x1_id as i32 - 1, y1_id as i32 - 1)
}

fn part2(points: &[Point]) {
    let mut grid = Grid {
        xs: vec![],
        ys: vec![],
        is_bad: vec![],
        prefix_sum: vec![],
    };

    cordinate_compress(&mut grid, points);

    build_bad_grid(&mut grid, points);

    build_prefix_sum(&mut grid);

    let mut max_area = 0;

    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let p1 = &points[i];
            let p2 = &points[j];

            let bad_count = query_bad_count(&grid, p1, p2);

            // If no bad cells, rectangle is fully inside polygon
            if bad_count == 0 {
                max_area = max_area.max(p1.rect_area(p2));
            }
        }
    }

    println!("Part 2: {max_area}");
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
    part2(&points);
}
