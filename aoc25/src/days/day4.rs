use std::collections::HashSet;

struct Grid {
    rolls: HashSet<(i32, i32)>,
}

impl Grid {
    fn neighbors_8((x, y): (i32, i32)) -> Vec<(i32, i32)> {
        (-1..=1)
            .flat_map(|dx| (-1..=1).map(move |dy| (x + dx, y + dy)))
            .filter(|(xx, yy)| (*xx, *yy) != (x, y))
            .collect()
    }
    fn neighs(&self, (x, y): (i32, i32)) -> i32 {
        let mut sum = 0;
        for pos in Grid::neighbors_8((x, y)) {
            if self.rolls.contains(&pos) {
                sum += 1;
            }
        }
        sum
    }
    fn new() -> Self {
        Grid {
            rolls: HashSet::new(),
        }
    }
    fn roll(&mut self, (x, y): (i32, i32)) {
        self.rolls.insert((x, y));
    }
    fn rm(&mut self, (x, y): (i32, i32)) {
        self.rolls.remove(&(x, y));
    }
}

fn part1(grid: &Grid) {
    let mut avail = 0;
    for roll in &grid.rolls {
        if grid.neighs(*roll) < 4 {
            //println!("{roll:?}");
            avail += 1;
        }
    }
    println!("Part 1: {avail}");
}
fn part2(mut grid: Grid) {
    let start_rolls = grid.rolls.len();
    while grid.rolls.len() != 0 {
        let mut deletable: Vec<(i32, i32)> = vec![];
        for roll in &grid.rolls {
            if grid.neighs(*roll) < 4 {
                deletable.push(*roll);
            }
        }
        // no change
        if deletable.is_empty() {
            break;
        }
        //needs 2nd pass due to borrow checker
        for del in deletable {
            grid.rm(del);
        }
    }
    println!("Part 2: {}", start_rolls - grid.rolls.len());
}

pub fn day4() {
    let input: &str = include_str!("../inputs/day4.txt").trim();

    let mut grid = Grid::new();

    for (y, line) in input.lines().enumerate() {
        for (x, c) in line.chars().enumerate() {
            if c == '@' {
                grid.roll((x as i32, y as i32));
            }
        }
    }

    part1(&grid);
    part2(grid);
}
