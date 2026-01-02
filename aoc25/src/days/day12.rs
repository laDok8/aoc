const BOX_SIZE: i32 = 3;

fn part1(input: &str) {
    let fitting_count = input
        .lines()
        .filter(|line| {
            let (dimensions, present_ids) = line.split_once(':').unwrap();

            let capacity = dimensions
                .split('x')
                .map(|dim| dim.trim().parse::<i32>().unwrap() / BOX_SIZE)
                .product::<i32>();

            let total_presents: i32 = present_ids
                .split_whitespace()
                .map(|id| id.parse::<i32>().unwrap())
                .sum();

            total_presents <= capacity
        })
        .count();

    println!("Part 1: {fitting_count}");
}

pub fn day12() {
    // First puzzle not working on example input :/
    let input = include_str!("../inputs/day12.txt").trim();

    let regions = input.split("\n\n").last().unwrap_or_default();

    part1(regions);
}
