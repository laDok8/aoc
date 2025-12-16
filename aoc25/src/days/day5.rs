use std::ops::Range;

fn part1(ranges: &Vec<Range<i64>>, ids: &Vec<i64>) -> usize {
    //count ids contained in any range
    ids.iter()
        .filter(|x| ranges.iter().any(|r| r.contains(x)))
        .count()
}

fn part2(ranges: &Vec<Range<i64>>) -> i64 {
    // Sort by start, then merge overlapping ranges
    let mut sorted: Vec<_> = ranges.iter().cloned().collect();
    sorted.sort_by_key(|r| r.start);

    let mut merged: Vec<Range<i64>> = vec![];
    for r in sorted {
        if let Some(last) = merged.last_mut() {
            if r.start <= last.end {
                // extend overlap
                last.end = last.end.max(r.end);
            } else {
                merged.push(r);
            }
        } else {
            merged.push(r);
        }
    }

    // Sum lengths of merged ranges
    merged.iter().map(|r| r.end - r.start).sum()
}

pub fn day5() {
    let input: &str = include_str!("../inputs/day5.txt").trim();
    let (ranges, ids) = input.split_once("\n\n").unwrap();
    let ranges: Vec<Range<i64>> = ranges
        .lines()
        .map(|f| {
                let (x,y) = f.split_once('-').unwrap();
                x.parse::<i64>().unwrap()..(y.parse::<i64>().unwrap() + 1)
            })
        .collect();
    let ids: Vec<i64> = ids.lines().map(|f| f.parse::<i64>().unwrap()).collect();

    println!("Part 1: {}", part1(&ranges, &ids));
    println!("Part 2: {}", part2(&ranges));
}
