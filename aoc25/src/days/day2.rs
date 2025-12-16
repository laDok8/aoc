use onig::Regex;

fn part1() {
    let input = include_str!("../inputs/day2.txt");
    let mut invalid_sum = 0;
    for line in input.split(",") {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let range = line.split_once("-").unwrap();
        let start: i64 = range.0.parse().unwrap();
        let end: i64 = range.1.parse().unwrap();

        for num in start..=end {
            let num_str = num.to_string();
            let str_len = num_str.len() / 2;
            let (f, e) = num_str.split_at(str_len);
            if f == e {
                invalid_sum += num;
            }
        }
    }
    println!("Part 1: {invalid_sum}");
}

fn part2() {
    let input = include_str!("../inputs/day2.txt");
    let mut invalid_sum = 0;
    for line in input.split(",") {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let range = line.split_once("-").unwrap();
        let start: i64 = range.0.parse().unwrap();
        let end: i64 = range.1.parse().unwrap();

        for num in start..=end {
            let num_str = num.to_string();
            // let's match via onig (supports backrefs)
            let re = Regex::new(r"^(.+?)\1+$").unwrap();
            if re.is_match(&num_str) {
                invalid_sum += num;
            }
        }
    }
    println!("Part 2: {invalid_sum}");
}

pub fn day2() {
    part1();
    part2();
}
