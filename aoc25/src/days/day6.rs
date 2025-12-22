fn part1(inp: &str) {
    let mut matrix: Vec<Vec<i64>> = vec![];

    let (nums, ops) = inp.rsplit_once("\n").unwrap();
    let ops: Vec<char> = ops
        .split_whitespace()
        .map(|s| s.chars().next().unwrap())
        .collect();

    for line in nums.lines() {
        let v: Vec<i64> = line
            .split_whitespace()
            .map(|s| s.parse::<i64>().unwrap())
            .collect();
        matrix.push(v);
    }

    let mut sum = 0;
    for (i, op) in ops.iter().enumerate() {
        let rest = matrix.iter().flat_map(|v| v.get(i)).fold(
            if *op == '*' { 1 } else { 0 },
            |acc, num| match op {
                '*' => acc * num,
                '+' => acc + num,
                _ => acc,
            },
        );
        sum += rest;
    }
    println!("Part 1: {sum}");
}

fn part2(inp: &str) {
    let mut sum: i64 = 0;
    let line_count = inp.lines().count();
    let mut nums: Vec<i64> = vec![];

    let mut symbols = vec![];
    for line in inp.lines() {
        symbols.push(line.as_bytes());
    }

    // right to left
    for x in (0..symbols.get(0).unwrap().iter().len()).rev() {
        let mut num: i64 = 0;

        for y in 0..line_count {
            let symb = match symbols.get(y).unwrap().get(x) {
                Some(b' ') => continue,
                Some(x) => x,
                None => continue,
            };

            if *symb == b'*' || *symb == b'+' {
                let cur_op = *symb;

                if num != 0 {
                    nums.push(num);
                    num = 0;
                }

                if !nums.is_empty() {
                    let rest = nums
                        .iter()
                        .fold(
                            if cur_op == b'*' { 1 } else { 0 },
                            |acc, &num| match cur_op {
                                b'*' => acc * num,
                                b'+' => acc + num,
                                _ => acc,
                            },
                        );
                    sum += rest;
                }

                nums.clear();
            } else {
                let symb = symb - b'0';
                num = num * 10 + symb as i64;
            }
        }

        if num != 0 {
            nums.push(num);
        }
    }

    println!("Part 2: {sum}");
}

pub fn day6() {
    let input: &str = include_str!("../inputs/day6.txt");

    part1(input);
    part2(input);
}
