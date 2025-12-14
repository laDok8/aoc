fn part1(inp: std::str::Lines) {
    let mut jolts_sum = 0;
    for line in inp {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let nums: Vec<u32> = line.chars().map(|c| c.to_digit(10).unwrap()).collect();

        let mut first_jolt = 0;
        let mut second_jolt = 0;

        for num in nums {
            // solve start - initialize with first two numbers
            if first_jolt == 0 {
                first_jolt = num;
                continue;
            } else if second_jolt == 0 {
                second_jolt = num;
                continue;
            }

            //treat each jolt separe
            if first_jolt < second_jolt {
                first_jolt = second_jolt;
                second_jolt = num;
            } else if num > second_jolt {
                second_jolt = num;
            }
        }
        let res = first_jolt * 10 + second_jolt;
        jolts_sum += res;
    }
    println!("Part 1: {jolts_sum}");
}

fn part2(inp: std::str::Lines) {
    let mut jolts_sum = 0;
    for line in inp {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let nums: Vec<u32> = line.chars().map(|c| c.to_digit(10).unwrap()).collect();

        let mut jolts = [0; 12];

        for num in nums {
            // solve start - initialize with first 12 numbers
            let mut changed = false;
            for i in 0..jolts.len() {
                if jolts[i] == 0 {
                    jolts[i] = num;
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }

            //treat each jolt separe
            for i in 0..jolts.len() - 1 {
                if jolts[i] < jolts[i + 1] {
                    //need to update whole train
                    for j in i..jolts.len() - 1 {
                        jolts[j] = jolts[j + 1];
                    }
                    jolts[jolts.len() - 1] = num;
                    //can do only once
                    break;
                }
            }
            // if no update last can still move
            if num > jolts[jolts.len() - 1] {
                jolts[jolts.len() - 1] = num
            }
        }
        let res: u64 = jolts.iter().fold(0, |acc, &d| acc * 10 + d as u64);
        jolts_sum += res;
    }
    println!("Part 2: {jolts_sum}");
}

pub fn day3() {
    let input: &str = include_str!("../inputs/day3.txt");
    part1(input.lines());
    part2(input.lines());
}
