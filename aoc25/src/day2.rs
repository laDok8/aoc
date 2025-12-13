fn part1() {
    let input = include_str!("input.txt");
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
            let (f,e) = num_str.split_at(str_len);
            if f == e {
                invalid_sum += num;
            }
        }
    }
    print!("Part 1: {invalid_sum}");
}
fn part2() {
    let input = include_str!("input.txt");
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
            //lets try all possible splits <1-N/2> can be N times
            for split_len in 1..=(num_str.len() / 2){
                let (f,_) = num_str.split_at(split_len);
                // should get exactly N/split_len matches
                let matches: Vec<&str> = num_str.matches(f).collect();
                if matches.len() == num_str.len()/split_len && (num_str.len()%split_len)==0 {
                    invalid_sum += num;
                    break;
                }

            }
        }
    }
    print!("Part 2: {invalid_sum}");
}



fn main() {
    part1();
    part2();
}
