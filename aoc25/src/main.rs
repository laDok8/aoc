fn aoc1() {
    let mut position = 50; // <0-99>
    let input = include_str!("input.txt");
    let mut hit = 0;
    let mut pass = 0;
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let instruction = &line[..1];
        let steps_total: i32 = line[1..].parse().unwrap();

        // clicks to next 0; if at 0, treat next as 100 (full wrap)
        let first_zero_d = if instruction == "R" {
            let offset_to_zero = (100 - (position % 100)) % 100;
            if offset_to_zero == 0 { 100 } else { offset_to_zero }
        } else {
            let offset_to_zero = position % 100;
            if offset_to_zero == 0 { 100 } else { offset_to_zero }
        };
        if steps_total >= first_zero_d {
            // count first hit and any full 100-click wraps after it
            pass += 1 + (steps_total - first_zero_d) / 100;
        }

        // move by remainder only
        let rem_steps: i32 = steps_total % 100;
        let signed_move = if instruction == "L" { -rem_steps } else { rem_steps};
        position = ((position + signed_move) % 100 + 100) % 100;
        hit += if position == 0 {1} else {0};
    }
    println!("Part 1: {} \nPart 2: {}", hit, pass);
}

fn main() {
    aoc1();
}
