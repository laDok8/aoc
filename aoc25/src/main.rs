fn main() {
    let day = std::env::args().nth(1).unwrap_or_else(|| "8".into());
    match day.as_str() {
        "1" | "day1" => aoc25::day1(),
        "2" | "day2" => aoc25::day2(),
        "3" | "day3" => aoc25::day3(),
        "4" | "day4" => aoc25::day4(),
        "5" | "day5" => aoc25::day5(),
        "6" | "day6" => aoc25::day6(),
        "7" | "day7" => aoc25::day7(),
        "8" | "day8" => aoc25::day8(),
        _ => eprintln!("unknown day"),
    }
}
