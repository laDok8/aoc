fn main() {
    let day = std::env::args().nth(1).unwrap_or_else(|| "3".into());
    match day.as_str() {
        "1" | "day1" => aoc25::day1(),
        "2" | "day2" => aoc25::day2(),
        "3" | "day3" => aoc25::day3(),
        _ => eprintln!("unknown day"),
    }
}
