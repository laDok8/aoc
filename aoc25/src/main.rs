fn main() {
    let day = std::env::args().nth(1).unwrap_or_else(|| "2".into());
    match day.as_str() {
        "1" | "day1" => aoc25::day1(),
        "2" | "day2" => aoc25::day2(),
        _ => eprintln!("unknown day"),
    }
}