use rayon::prelude::*;

struct Machine {
    lights_target: Vec<bool>,
    lights_cost: Vec<usize>,
    buttons: Vec<Button>,
}

struct Button {
    toggle_indices: Vec<usize>,
}

fn parse_bracketed(s: &str) -> Vec<usize> {
    s[1..s.len() - 1]
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect()
}

fn parse_machine(line: &str) -> Machine {
    let mut parts = line.split_whitespace();

    let lights_target = parts
        .next()
        .unwrap()
        .chars()
        .filter(|&c| c == '#' || c == '.')
        .map(|c| c == '#')
        .collect();

    let (toggles, jolts): (Vec<_>, Vec<_>) = parts.partition(|s| s.starts_with('('));

    let buttons = toggles
        .into_iter()
        .map(|s| Button {
            toggle_indices: parse_bracketed(s),
        })
        .collect();

    let lights_cost = jolts.first().map(|s| parse_bracketed(s)).unwrap();

    Machine {
        lights_target,
        lights_cost,
        buttons,
    }
}

fn part1(machines: &Vec<Machine>) {
    let mut press_sum = 0;

    for machine in machines {
        let combinations = 1 << machine.buttons.len();
        let mut min_ones = combinations;

        for combo in 0u32..combinations {
            let mut lights = vec![false; machine.lights_target.len()];
            let hamings_combo_weight = combo.count_ones();

            if hamings_combo_weight >= min_ones {
                continue;
            }

            for (i, button) in machine.buttons.iter().enumerate() {
                if (combo & (1 << i)) == 0 {
                    continue;
                }
                for &index in &button.toggle_indices {
                    lights[index] = !lights[index];
                }
            }

            if lights == machine.lights_target {
                min_ones = hamings_combo_weight;
            }
        }
        press_sum += min_ones;
    }
    println!("Part 1: {}", press_sum);
}

fn solve_machine_part2_z3(machine: &Machine) -> usize {
    use z3::{Optimize, ast::Int};

    let opt = Optimize::new();

    // Create a variable for each button (number of presses)
    let button_vars: Vec<Int> = (0..machine.buttons.len())
        .map(|i| Int::new_const(format!("b{}", i)))
        .collect();

    let zero = Int::from_i64(0);

    // Each button press count must be >= 0
    for var in &button_vars {
        opt.assert(&var.ge(&zero));
    }

    // For each light, the sum of presses from buttons that toggle it must equal the target cost
    for (light_idx, &target_cost) in machine.lights_cost.iter().enumerate() {
        let target = Int::from_i64(target_cost as i64);

        // Sum up presses from all buttons that toggle this light
        let contributing_presses: Vec<&Int> = machine
            .buttons
            .iter()
            .enumerate()
            .filter(|(_, button)| button.toggle_indices.contains(&light_idx))
            .map(|(btn_idx, _)| &button_vars[btn_idx])
            .collect();

        let sum = Int::add(&contributing_presses);
        opt.assert(&sum.eq(&target));
    }

    // Minimize total presses
    let refs: Vec<&Int> = button_vars.iter().collect();
    let total_presses = Int::add(&refs);
    opt.minimize(&total_presses);

    match opt.check(&[]) {
        z3::SatResult::Sat => {
            let model = opt.get_model().unwrap();
            let result: i64 = button_vars
                .iter()
                .map(|v| model.eval(v, true).unwrap().as_i64().unwrap())
                .sum();
            result as usize
        }
        _ => usize::MAX,
    }
}

fn part2(machines: &Vec<Machine>) {
    // now instead of binary target we target cost -> more than one press, less than machine.lights_cost.max()
    // also parallel for funsies
    let press_sum: usize = machines.par_iter().map(solve_machine_part2_z3).sum();

    println!("Part 2: {}", press_sum);
}

pub fn day10() {
    let input = include_str!("../inputs/day10.txt").trim();

    let machines: Vec<Machine> = input.lines().map(parse_machine).collect();

    part1(&machines);
    part2(&machines);
}
