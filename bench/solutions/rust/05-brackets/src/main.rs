use std::io::BufRead;

fn check(line: &str) -> String {
    let mut stack = Vec::new();
    let mut n = 0;
    for (i, c) in line.chars().enumerate() {
        n = i + 1;
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' | ']' | '}' => {
                let want = match c { ')' => '(', ']' => '[', _ => '{' };
                if stack.pop() != Some(want) {
                    return format!("error {}", i + 1);
                }
            }
            _ => {}
        }
    }
    if stack.is_empty() { "ok".to_string() } else { format!("error {}", n + 1) }
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", check(&line.unwrap()));
    }
}
