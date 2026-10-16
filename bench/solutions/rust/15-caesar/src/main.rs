use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shift = match args.as_slice() {
        [a] => {
            let digits = a.strip_prefix(['+', '-']).unwrap_or(a);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) { None } else { a.parse::<i64>().ok().filter(|n| n.abs() <= 1_000_000_000) }
        }
        _ => None,
    };
    let Some(shift) = shift else {
        println!("usage: caesar <shift>");
        std::process::exit(2);
    };
    let k = shift.rem_euclid(26) as u8;
    for line in std::io::stdin().lock().lines() {
        let out: String = line
            .unwrap()
            .chars()
            .map(|c| match c {
                'a'..='z' => ((c as u8 - b'a' + k) % 26 + b'a') as char,
                'A'..='Z' => ((c as u8 - b'A' + k) % 26 + b'A') as char,
                _ => c,
            })
            .collect();
        println!("{out}");
    }
}
