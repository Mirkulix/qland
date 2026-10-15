use std::io::BufRead;

const SYMBOLS: [(u32, &str); 13] = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
    (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
];

fn to_roman(mut n: u32) -> String {
    let mut s = String::new();
    for (v, r) in SYMBOLS {
        while n >= v {
            s.push_str(r);
            n -= v;
        }
    }
    s
}

fn from_roman(s: &str) -> Option<u32> {
    let value = |c| match c {
        'I' => Some(1), 'V' => Some(5), 'X' => Some(10), 'L' => Some(50),
        'C' => Some(100), 'D' => Some(500), 'M' => Some(1000), _ => None,
    };
    let vals: Vec<i64> = s.chars().map(value).collect::<Option<Vec<_>>>()?;
    let mut total = 0i64;
    for (i, v) in vals.iter().enumerate() {
        if vals.get(i + 1).is_some_and(|n| v < n) { total -= v } else { total += v }
    }
    if (1..=3999).contains(&total) && to_roman(total as u32) == s { Some(total as u32) } else { None }
}

fn convert(line: &str) -> String {
    let t = line.trim_matches([' ', '\t', '\r']);
    if !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(n) = t.parse::<u64>() {
            if (1..=3999).contains(&n) {
                return to_roman(n as u32);
            }
        }
    }
    from_roman(t).map_or("error".to_string(), |v| v.to_string())
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", convert(&line.unwrap()));
    }
}
