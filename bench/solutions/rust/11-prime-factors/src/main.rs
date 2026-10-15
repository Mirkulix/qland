use std::io::BufRead;

fn describe(line: &str) -> String {
    let t = line.trim_matches([' ', '\t', '\r']);
    if t.is_empty() || !t.bytes().all(|b| b.is_ascii_digit()) {
        return "error".into();
    }
    let n: u64 = match t.parse() {
        Ok(n) if (2..=1_000_000_000_000).contains(&n) => n,
        _ => return "error".into(),
    };
    let (mut n, mut d, mut parts) = (n, 2u64, Vec::new());
    while d * d <= n {
        let mut k = 0;
        while n % d == 0 {
            n /= d;
            k += 1;
        }
        if k > 0 {
            parts.push(if k > 1 { format!("{d}^{k}") } else { d.to_string() });
        }
        d += 1;
    }
    if n > 1 {
        parts.push(n.to_string());
    }
    format!("{t} = {}", parts.join(" * "))
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", describe(&line.unwrap()));
    }
}
