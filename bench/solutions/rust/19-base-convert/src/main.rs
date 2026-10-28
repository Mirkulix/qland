use std::io::BufRead;

fn convert(line: &str) -> Option<String> {
    let t: Vec<&str> = line.split(' ').collect();
    if t.len() != 3 {
        return None;
    }
    let base = |s: &str| if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) { s.parse::<u32>().ok().filter(|b| (2..=36).contains(b)) } else { None };
    let (from, to) = (base(t[0])?, base(t[1])?);
    let (neg, digits) = match t[2].strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t[2]),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for c in digits.chars() {
        v = v * from as i64 + c.to_digit(from)? as i64;
    }
    let mut out = Vec::new();
    loop {
        out.push(std::char::from_digit((v % to as i64) as u32, to)?);
        v /= to as i64;
        if v == 0 {
            break;
        }
    }
    if neg && !(out.len() == 1 && out[0] == '0') {
        out.push('-');
    }
    Some(out.iter().rev().collect())
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", convert(&line.unwrap()).unwrap_or_else(|| "error".into()));
    }
}
