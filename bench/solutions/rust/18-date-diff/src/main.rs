use std::io::BufRead;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe
}

fn date(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| b[r.clone()].iter().all(|c| c.is_ascii_digit()).then(|| s[r].parse::<i64>().unwrap());
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dim = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if y < 1 || !(1..=12).contains(&m) || d < 1 || d > dim[m as usize - 1] {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split(' ').collect();
        let r = if t.len() == 2 { date(t[0]).zip(date(t[1])).map(|(a, b)| (b - a).to_string()) } else { None };
        println!("{}", r.unwrap_or_else(|| "error".into()));
    }
}
