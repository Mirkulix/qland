use std::io::BufRead;
use std::process::exit;

fn fields(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut quoted) = (Vec::new(), String::new(), false);
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out.into_iter().map(|f| f.trim_matches(' ').to_string()).collect()
}

fn integer(s: &str) -> Option<i64> {
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        println!("usage: csv_sum <column>");
        exit(2);
    }
    let rows: Vec<String> = std::io::stdin().lock().lines().map(|l| l.unwrap()).filter(|l| !l.trim().is_empty()).collect();
    let col = rows.first().and_then(|h| fields(h).iter().position(|f| *f == args[0]));
    let Some(col) = col else {
        println!("error: no column {}", args[0]);
        exit(1);
    };
    let (mut sum, mut skipped) = (0i64, 0);
    for row in &rows[1..] {
        match fields(row).get(col).and_then(|v| integer(v)) {
            Some(n) => sum += n,
            None => skipped += 1,
        }
    }
    println!("sum {sum}\nskipped {skipped}");
}
