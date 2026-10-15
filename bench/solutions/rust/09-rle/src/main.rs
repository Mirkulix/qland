use std::io::BufRead;

fn encode(line: &str) -> String {
    let cs: Vec<char> = line.chars().collect();
    let (mut out, mut i) = (String::new(), 0);
    while i < cs.len() {
        let mut j = i;
        while j < cs.len() && cs[j] == cs[i] {
            j += 1;
        }
        out.push_str(&format!("{}{}", j - i, cs[i]));
        i = j;
    }
    out
}

fn decode(line: &str) -> Option<String> {
    let cs: Vec<char> = line.chars().collect();
    let (mut out, mut i) = (String::new(), 0);
    while i < cs.len() {
        let start = i;
        while i < cs.len() && cs[i].is_ascii_digit() {
            i += 1;
        }
        if i == start || i == cs.len() {
            return None;
        }
        let n: usize = cs[start..i].iter().collect::<String>().parse().ok()?;
        if n == 0 {
            return None;
        }
        out.extend(std::iter::repeat(cs[i]).take(n));
        i += 1;
    }
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.as_slice() {
        [m] if m == "encode" || m == "decode" => m.clone(),
        _ => {
            println!("usage: rle encode|decode");
            std::process::exit(2);
        }
    };
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let out = if mode == "encode" { encode(&line) } else { decode(&line).unwrap_or_else(|| "error".into()) };
        println!("{out}");
    }
}
