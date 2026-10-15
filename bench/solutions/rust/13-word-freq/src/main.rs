use std::collections::HashMap;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k = match args.as_slice() {
        [a] if !a.is_empty() && a.bytes().all(|b| b.is_ascii_digit()) => a.parse::<usize>().unwrap_or(0),
        _ => 0,
    };
    if k < 1 {
        println!("usage: word_freq <k>");
        std::process::exit(2);
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in text.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()) {
        *counts.entry(w.to_ascii_lowercase()).or_default() += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (w, c) in ranked.into_iter().take(k) {
        println!("{c} {w}");
    }
}
