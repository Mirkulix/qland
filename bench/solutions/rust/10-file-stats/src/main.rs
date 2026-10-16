use std::collections::HashMap;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        println!("usage: file_stats <path>");
        exit(2);
    }
    let Ok(text) = std::fs::read_to_string(&args[0]) else {
        println!("error: cannot read {}", args[0]);
        exit(1);
    };
    let lines: Vec<&str> = text.lines().collect();
    let words: Vec<&str> = text.split([' ', '\t', '\r', '\n']).filter(|w| !w.is_empty()).collect();
    println!("lines {}", lines.len());
    println!("words {}", words.len());
    println!("chars {}", text.chars().count());
    println!("longest {}", text.split('\n').map(|l| l.chars().count()).max().unwrap_or(0));
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in &words {
        *counts.entry(w.to_ascii_lowercase()).or_default() += 1;
    }
    if let Some((w, c)) = counts.iter().min_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0))) {
        println!("top {w} {c}");
    }
}
