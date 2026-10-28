use std::collections::HashMap;
use std::io::BufRead;

fn main() {
    let ws: &[char] = &[' ', '\t', '\r'];
    let mut vals: HashMap<(String, String), String> = HashMap::new();
    let mut section = String::new();
    let mut lines = std::io::stdin().lock().lines().map(|l| l.unwrap());
    for line in lines.by_ref() {
        if line == "---" {
            break;
        }
        let t = line.trim_matches(ws);
        if t.is_empty() || t.starts_with(';') || t.starts_with('#') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') && t.len() >= 2 {
            section = t[1..t.len() - 1].trim_matches(ws).to_string();
        } else if let Some((k, v)) = t.split_once('=') {
            vals.insert((section.clone(), k.trim_matches(ws).to_string()), v.trim_matches(ws).to_string());
        }
    }
    for q in lines {
        let q = q.trim_matches(ws);
        let (sec, key) = q.split_once('.').unwrap_or(("", q));
        println!("{}", vals.get(&(sec.to_string(), key.to_string())).map_or("<missing>", |v| v.as_str()));
    }
}
