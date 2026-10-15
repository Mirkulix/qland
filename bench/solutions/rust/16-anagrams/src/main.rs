use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;

fn main() {
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let w = line.unwrap().trim_matches([' ', '\t', '\r']).to_string();
        if w.is_empty() {
            continue;
        }
        let mut key: Vec<char> = w.chars().collect();
        key.sort_unstable();
        groups.entry(key.into_iter().collect()).or_default().insert(w);
    }
    let mut lines: Vec<String> = groups.values().filter(|g| g.len() >= 2).map(|g| g.iter().cloned().collect::<Vec<_>>().join(" ")).collect();
    lines.sort();
    for l in lines {
        println!("{l}");
    }
}
