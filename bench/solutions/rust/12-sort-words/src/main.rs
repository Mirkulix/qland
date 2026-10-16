use std::collections::BTreeSet;
use std::io::Read;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let words: BTreeSet<String> = text.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()).map(|w| w.to_ascii_lowercase()).collect();
    for w in words {
        println!("{w}");
    }
}
