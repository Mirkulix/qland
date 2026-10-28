use std::io::Read;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let lines = text.matches('\n').count();
    let words = text.split([' ', '\t', '\r', '\n']).filter(|w| !w.is_empty()).count();
    println!("{} {} {}", lines, words, text.chars().count());
}
