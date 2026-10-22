use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let width = match args.as_slice() {
        [a] if !a.is_empty() && a.bytes().all(|b| b.is_ascii_digit()) => a.parse::<usize>().unwrap_or(0),
        _ => 0,
    };
    if width < 1 {
        println!("usage: word_wrap <width>");
        std::process::exit(2);
    }
    let mut paras: Vec<Vec<String>> = vec![vec![]];
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let words: Vec<String> = line.split([' ', '\t']).filter(|w| !w.is_empty()).map(String::from).collect();
        if words.is_empty() {
            if !paras.last().unwrap().is_empty() {
                paras.push(vec![]);
            }
        } else {
            paras.last_mut().unwrap().extend(words);
        }
    }
    let paras: Vec<&Vec<String>> = paras.iter().filter(|p| !p.is_empty()).collect();
    for (k, p) in paras.iter().enumerate() {
        if k > 0 {
            println!();
        }
        let mut line = String::new();
        for w in p.iter() {
            if line.is_empty() {
                line = w.clone();
            } else if line.chars().count() + 1 + w.chars().count() <= width {
                line.push(' ');
                line.push_str(w);
            } else {
                println!("{line}");
                line = w.clone();
            }
        }
        println!("{line}");
    }
}
