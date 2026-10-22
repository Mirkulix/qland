use std::io::BufRead;

fn tokenize(line: &str) -> String {
    let cs: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        let run = |i: usize, ok: &dyn Fn(char) -> bool| {
            let mut j = i;
            while j < cs.len() && ok(cs[j]) {
                j += 1;
            }
            j
        };
        if c == ' ' || c == '\t' {
            i += 1;
        } else if c.is_ascii_digit() {
            let j = run(i, &|c| c.is_ascii_digit());
            toks.push(format!("NUM({})", cs[i..j].iter().collect::<String>()));
            i = j;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let j = run(i, &|c| c.is_ascii_alphanumeric() || c == '_');
            toks.push(format!("ID({})", cs[i..j].iter().collect::<String>()));
            i = j;
        } else if "+-*/%".contains(c) {
            toks.push(format!("OP({c})"));
            i += 1;
        } else if c == '(' || c == ')' {
            toks.push(if c == '(' { "LP" } else { "RP" }.to_string());
            i += 1;
        } else {
            return format!("error at {}: '{}'", i + 1, c);
        }
    }
    toks.join(" ")
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", tokenize(&line.unwrap()));
    }
}
