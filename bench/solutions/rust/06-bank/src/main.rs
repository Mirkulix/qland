use std::collections::BTreeMap;
use std::io::BufRead;

fn amount(s: &str) -> Option<i64> {
    if s.is_empty() || s.len() > 12 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok().filter(|&n| n > 0)
}

fn command(acc: &mut BTreeMap<String, i64>, line: &str) -> Result<Option<String>, String> {
    let t: Vec<&str> = line.split(' ').collect();
    let exists = |acc: &BTreeMap<String, i64>, name: &str| if acc.contains_key(name) { Ok(()) } else { Err(format!("error: no account {name}")) };
    let amt = |s: &str| amount(s).ok_or("error: invalid amount".to_string());
    match (t[0], t.len()) {
        ("open", 2) => {
            if acc.contains_key(t[1]) {
                return Err(format!("error: account {} exists", t[1]));
            }
            acc.insert(t[1].to_string(), 0);
        }
        ("deposit", 3) => {
            exists(acc, t[1])?;
            *acc.get_mut(t[1]).unwrap() += amt(t[2])?;
        }
        ("withdraw", 3) => {
            exists(acc, t[1])?;
            let n = amt(t[2])?;
            if acc[t[1]] < n {
                return Err("error: insufficient funds".into());
            }
            *acc.get_mut(t[1]).unwrap() -= n;
        }
        ("transfer", 4) => {
            exists(acc, t[1])?;
            exists(acc, t[2])?;
            let n = amt(t[3])?;
            if acc[t[1]] < n {
                return Err("error: insufficient funds".into());
            }
            *acc.get_mut(t[1]).unwrap() -= n;
            *acc.get_mut(t[2]).unwrap() += n;
        }
        ("balance", 2) => {
            exists(acc, t[1])?;
            return Ok(Some(format!("{} {}", t[1], acc[t[1]])));
        }
        _ => return Err("error: invalid command".into()),
    }
    Ok(None)
}

fn main() {
    let mut acc = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        match command(&mut acc, &line.unwrap()) {
            Ok(Some(s)) | Err(s) => println!("{s}"),
            Ok(None) => {}
        }
    }
    println!("---");
    for (name, balance) in &acc {
        println!("{name} {balance}");
    }
}
