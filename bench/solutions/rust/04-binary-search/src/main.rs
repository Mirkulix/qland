use std::io::BufRead;

fn main() {
    let mut lines = std::io::stdin().lock().lines().map(|l| l.unwrap());
    let Some(first) = lines.next() else { return };
    let xs: Vec<i64> = first.split(' ').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    for q in lines {
        let v: i64 = q.trim().parse().unwrap();
        let i = xs.partition_point(|&x| x < v);
        println!("{}", if xs.get(i) == Some(&v) { i as i64 } else { -1 });
    }
}
