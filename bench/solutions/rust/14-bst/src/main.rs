use std::io::Read;

#[derive(Default)]
struct Tree(Option<Box<(Tree, i64, Tree)>>);

impl Tree {
    fn insert(&mut self, v: i64) {
        match &mut self.0 {
            None => self.0 = Some(Box::new((Tree::default(), v, Tree::default()))),
            Some(n) if v < n.1 => n.0.insert(v),
            Some(n) if v > n.1 => n.2.insert(v),
            Some(_) => {}
        }
    }
    fn height(&self) -> usize {
        self.0.as_ref().map_or(0, |n| 1 + n.0.height().max(n.2.height()))
    }
    fn walk(&self, order: u8, out: &mut Vec<i64>) {
        if let Some(n) = &self.0 {
            if order == 0 { out.push(n.1) }
            n.0.walk(order, out);
            if order == 1 { out.push(n.1) }
            n.2.walk(order, out);
            if order == 2 { out.push(n.1) }
        }
    }
}

fn line(name: &str, t: &Tree, order: u8) -> String {
    let mut xs = Vec::new();
    t.walk(order, &mut xs);
    std::iter::once(name.to_string()).chain(xs.iter().map(|x| x.to_string())).collect::<Vec<_>>().join(" ")
}

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut t = Tree::default();
    for v in text.split_whitespace() {
        t.insert(v.parse().unwrap());
    }
    let mut ino = Vec::new();
    t.walk(1, &mut ino);
    println!("size {}", ino.len());
    println!("height {}", t.height());
    println!("{}", line("inorder", &t, 1));
    println!("{}", line("preorder", &t, 0));
    println!("{}", line("postorder", &t, 2));
}
