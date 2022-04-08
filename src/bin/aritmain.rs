
fn main() {

    let a = 1.0;
    let b = 0.0;
    let c = a / b; // inf
    let d = b / b; // NaN
    let e = c * b;
    println!("{:?}", c);
    println!("{:?}", d);
    println!("{:?}", e);
}
