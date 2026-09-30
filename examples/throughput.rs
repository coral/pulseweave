//! Run with `cargo run --release --example throughput`.
use pulseweave::stream::MeterStream;
use std::time::{Duration, Instant};
fn main() {
    let rate = 48_000;
    let seconds = 60;
    let mut stream = MeterStream::new(rate).unwrap();
    let mut block = [0_f32; 512];
    let mut updates = 0;
    let mut total = Duration::ZERO;
    let mut worst = Duration::ZERO;
    for start in (0..rate * seconds).step_by(512) {
        for (offset, x) in block.iter_mut().enumerate() {
            let i = start + offset;
            *x = (i as f32 * 0.13).sin() * if i % 24000 < 120 { 1. } else { 0.1 };
        }
        let now = Instant::now();
        stream.process(&block, |_| updates += 1);
        let elapsed = now.elapsed();
        total += elapsed;
        worst = worst.max(elapsed);
    }
    println!(
        "{seconds}s audio: {updates} updates, {:.3}s processing ({:.1}x realtime), worst 512-sample call {:.3}ms",
        total.as_secs_f64(),
        seconds as f64 / total.as_secs_f64(),
        worst.as_secs_f64() * 1000.
    );
}
