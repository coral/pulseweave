//! Stage timings with deterministic generated input. Run in release mode.
use pulseweave::{
    decoder::{PeriodDecoder, PhaseDecoder},
    onset::OnsetAnalyzer,
    period::PeriodEvidence,
};
use std::{hint::black_box, time::Instant};
fn main() {
    let audio: Vec<_> = (0..44100 * 60).map(|i| (i as f64 * 0.13).sin()).collect();
    let mut onset = OnsetAnalyzer::new();
    let start = Instant::now();
    onset.process(&audio, |frame| {
        black_box(frame);
    });
    println!("onset 60s: {:.3} ms", start.elapsed().as_secs_f64() * 1000.);
    let mut bank = PeriodEvidence::new();
    let mut periods = PeriodDecoder::new();
    let mut phase = PhaseDecoder::new();
    let groups: [[f64; 86]; 4] =
        std::array::from_fn(|g| std::array::from_fn(|i| ((i + g) as f64 * 0.13).sin()));
    let mut elapsed = [std::time::Duration::ZERO; 3];
    for i in 1..=1200 {
        let start = Instant::now();
        let candidates = bank.process(std::array::from_fn(|g| &groups[g][..]));
        elapsed[0] += start.elapsed();
        let start = Instant::now();
        let estimate = periods.process(black_box(candidates));
        elapsed[1] += start.elapsed();
        let start = Instant::now();
        black_box(phase.process(&bank, estimate, i * 86));
        elapsed[2] += start.elapsed();
    }
    for (name, time) in ["period evidence", "period decoder", "phase decoder"]
        .into_iter()
        .zip(elapsed)
    {
        println!(
            "{name}: {:.3} ms / update",
            time.as_secs_f64() * 1000. / 1200.
        );
    }
}
