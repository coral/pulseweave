use pulseweave::period::PeriodEvidence;

#[test]
fn reset_clears_visible_evidence_and_restores_initial_behavior() {
    let mut bank = PeriodEvidence::new();
    let audio: [f64; 86] = std::array::from_fn(|i| (i as f64 * 0.31).sin());
    let initial = bank.process([&audio; 4]);
    bank.process([&audio; 4]);
    assert!(bank.strengths().0.iter().any(|&v| v > 0.));
    bank.reset();
    let (beat, bar, tatum) = bank.strengths();
    assert!(beat.iter().chain(bar).chain(tatum).all(|&v| v == 0.));
    assert_eq!(bank.phase_state(0, 10, 0), 0.);
    let restarted = bank.process([&audio; 4]);
    assert_eq!(initial.beatyness, restarted.beatyness);
    for (a, b) in initial.beat.iter().zip(restarted.beat) {
        assert_eq!((a.period, a.strength), (b.period, b.strength));
    }
}
