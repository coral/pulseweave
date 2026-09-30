use pulseweave::dsp::{Resonator, peak_pick};

#[test]
fn resonator_phase_wraps_even_for_large_indices() {
    let mut resonator = Resonator::new(3, 0.5);
    resonator.process(1.);
    resonator.process(0.);
    assert_eq!(
        resonator.phase_state(usize::MAX),
        resonator.phase_state(usize::MAX % 3)
    );
}

#[test]
fn peak_selection_handles_empty_output_plateaus_and_endpoints() {
    assert_eq!(peak_pick(&[1., 2., 1.], &mut []), 0);
    let mut output = [0; 4];
    assert_eq!(peak_pick(&[3., 1., 2., 2., 1., 4.], &mut output), 2);
    assert_eq!(output, [5, 0, 6, 6]);
    assert_eq!(peak_pick(&[], &mut output), 0);
    assert_eq!(output, [0; 4]);
}

#[test]
fn block_resonator_matches_sample_processing_across_wraps() {
    // Short periods exercise feedback within a block; odd lengths exercise
    // vector remainders. No expected output depends on an external fixture.
    let signal: Vec<_> = (0..2049)
        .map(|i| ((i * 37 % 101) as f64 - 50.) / 50.)
        .collect();
    for period in [1, 2, 3, 7, 16, 85, 86, 87, 128, 689] {
        for feedback in [0.01, 0.5, 0.999] {
            let mut samplewise = Resonator::new(period, feedback);
            let mut blocked = Resonator::new(period, feedback);
            for block in signal.chunks(93) {
                for &sample in block {
                    samplewise.process(sample);
                }
                blocked.process_block(block);
                for phase in 0..period {
                    assert_eq!(
                        blocked.phase_state(phase),
                        samplewise.phase_state(phase),
                        "period {period}, phase {phase}"
                    );
                }
                let energy = (0..period)
                    .map(|phase| samplewise.phase_state(phase).powi(2))
                    .sum::<f64>();
                assert_eq!(blocked.state_energy(), energy);
            }
            blocked.process_block(&[]);
            blocked.reset();
            assert_eq!(blocked.state_energy(), 0.);
        }
    }
}
