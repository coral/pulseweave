use pulseweave::decoder::{PeriodDecoder, PeriodEstimate, PhaseDecoder};
use pulseweave::period::{CANDIDATES, Candidate, PeriodCandidates, PeriodEvidence};

#[test]
fn missing_candidates_keep_scores_finite_and_allow_recovery() {
    let empty = PeriodCandidates {
        beat: [Candidate::default(); CANDIDATES],
        bar: [Candidate::default(); CANDIDATES],
        tatum: [Candidate::default(); CANDIDATES],
        beatyness: 0.,
    };
    let mut decoder = PeriodDecoder::new();
    for _ in 0..10 {
        let estimate = decoder.process(empty);
        assert!(decoder.scores().iter().all(|s| s.is_finite()));
        assert!(estimate.beat.is_finite() && estimate.beat > 0.);
    }
    let signal = PeriodCandidates {
        beat: [Candidate {
            period: 0.5,
            strength: 1.,
        }; CANDIDATES],
        bar: [Candidate {
            period: 2.,
            strength: 1.,
        }; CANDIDATES],
        tatum: [Candidate {
            period: 0.25,
            strength: 1.,
        }; CANDIDATES],
        beatyness: 1.,
    };
    assert_eq!(
        decoder.process(signal),
        PeriodEstimate {
            beat: 0.5,
            bar: 2.,
            tatum: 0.25
        }
    );
    assert!(decoder.scores().iter().all(|s| s.is_finite()));
}

#[test]
fn phase_fallback_handles_a_one_sample_period() {
    let mut phase = PhaseDecoder::new();
    let bank = PeriodEvidence::new();
    let periods = PeriodEstimate {
        beat: 1e-6,
        bar: 1e-6,
        tatum: 1e-6,
    };
    let mut previous = f64::NEG_INFINITY;
    for elapsed in [86, 172, 258] {
        let result = phase.process(&bank, periods, elapsed);
        assert!(result.beat_time.is_finite() && result.bar_time.is_finite());
        assert!(result.beat_time > previous);
        previous = result.beat_time;
    }
}
