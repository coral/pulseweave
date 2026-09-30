use pulseweave::pulse::{Anchor, PulseError, reconstruct};

#[test]
fn interpolate_missing_pulses_and_skip_near_duplicate_anchors() {
    let anchors = [
        Anchor {
            time: 0.,
            period: 0.5,
        },
        Anchor {
            time: 0.125,
            period: 0.5,
        },
        Anchor {
            time: 1.,
            period: 0.5,
        },
        Anchor {
            time: 3.,
            period: 0.5,
        },
    ];
    let mut pulses = Vec::new();
    reconstruct(&anchors, |t| pulses.push(t)).unwrap();
    assert_eq!(pulses, [0., 1., 1.5, 2., 2.5, 3.]);
}

#[test]
fn empty_history_emits_nothing() {
    reconstruct(&[], |_| panic!("empty history emitted a pulse")).unwrap();
}

#[test]
fn invalid_input_is_rejected_before_emission() {
    for invalid in [
        Anchor {
            time: f64::NAN,
            period: 1.,
        },
        Anchor {
            time: 0.,
            period: 0.,
        },
        Anchor {
            time: 0.,
            period: -1.,
        },
        Anchor {
            time: 0.,
            period: f64::INFINITY,
        },
    ] {
        let anchors = [
            Anchor {
                time: 0.,
                period: 0.5,
            },
            invalid,
        ];
        assert_eq!(
            reconstruct(&anchors, |_| panic!("invalid input emitted a pulse")),
            Err(PulseError::InvalidAnchor { index: 1 })
        );
    }
}

#[test]
fn unrepresentable_step_reports_error_without_duplicate_emission() {
    let anchors = [
        Anchor {
            time: 1e20,
            period: 0.5,
        },
        Anchor {
            time: 2e20,
            period: 0.5,
        },
    ];
    let mut pulses = Vec::new();
    assert_eq!(
        reconstruct(&anchors, |t| pulses.push(t)),
        Err(PulseError::CannotAdvance { index: 0 })
    );
    assert_eq!(pulses, [1e20]);
}
