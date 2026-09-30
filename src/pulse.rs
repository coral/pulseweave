//! Pulse timestamp reconstruction from a history of period and phase estimates.

/// A period and phase anchor in seconds on the analysis timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub period: f64,
    pub time: f64,
}

/// Failure to reconstruct a pulse history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PulseError {
    /// An anchor has a nonfinite time or a nonpositive/nonfinite period.
    #[error("invalid pulse anchor at index {index}")]
    InvalidAnchor { index: usize },
    /// Floating-point precision cannot represent the next pulse timestamp.
    #[error("pulse timestamp cannot advance at anchor index {index}")]
    CannotAdvance { index: usize },
}

/// Interpolate missing pulses and skip closely spaced phase anchors.
/// Supply decoder updates in arrival order for one metrical level. Phase times
/// may move backward as estimates change. Empty input emits no pulses.
///
/// A new estimate may revise the previous list; emitted timestamps are not
/// necessarily new events. No memory is allocated by this function.
///
/// # Errors
/// Invalid anchors are rejected before emitting anything. If a timestamp cannot
/// advance, returns [`PulseError::CannotAdvance`]; earlier pulses may already
/// have been emitted. Work is proportional to the number of reconstructed pulses.
pub fn reconstruct(anchors: &[Anchor], mut emit: impl FnMut(f64)) -> Result<(), PulseError> {
    if let Some(index) = anchors
        .iter()
        .position(|a| !a.time.is_finite() || !a.period.is_finite() || a.period <= 0.)
    {
        return Err(PulseError::InvalidAnchor { index });
    }
    let mut i = 0;
    while let Some(&anchor) = anchors.get(i) {
        emit(anchor.time);
        let Some(next_anchor) = anchors.get(i + 1) else {
            break;
        };
        let limit = next_anchor.time - 0.5 * anchor.period;
        let mut multiple = 1.;
        let mut time = anchor.time + anchor.period;
        if time < limit && time <= anchor.time {
            return Err(PulseError::CannotAdvance { index: i });
        }
        while time < limit {
            emit(time);
            multiple += 1.;
            let next = anchor.time + multiple * anchor.period;
            if next <= time {
                return Err(PulseError::CannotAdvance { index: i });
            }
            time = next;
        }
        i += 1;
        while i < anchors.len()
            && anchors[i].time < anchors[i - 1].time + 0.5 * anchors[i - 1].period
        {
            i += 1;
        }
    }
    Ok(())
}
