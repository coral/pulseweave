//! Temporal decoding of joint bar/beat/tatum periods and pulse phases.
use crate::dsp::{
    conditional_period, pattern_phase, peak_pick, period_transition, phase_transition,
};
use crate::onset::ACCENT_RATE;
use crate::period::{CANDIDATES, Candidate, GROUPS, LAGS, PeriodCandidates, PeriodEvidence};
const STATES: usize = CANDIDATES * CANDIDATES * CANDIDATES;
const FLOOR: f64 = 1e-60;
const WEIGHTS: [f64; 9] = [
    0.9020073524848866,
    1.,
    0.7302768106235606,
    1.1438508100093516,
    0.443270652313484,
    0.8863431230976493,
    0.43706546371614174,
    0.796691092092133,
    0.6586692681470597,
];

#[derive(Clone, Copy, Debug, PartialEq)]
/// Selected beat, bar, and tatum durations in seconds.
pub struct PeriodEstimate {
    pub beat: f64,
    pub bar: f64,
    pub tatum: f64,
}

/// Selects a joint period estimate while accumulating temporal transition scores.
pub struct PeriodDecoder {
    previous: Option<PeriodCandidates>,
    scores: [f64; STATES],
}
impl Default for PeriodDecoder {
    fn default() -> Self {
        Self::new()
    }
}
impl PeriodDecoder {
    /// Initialize the joint period decoder with no preceding candidate history.
    pub fn new() -> Self {
        Self {
            previous: None,
            scores: [0.; STATES],
        }
    }
    /// Clear accumulated scores and preceding candidates.
    pub fn reset(&mut self) {
        self.previous = None;
        self.scores.fill(0.);
    }
    /// Return the 125 joint-state scores; beat varies fastest, then bar, then tatum.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }
    /// Score candidate combinations using period ratios and temporal transitions.
    /// Returns the strongest joint estimate, substituting seven seconds for zero periods.
    pub fn process(&mut self, candidates: PeriodCandidates) -> PeriodEstimate {
        let mut next_scores = [0.; STATES];
        let mut transitions = [[[0.; CANDIDATES]; CANDIDATES]; 3];
        if let Some(previous) = &self.previous {
            for (level, (old, new)) in [&previous.beat, &previous.bar, &previous.tatum]
                .into_iter()
                .zip([&candidates.beat, &candidates.bar, &candidates.tatum])
                .enumerate()
            {
                for i in 0..CANDIDATES {
                    for j in 0..CANDIDATES {
                        // Missing candidates carry no temporal evidence.
                        transitions[level][i][j] = if old[i].period > 0. && new[j].period > 0. {
                            period_transition(old[i].period, new[j].period, 0.1)
                        } else {
                            0.
                        };
                    }
                }
            }
        }
        // Each pairwise likelihood appears in five joint states. Evaluate its
        // exponential/logarithm once while preserving observation summation order.
        let beat_logs = candidates.beat.map(|c| (c.strength + FLOOR).ln());
        let bar_logs = candidates.bar.map(|c| (c.strength + FLOOR).ln());
        let tatum_logs = candidates.tatum.map(|c| (c.strength + FLOOR).ln());
        let bar_beat: [[f64; CANDIDATES]; CANDIDATES] = std::array::from_fn(|r| {
            std::array::from_fn(|b| {
                (period_compatibility(candidates.bar[r].period, candidates.beat[b].period) + FLOOR)
                    .ln()
            })
        });
        let beat_tatum: [[f64; CANDIDATES]; CANDIDATES] = std::array::from_fn(|b| {
            std::array::from_fn(|t| {
                (period_compatibility(candidates.beat[b].period, candidates.tatum[t].period)
                    + FLOOR)
                    .ln()
            })
        });
        let mut best = 0;
        for (index, score) in next_scores.iter_mut().enumerate() {
            let [b, r, t] = indices(index);
            let observation = 3.0 * tatum_logs[t]
                + bar_logs[r]
                + beat_logs[b]
                + bar_beat[r][b]
                + beat_tatum[b][t];
            let mut accumulated = 0.;
            if self.previous.is_some() {
                accumulated = f64::NEG_INFINITY;
                for (i, &previous_score) in self.scores.iter().enumerate() {
                    let [pb, pr, pt] = indices(i);
                    let value = previous_score
                        + transitions[0][pb][b]
                        + transitions[1][pr][r]
                        + transitions[2][pt][t];
                    if value > accumulated {
                        accumulated = value;
                    }
                }
            }
            *score = accumulated + observation;
        }
        for i in 1..STATES {
            if next_scores[i] > next_scores[best] {
                best = i;
            }
        }
        self.scores = next_scores;
        self.previous = Some(candidates);
        let [b, r, t] = indices(best);
        let valid = |c: Candidate| if c.period == 0. { 7. } else { c.period };
        PeriodEstimate {
            beat: valid(candidates.beat[b]),
            bar: valid(candidates.bar[r]),
            tatum: valid(candidates.tatum[t]),
        }
    }
}
// Empty evidence uses zero-period candidates. They contribute the observation
// floor without passing zero into period-ratio calculations.
fn period_compatibility(slower: f64, faster: f64) -> f64 {
    if slower > 0. && faster > 0. {
        conditional_period(slower, faster, 0.3, &WEIGHTS)
    } else {
        0.
    }
}

fn indices(index: usize) -> [usize; 3] {
    [
        index % CANDIDATES,
        index / CANDIDATES % CANDIDATES,
        index / (CANDIDATES * CANDIDATES),
    ]
}

const PHASE_CANDIDATES: usize = 15;

#[derive(Clone, Copy, Debug, PartialEq)]
/// Period estimates and phase anchors on the analysis timeline.
pub struct MeterEstimate {
    pub periods: PeriodEstimate,
    /// Beat phase anchor in seconds on the analysis timeline.
    pub beat_time: f64,
    /// Bar phase anchor in seconds on the analysis timeline.
    pub bar_time: f64,
    /// Tatum phase anchor in seconds, equal to the beat phase anchor.
    pub tatum_time: f64,
}

/// Tracks beat and bar phase using resonator evidence and temporal continuity.
pub struct PhaseDecoder {
    previous_times: [[f64; PHASE_CANDIDATES]; 2],
    scores: [[f64; PHASE_CANDIDATES]; 2],
    previous_lag: [usize; 2],
    initialized: bool,
    matrix: Vec<f64>,
    evidence: Vec<f64>,
}
impl Default for PhaseDecoder {
    fn default() -> Self {
        Self::new()
    }
}
impl PhaseDecoder {
    /// Allocate phase-scoring workspaces and initialize temporal state.
    pub fn new() -> Self {
        Self {
            previous_times: [[0.; PHASE_CANDIDATES]; 2],
            scores: [[0.; PHASE_CANDIDATES]; 2],
            previous_lag: [0; 2],
            initialized: false,
            matrix: vec![0.; LAGS * GROUPS],
            evidence: vec![0.; LAGS],
        }
    }
    /// Clear phase history and accumulated transition scores.
    pub fn reset(&mut self) {
        self.previous_times = [[0.; PHASE_CANDIDATES]; 2];
        self.scores = [[0.; PHASE_CANDIDATES]; 2];
        self.previous_lag = [0; 2];
        self.initialized = false;
    }
    /// Select beat and bar phase anchors from the resonator state and prior phases.
    /// `elapsed` is the number of accent samples consumed by the resonator bank.
    /// The tatum phase anchor is set to the selected beat phase.
    pub fn process(
        &mut self,
        bank: &PeriodEvidence,
        periods: PeriodEstimate,
        elapsed: usize,
    ) -> MeterEstimate {
        let mut selected = [0.; 2];
        for (level, period) in [periods.beat, periods.bar].into_iter().enumerate() {
            let lag = ((period * ACCENT_RATE).round() as usize).clamp(1, LAGS);
            for phase in 0..lag {
                for group in 0..GROUPS {
                    self.matrix[phase * GROUPS + group] = bank.phase_state(group, lag, phase);
                }
            }
            if level == 0 {
                for phase in 0..lag {
                    self.evidence[phase] = 0.;
                    for group in 0..GROUPS {
                        self.evidence[phase] +=
                            (GROUPS + 1 - group) as f64 * self.matrix[phase * GROUPS + group];
                    }
                }
            } else {
                pattern_phase(
                    &self.matrix[..lag * GROUPS],
                    GROUPS,
                    &mut self.evidence[..lag],
                );
            }
            let mut peaks = [0; PHASE_CANDIDATES];
            peak_pick(&self.evidence[..lag], &mut peaks);
            if peaks[0] >= lag {
                // All candidates use the midpoint when phase evidence is flat.
                // Populate every slot so prior peaks cannot leak into this update.
                peaks.fill(lag / 2);
            }
            let mut last = 0;
            for p in &mut peaks {
                if *p >= lag {
                    *p = last;
                }
                last = *p;
            }
            let mut times = [0.; PHASE_CANDIDATES];
            let mut observations = [0.; PHASE_CANDIDATES];
            for i in 0..PHASE_CANDIDATES {
                observations[i] = (self.evidence[peaks[i]].max(0.) + FLOOR).ln();
                times[i] = (peaks[i] as f64 + (elapsed as f64 - lag as f64) + 1.) / ACCENT_RATE
                    + 0.5 / ACCENT_RATE;
            }
            let mut scores = observations;
            if self.initialized {
                for j in 0..PHASE_CANDIDATES {
                    let mut best = f64::NEG_INFINITY;
                    for i in 0..PHASE_CANDIDATES {
                        let value = self.scores[level][i]
                            + phase_transition(
                                self.previous_times[level][i],
                                times[j],
                                self.previous_lag[level] as f64 / ACCENT_RATE,
                                0.1,
                            );
                        if value > best {
                            best = value;
                        }
                    }
                    scores[j] += best;
                }
            }
            let mut best = 0;
            for i in 1..PHASE_CANDIDATES {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
            selected[level] = times[best];
            self.scores[level] = scores;
            self.previous_times[level] = times;
            self.previous_lag[level] = lag;
        }
        self.initialized = true;
        MeterEstimate {
            periods,
            beat_time: selected[0],
            bar_time: selected[1],
            tatum_time: selected[0],
        }
    }
}
