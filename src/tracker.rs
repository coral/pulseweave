//! Bounded, synchronous 44.1 kHz audio-to-meter analysis.
use crate::decoder::{MeterEstimate, PeriodDecoder, PhaseDecoder};
use crate::onset::{AccentFrame, BANDS, GROUP_DELAY, OnsetAnalyzer};
use crate::period::{GROUPS, PeriodEvidence};
const UPDATE_HOP: usize = 86;
const SNAPSHOT: usize = UPDATE_HOP + GROUP_DELAY + 1;
const HISTORY: usize = 128;
const BANDS_PER_GROUP: usize = BANDS / GROUPS;
const _: () = assert!(BANDS.is_multiple_of(GROUPS) && HISTORY >= SNAPSHOT);

/// A meter update. Estimates refer to the analysis timeline;
/// `audio_samples` records when sufficient input arrived to produce the result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerUpdate {
    pub meter: MeterEstimate,
    pub beatyness: f64,
    pub audio_samples: u64,
}

/// Native 44.1 kHz, mono f64 analysis path. Construction allocates the complete
/// workspace; processing and reset use bounded storage and no allocation.
pub struct Tracker {
    onset: OnsetAnalyzer,
    evidence: PeriodEvidence,
    periods: PeriodDecoder,
    phase: PhaseDecoder,
    history: Vec<[f64; BANDS]>,
    means: [f64; GROUPS],
    groups: [[f64; SNAPSHOT]; GROUPS],
    consumed: usize,
    audio_samples: u64,
}
impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}
impl Tracker {
    /// Allocate the onset, resonator, and decoder workspaces.
    pub fn new() -> Self {
        Self {
            onset: OnsetAnalyzer::new(),
            evidence: PeriodEvidence::new(),
            periods: PeriodDecoder::new(),
            phase: PhaseDecoder::new(),
            history: vec![[0.; BANDS]; HISTORY],
            means: [0.; GROUPS],
            groups: [[0.; SNAPSHOT]; GROUPS],
            consumed: 0,
            audio_samples: 0,
        }
    }
    /// Clear analysis history and restart the audio timeline at zero.
    pub fn reset(&mut self) {
        self.onset.reset();
        self.evidence.reset();
        self.periods.reset();
        self.phase.reset();
        self.history.fill([0.; BANDS]);
        self.means.fill(0.);
        self.consumed = 0;
        self.audio_samples = 0;
    }
    /// Consume mono 44.1 kHz samples and emit meter updates roughly twice per second.
    /// Caller block boundaries do not affect estimates. Nonfinite samples become zero.
    pub fn process(&mut self, audio: &[f64], mut emit: impl FnMut(TrackerUpdate)) {
        for &sample in audio {
            self.audio_samples += 1;
            let history = &mut self.history;
            let ready = self.onset.push(sample, &mut |frame: AccentFrame| {
                history[frame.index as usize % HISTORY] = frame.bands
            });
            if !ready {
                continue;
            }
            let raw_frames = self.onset.raw_frames();
            if raw_frames < self.consumed + UPDATE_HOP + GROUP_DELAY {
                continue;
            }
            self.prepare_snapshot(raw_frames);
            let candidates = self
                .evidence
                .process(std::array::from_fn(|g| &self.groups[g][..UPDATE_HOP]));
            let periods = self.periods.process(candidates);
            self.consumed += UPDATE_HOP;
            let meter = self.phase.process(&self.evidence, periods, self.consumed);
            emit(TrackerUpdate {
                meter,
                beatyness: candidates.beatyness,
                audio_samples: self.audio_samples,
            });
        }
    }

    /// Group the delayed accents and predicted tail, then remove the running mean.
    fn prepare_snapshot(&mut self, raw_frames: usize) {
        let tail = self.onset.predicted_tail();
        let count = raw_frames - self.consumed;
        debug_assert!(count <= SNAPSHOT);
        for i in 0..count {
            let absolute = self.consumed + GROUP_DELAY + i;
            let bands = if absolute < raw_frames - 1 {
                &self.history[absolute % HISTORY]
            } else {
                &tail[absolute - (raw_frames - 1)]
            };
            for group in 0..GROUPS {
                self.groups[group][i] = bands
                    [group * BANDS_PER_GROUP..(group + 1) * BANDS_PER_GROUP]
                    .iter()
                    .sum();
            }
        }
        for group in 0..GROUPS {
            for i in 0..count {
                self.means[group] = (self.means[group] * (self.consumed + i) as f64
                    + self.groups[group][i])
                    / (self.consumed + i + 1) as f64;
            }
            for x in &mut self.groups[group][..UPDATE_HOP] {
                *x -= self.means[group];
            }
        }
    }
}
