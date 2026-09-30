//! Probability, filtering, peak selection, and rhythmic phase scoring.
use fearless_simd::{Level, dispatch, prelude::*};
use fearless_simd_macros::simd;

/// Log-normal probability density for a period prior.
/// `median`, `sigma`, and `x` must be finite and positive.
pub fn log_normal(x: f64, median: f64, sigma: f64) -> f64 {
    let z = (x / median).ln();
    (z * z / (sigma * sigma * -2.0)).exp() / (x * sigma * 2.5066282746310002)
}

/// Gaussian log transition score between period candidates.
/// Uses the base-10 logarithm of the period ratio. Periods and sigma must be positive.
pub fn period_transition(previous: f64, next: f64, sigma: f64) -> f64 {
    let z = (next / previous).log10() / sigma;
    z * z * -0.5
}

/// Weighted Gaussian mixture over nine integer multiples of a faster pulse.
/// Periods and sigma must be positive; weights must be nonnegative with a positive sum.
pub fn conditional_period(period: f64, faster_period: f64, sigma: f64, weights: &[f64; 9]) -> f64 {
    let total: f64 = weights.iter().sum();
    let mut result = 0.0;
    for (i, weight) in weights.iter().enumerate() {
        let z = (period - faster_period * (i + 1) as f64) / (sigma * faster_period);
        result += (weight / total / (sigma * 2.5066282746310002)) * (z * z * -0.5).exp();
    }
    result
}

/// Convert a nonnegative frequency in Hz to the ERB-rate scale.
pub fn hz_to_erb(hz: f64) -> f64 {
    ((hz * 4.368) / 1000.0 + 1.0).log10() * 21.365541873789756
}

/// Convert an ERB-rate value to frequency in Hz.
pub fn erb_to_hz(erb: f64) -> f64 {
    ((10.0_f64.powf(erb / 21.365541873789756) - 1.0) * 1000.0) / 4.368
}

/// Numerator coefficients for a sixth-order 10 Hz Butterworth low-pass filter.
/// The envelope sample rate is 44100 / 256 Hz.
pub const BUTTER_B: [f64; 7] = [
    1.934821074187553e-5,
    0.0001160892644512532,
    0.000290223161128133,
    0.0003869642148375107,
    0.000290223161128133,
    0.0001160892644512532,
    1.934821074187553e-5,
];
/// Denominator coefficients for the envelope low-pass filter, with `a[0] = 1`.
pub const BUTTER_A: [f64; 7] = [
    1.0,
    -4.592236133523747,
    8.918281529847143,
    -9.353758411594637,
    5.578960297450523,
    -1.791904459442157,
    0.2418954627503554,
];

/// Fixed-size, allocation-free direct-form IIR with persistent history.
#[derive(Clone, Debug)]
pub struct EnvelopeFilter {
    x: [f64; 7],
    y: [f64; 7],
    head: usize,
}
impl Default for EnvelopeFilter {
    fn default() -> Self {
        Self {
            x: [0.; 7],
            y: [0.; 7],
            head: 0,
        }
    }
}
impl EnvelopeFilter {
    /// Clear the input and output histories.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    /// Filter one envelope sample and advance the filter history.
    pub fn process(&mut self, sample: f64) -> f64 {
        self.x[self.head] = sample;
        let mut y = 0.0;
        for (i, b) in BUTTER_B.iter().enumerate() {
            y += b * self.x[(self.head + 7 - i) % 7];
        }
        for (i, a) in BUTTER_A.iter().enumerate().skip(1) {
            y -= a * self.y[(self.head + 7 - i) % 7];
        }
        self.y[self.head] = y;
        self.head = (self.head + 1) % 7;
        y
    }
}

/// Exponentially decaying comb resonator with a fixed delay in samples.
#[derive(Clone, Debug)]
pub struct Resonator {
    delay: Vec<f64>,
    position: usize,
    feedback: f64,
}
impl Resonator {
    /// Allocate a delay line of `period` samples. Panics unless period > 0 and 0 < feedback < 1.
    pub fn new(period: usize, feedback: f64) -> Self {
        assert!(period > 0 && feedback > 0.0 && feedback < 1.0);
        Self {
            delay: vec![0.; period],
            position: 0,
            feedback,
        }
    }
    /// Clear delayed feedback and restart the phase position.
    pub fn reset(&mut self) {
        self.delay.fill(0.);
        self.position = 0;
    }
    /// Process one input sample and return the decaying comb output.
    pub fn process(&mut self, sample: f64) -> f64 {
        let y = sample * (1.0 - self.feedback) + self.delay[self.position];
        self.delay[self.position] = y * self.feedback;
        self.position += 1;
        if self.position == self.delay.len() {
            self.position = 0;
        }
        y
    }
    /// Advance through a block without retaining the individual output samples.
    /// Uses SIMD for independent positions within each delay-line segment.
    pub fn process_block(&mut self, samples: &[f64]) {
        dispatch!(Level::new(), simd => process_bank_simd(simd, std::slice::from_mut(self), samples));
    }

    /// Sum squared delayed feedback in phase order, without allocating.
    pub fn state_energy(&self) -> f64 {
        let (before, after) = self.delay.split_at(self.position);
        after.iter().chain(before).fold(0., |sum, &x| sum + x * x)
    }

    /// Read delayed feedback relative to the next input sample, wrapping by the period.
    pub fn phase_state(&self, phase: usize) -> f64 {
        self.delay[(self.position + phase % self.delay.len()) % self.delay.len()]
    }
}

pub(crate) fn process_bank(bank: &mut [Resonator], samples: &[f64], level: Level) {
    dispatch!(level, simd => process_bank_simd(simd, bank, samples));
}

// Native lane count depends on the generic backend and cannot be a stable Rust
// const-generic argument to as_chunks.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[simd]
fn process_bank_simd<S: Simd>(simd: S, bank: &mut [Resonator], samples: &[f64]) {
    for resonator in bank {
        let feedback = resonator.feedback;
        let input_gain = 1. - feedback;
        let mut remaining = samples;
        while !remaining.is_empty() {
            // A segment never crosses a wrap, so every lane reads independent
            // state. Even periods shorter than the SIMD width remain causal.
            let count = remaining
                .len()
                .min(resonator.delay.len() - resonator.position);
            let (input, rest) = remaining.split_at(count);
            let state = &mut resonator.delay[resonator.position..resonator.position + count];
            let mut inputs = input.chunks_exact(S::f64s::LEN);
            let mut states = state.chunks_exact_mut(S::f64s::LEN);
            for (input, state) in inputs.by_ref().zip(states.by_ref()) {
                let x = S::f64s::from_slice(simd, input);
                let delayed = S::f64s::from_slice(simd, state);
                // Keep multiply and add separate to preserve scalar rounding.
                ((x * input_gain + delayed) * feedback).store_slice(state);
            }
            for (&x, delayed) in inputs.remainder().iter().zip(states.into_remainder()) {
                *delayed = (x * input_gain + *delayed) * feedback;
            }
            resonator.position += count;
            if resonator.position == resonator.delay.len() {
                resonator.position = 0;
            }
            remaining = rest;
        }
    }
}

/// Gaussian density of phase distance wrapped to half a period.
/// Times and period use the same units; period and sigma must be positive.
/// Sigma is measured in cycles.
pub fn phase_transition(previous: f64, next: f64, period: f64, sigma: f64) -> f64 {
    let distance = ((next - previous).abs() + period * 0.5) % period - period * 0.5;
    let z = (distance / period) / sigma;
    (z * z * -0.5).exp() / (sigma * 2.5066282746310002)
}

/// Write indices of strict local maxima, highest first. Endpoints compare only
/// their one neighbor. Unfilled output slots receive `values.len()`.
/// Equal heights keep their input order.
/// Returns the number of peaks written. No allocation is performed.
pub fn peak_pick(values: &[f64], output: &mut [usize]) -> usize {
    output.fill(values.len());
    let mut found = 0;
    for (i, &value) in values.iter().enumerate() {
        if (i > 0 && value <= values[i - 1]) || (i + 1 < values.len() && value <= values[i + 1]) {
            continue;
        }
        let position = output
            .iter()
            .position(|&index| index == values.len() || value > values[index]);
        if let Some(position) = position {
            output.copy_within(position..output.len() - 1, position + 1);
            output[position] = i;
            found = (found + 1).min(output.len());
        }
    }
    found
}

/// Score bar phase using the maximum of two weighted rhythmic templates.
/// Input is row-major, one row per
/// phase and one column per spectral group. `scores.len()` determines the phase
/// count. Returns the first maximum, or `None` for an empty phase grid.
/// Panics unless groups is positive and input.len() equals groups * scores.len().
pub fn pattern_phase(input: &[f64], groups: usize, scores: &mut [f64]) -> Option<usize> {
    assert!(
        groups > 0 && input.len() / groups == scores.len() && input.len().is_multiple_of(groups)
    );
    let phases = scores.len();
    if phases == 0 {
        return None;
    }
    let quarter = (phases as f64 * 0.25).round() as usize;
    let half = (phases as f64 * 0.5).round() as usize;
    let third_quarter = (phases as f64 * 0.75).round() as usize;
    let sum_weights: f64 = (1..=groups).map(|g| g as f64).sum();
    let weighted = |phase: usize| -> f64 {
        input[phase * groups..(phase + 1) * groups]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i + 1) as f64 * v)
            .sum()
    };
    let mut best = 0;
    for phase in 0..phases {
        let q1 = (phase + quarter) % phases;
        let q2 = (phase + half) % phases;
        let q3 = (phase + third_quarter) % phases;
        let first = (weighted(q1)
            + weighted(q3)
            + (input[q3 * groups] * (0.4738 * sum_weights)
                + input[phase * groups] * (1.1665 * sum_weights)))
            * 2.1;
        let second = weighted(q2) * 3.0314
            + weighted(q3) * 0.8123
            + (input[q3 * groups] * (0.19 * sum_weights)
                + input[phase * groups] * (sum_weights * 2.1496));
        scores[phase] = first.max(second);
        if scores[phase] > scores[best] {
            best = phase;
        }
    }
    Some(best)
}
