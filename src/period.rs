//! Comb-resonator periodicity evidence and candidate extraction.
use crate::dsp::{Resonator, log_normal, peak_pick, process_bank};
use crate::onset::ACCENT_RATE;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub const GROUPS: usize = 4;
pub const LAGS: usize = 689;
pub const CANDIDATES: usize = 5;
const FFT_SIZE: usize = 1024;
const TATUM_BINS: usize = 120;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
/// A period in seconds and its evidence strength.
pub struct Candidate {
    pub period: f64,
    pub strength: f64,
}

/// Ranked period candidates for beat, bar, and tatum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodCandidates {
    pub beat: [Candidate; CANDIDATES],
    pub bar: [Candidate; CANDIDATES],
    pub tatum: [Candidate; CANDIDATES],
    pub beatyness: f64,
}

/// Stateful resonator banks that score periodicity across four spectral groups.
pub struct PeriodEvidence {
    banks: [Vec<Resonator>; GROUPS],
    feedback: Vec<f64>,
    simd: fearless_simd::Level,
    decay: f64,
    window: [f64; LAGS],
    beat_prior: [f64; LAGS],
    bar_prior: [f64; LAGS],
    tatum_prior: [f64; TATUM_BINS],
    energy: [f64; GROUPS],
    elapsed: usize,
    bias: Vec<f64>,
    combined: Vec<f64>,
    beat: Vec<f64>,
    bar: Vec<f64>,
    tatum: [f64; TATUM_BINS],
    fft: Arc<dyn Fft<f64>>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}
impl Default for PeriodEvidence {
    fn default() -> Self {
        Self::new()
    }
}
impl PeriodEvidence {
    /// Allocate four resonator banks and FFT workspaces for period scoring.
    pub fn new() -> Self {
        let feedback: Vec<_> = (1..=LAGS)
            .map(|lag| 0.5_f64.powf(lag as f64 / (3.0 * ACCENT_RATE)))
            .collect();
        let banks = std::array::from_fn(|_| {
            feedback
                .iter()
                .enumerate()
                .map(|(i, &a)| Resonator::new(i + 1, a))
                .collect()
        });
        let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
        let scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
        Self {
            banks,
            feedback,
            simd: fearless_simd::Level::new(),
            decay: 0.5_f64.powf(1.0 / (3.0 * ACCENT_RATE)),
            window: std::array::from_fn(|i| {
                (((i + 1) as f64 * std::f64::consts::PI) / (LAGS + 1) as f64).cos() * 0.5 + 0.5
            }),
            beat_prior: std::array::from_fn(|i| {
                log_normal((i + 1) as f64 / ACCENT_RATE, 0.55, 0.28).powf(0.31)
            }),
            bar_prior: std::array::from_fn(|i| {
                log_normal((i + 1) as f64 / ACCENT_RATE, 2.1, 0.26).powf(0.29)
            }),
            tatum_prior: std::array::from_fn(|i| {
                let period = FFT_SIZE as f64 / (i.max(1) as f64 * ACCENT_RATE);
                log_normal(period, 0.18, 0.39).powf(0.16) * period.sqrt()
            }),
            energy: [0.; GROUPS],
            elapsed: 0,
            bias: vec![0.; LAGS],
            combined: vec![0.; LAGS],
            beat: vec![0.; LAGS],
            bar: vec![0.; LAGS],
            tatum: [0.; TATUM_BINS],
            fft,
            spectrum: vec![Complex::default(); FFT_SIZE],
            scratch,
        }
    }
    /// Clear resonator histories, energy estimates, and cached strengths.
    pub fn reset(&mut self) {
        for bank in &mut self.banks {
            for r in bank {
                r.reset();
            }
        }
        self.energy.fill(0.);
        self.elapsed = 0;
        self.bias.fill(0.);
        self.combined.fill(0.);
        self.beat.fill(0.);
        self.bar.fill(0.);
        self.tatum.fill(0.);
    }
    /// Process an update of the four detrended spectral groups. All slices must
    /// have equal nonzero length. This stage expects mean removal
    /// and group-delay adjustment to have already been applied.
    pub fn process(&mut self, groups: [&[f64]; GROUPS]) -> PeriodCandidates {
        let count = groups[0].len();
        assert!(count > 0 && groups.iter().all(|x| x.len() == count));
        self.elapsed += count;
        let decay = self.decay;
        self.combined.fill(0.);
        for lag in 1..=LAGS {
            let a = self.feedback[lag - 1];
            let startup = a.powf((self.elapsed as f64 / lag as f64).ceil());
            self.bias[lag - 1] =
                (1.0 - startup * startup) * ((1.0 - a) * (1.0 - a) / (1.0 - a * a));
        }
        for (group, samples) in groups.iter().enumerate() {
            for &sample in *samples {
                self.energy[group] = self.energy[group] * decay + sample * sample;
            }
            process_bank(&mut self.banks[group], samples, self.simd);
            for lag in 1..=LAGS {
                let resonator = &self.banks[group][lag - 1];
                let energy = resonator.state_energy();
                let a = self.feedback[lag - 1];
                let value = if self.energy[group] > 0. {
                    ((energy / (a * a)) / lag as f64) / ((1.0 - decay) * self.energy[group])
                        - self.bias[lag - 1]
                } else {
                    0.
                };
                self.combined[lag - 1] += value.max(0.);
            }
        }
        for value in &mut self.combined {
            *value /= GROUPS as f64;
        }
        let mean = self.combined.iter().sum::<f64>() / LAGS as f64;
        self.spectrum.fill(Complex::default());
        for i in 0..LAGS {
            self.spectrum[i].re = (self.combined[i] - mean) * self.window[i];
        }
        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);
        for i in 0..TATUM_BINS {
            self.tatum[i] = self.tatum_prior[i] * (self.spectrum[i].norm() / LAGS as f64);
        }
        for i in 0..LAGS {
            let evidence = self.combined[i] / (1.0 - self.bias[i]);
            self.beat[i] = self.beat_prior[i] * evidence;
            self.bar[i] = self.bar_prior[i] * evidence;
        }
        let mut beatyness = 0.;
        for &value in &self.beat {
            beatyness += value / LAGS as f64;
        }
        PeriodCandidates {
            beat: candidates(&self.beat, |i| (i + 1) as f64 / ACCENT_RATE),
            bar: candidates(&self.bar, |i| (i + 1) as f64 / ACCENT_RATE),
            tatum: candidates(&self.tatum, |i| {
                FFT_SIZE as f64 / (i.max(1) as f64 * ACCENT_RATE)
            }),
            beatyness,
        }
    }
    /// Return cached beat, bar, and tatum evidence from the latest update, in that order.
    pub fn strengths(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.beat, &self.bar, &self.tatum)
    }
    /// Read a resonator phase value. Phase wraps modulo lag.
    /// Panics unless group < GROUPS and 1 <= lag <= LAGS.
    pub fn phase_state(&self, group: usize, lag: usize, phase: usize) -> f64 {
        self.banks[group][lag - 1].phase_state(phase)
    }
}
fn candidates(values: &[f64], period: impl Fn(usize) -> f64) -> [Candidate; CANDIDATES] {
    let mut peaks = [0; CANDIDATES];
    peak_pick(values, &mut peaks);
    let mut last = Candidate::default();
    std::array::from_fn(|i| {
        if peaks[i] < values.len() {
            last = Candidate {
                period: period(peaks[i]),
                strength: values[peaks[i]],
            };
        }
        last
    })
}
