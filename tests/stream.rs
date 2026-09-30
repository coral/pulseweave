use pulseweave::stream::{FeatureStream, audio_channel};

#[test]
fn arbitrary_callback_sizes_are_equivalent() {
    for rate in [8000, 44100, 48000, 96000, 192000] {
        let input: Vec<f32> = (0..rate).map(|i| (i as f32 * 0.13).sin()).collect();
        let run = |chunk| {
            let mut stream = FeatureStream::new(rate).unwrap();
            let mut out = Vec::new();
            for samples in input.chunks(chunk) {
                stream.process(samples, |f| out.push(f.bands));
            }
            out
        };
        assert_eq!(run(1), run(997));
    }
}

#[test]
fn queue_overflow_resets_generation() {
    let (mut input, mut worker) = audio_channel(44100, 1024).unwrap();
    assert_eq!(input.push(&[0.2; 2048]), 1024);
    let mut generations = Vec::new();
    assert_eq!(worker.poll(4096, |g, _| generations.push(g)), 1024);
    assert_eq!(input.push(&[0.4; 1024]), 0);
    worker.poll(4096, |g, _| generations.push(g));
    assert_eq!(generations, vec![0, 1]);
}

#[test]
fn silence_and_nonfinite_input_remain_finite() {
    let mut stream = FeatureStream::new(48000).unwrap();
    let input = [f32::NAN, f32::INFINITY, 0., f32::NEG_INFINITY].repeat(4000);
    stream.process(&input, |f| {
        assert!(f.bands.iter().all(|v| v.is_finite() && *v == 0.))
    });
}

#[test]
fn meter_partition_reset_and_silence() {
    use pulseweave::stream::MeterStream;
    for rate in [8000, 44100, 48000, 96000, 192000] {
        let input: Vec<f32> = (0..rate * 2).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut stream = MeterStream::new(rate).unwrap();
        let mut run = |chunk| {
            stream.reset();
            let mut output = Vec::new();
            for block in input.chunks(chunk) {
                stream.process(block, |u| {
                    let m = u.meter;
                    output.push((
                        u.audio_samples,
                        [
                            m.periods.tatum,
                            m.periods.beat,
                            m.periods.bar,
                            m.tatum_time,
                            m.beat_time,
                            m.bar_time,
                            u.beatyness,
                        ],
                    ));
                });
            }
            output
        };
        let first = run(997);
        assert!(!first.is_empty());
        assert_eq!(first, run(1));
        stream.reset();
        let mut updates = 0;
        for _ in 0..rate * 2 / 512 {
            stream.process(&[f32::NAN; 512], |u| {
                let m = u.meter;
                assert!(
                    [
                        m.periods.tatum,
                        m.periods.beat,
                        m.periods.bar,
                        m.tatum_time,
                        m.beat_time,
                        m.bar_time,
                        u.beatyness
                    ]
                    .iter()
                    .all(|v| v.is_finite())
                );
                updates += 1;
            });
        }
        assert!(updates > 0);
    }
}

#[test]
fn invalid_configuration_returns_typed_errors() {
    use pulseweave::stream::{MeterStream, StreamError, meter_channel};
    for rate in [0, 7999, 192001, usize::MAX] {
        assert!(
            matches!(MeterStream::new(rate), Err(StreamError::UnsupportedSampleRate(actual)) if actual == rate)
        );
    }
    assert!(matches!(
        meter_channel(44100, 0),
        Err(StreamError::ZeroCapacity)
    ));
}

#[test]
fn meter_gap_discards_partial_audio_and_restarts_timeline() {
    use pulseweave::stream::{MeterStream, meter_channel};
    let (mut input, mut worker) = meter_channel(44100, 1024).unwrap();
    assert_eq!(
        worker.poll(0, |_, _| panic!("zero limit emitted output")),
        0
    );
    // Leave a partial FFT window in the stream, then drop incoming samples.
    assert_eq!(input.push(&[0.2; 1536]), 512);
    assert_eq!(
        worker.poll(1024, |_, _| panic!("too little audio for meter")),
        1024
    );
    let audio: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut actual = Vec::new();
    for block in audio.chunks(997) {
        assert_eq!(input.push(block), 0);
        assert_eq!(
            worker.poll(4096, |generation, update| {
                assert_eq!(generation, 1);
                actual.push(update);
            }),
            block.len()
        );
    }
    let mut fresh = MeterStream::new(44100).unwrap();
    let mut expected = Vec::new();
    fresh.process(&audio, |update| expected.push(update));
    assert!(!actual.is_empty());
    assert_eq!(actual, expected);
}
