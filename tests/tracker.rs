use pulseweave::{stream::MeterStream, tracker::Tracker};

#[test]
fn native_and_adapted_audio_share_the_same_timeline() {
    let audio: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.13).sin()).collect();
    let mut native = Tracker::new();
    let mut adapted = MeterStream::new(44_100).unwrap();
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    native.process(
        &audio.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        |u| expected.push(u),
    );
    adapted.process(&audio, |u| actual.push(u));
    assert!(!actual.is_empty());
    assert_eq!(actual, expected);
    assert!(
        actual
            .windows(2)
            .all(|pair| pair[0].audio_samples < pair[1].audio_samples)
    );
}
