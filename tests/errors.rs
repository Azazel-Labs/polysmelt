//! Errors through the public API: every `ErrorKind` is reachable, errors name the file they
//! were reading, and `source()` hands back the error underneath.

use std::error::Error as _;
use std::path::PathBuf;

use polysmelt::{Error, ErrorKind};

/// A corpus file's path.
#[allow(dead_code, reason = "which tests use it depends on the enabled features")]
fn corpus(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus").join(rel)
}

/// The error reading `bytes` from memory, as a file named `name`.
fn from_memory(bytes: &[u8], name: &str) -> Error {
    match polysmelt::read(bytes.to_vec(), name) {
        Ok(_) => panic!("{name} loaded, expected an error"),
        Err(err) => err,
    }
}

/// The error loading a corpus file through the path API.
#[cfg(feature = "fs")]
#[allow(dead_code, reason = "which tests use it depends on the enabled features")]
fn from_file(rel: &str) -> Error {
    match polysmelt::open(corpus(rel)) {
        Ok(_) => panic!("{rel} loaded, expected an error"),
        Err(err) => err,
    }
}

/// An error's message has no trailing period.
fn assert_phrase(err: &Error) {
    let text = err.to_string();
    assert!(!text.ends_with('.'), "{text:?} ends with a period");
}

#[test]
fn unknown_format() {
    let err = from_memory(b"\x01\x02 these bytes name no format", "junk");
    assert_eq!(err.kind(), ErrorKind::UnknownFormat, "{err}");
    assert_eq!(err.format(), None);
    assert_phrase(&err);
    let err = from_memory(b"anything", "a.nope");
    assert_eq!(err.kind(), ErrorKind::UnknownFormat, "{err}");
    assert!(err.to_string().contains(".nope"), "{err}");
}

#[cfg(not(feature = "usd"))]
#[test]
fn feature_disabled() {
    let err = from_memory(b"#usda 1.0\n", "a.usda");
    assert_eq!(err.kind(), ErrorKind::FeatureDisabled, "{err}");
    assert!(err.to_string().contains("`usd` feature"), "{err}");
}

#[cfg(all(feature = "fs", feature = "iqm"))]
#[test]
fn unsupported() {
    let err = from_file("iqm/bad_version/bad_version.iqm");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    assert_eq!(err.format(), Some("iqm"));
    assert_phrase(&err);
}

#[cfg(feature = "ply")]
#[test]
fn corrupt() {
    let err = from_memory(b"ply\nformat ascii 1.0\nelement vertex 1\n", "a.ply");
    assert_eq!(err.kind(), ErrorKind::Corrupt, "{err}");
    assert_eq!(err.format(), Some("ply"));
    assert_eq!(err.to_string(), "PLY header has no end_header");
}

#[cfg(feature = "gltf")]
#[test]
fn truncated() {
    let err = from_memory(b"glTF\x02\0\0\0", "a.glb");
    assert_eq!(err.kind(), ErrorKind::Truncated, "{err}");
    assert_eq!(err.format(), Some("gltf"));
    assert_eq!(err.to_string(), "GLB is cut short");
}

#[cfg(feature = "stl")]
#[test]
fn empty() {
    // A binary STL's 80-byte header and a triangle count of 0.
    let err = from_memory(&[0; 84], "a.stl");
    assert_eq!(err.kind(), ErrorKind::Empty, "{err}");
    assert_phrase(&err);
}

#[cfg(all(feature = "fs", feature = "spz"))]
#[test]
fn limit_exceeded() {
    let err = from_file("spz/count_too_large/big.spz");
    assert_eq!(err.kind(), ErrorKind::LimitExceeded, "{err}");
    assert!(err.to_string().contains("Keep::max_points"), "{err}");
}

#[cfg(feature = "gltf")]
#[test]
fn missing_asset_from_memory() {
    let gltf = br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"absent.bin","byteLength":4}],
        "bufferViews":[{"buffer":0,"byteLength":4}]}"#;
    let err = from_memory(gltf, "models/a.gltf");
    assert_eq!(err.kind(), ErrorKind::MissingAsset, "{err}");
    assert!(
        err.path().is_some_and(|p| p.ends_with("absent.bin")),
        "{:?}",
        err.path()
    );
    assert_eq!(err.format(), Some("gltf"));
}

#[cfg(all(feature = "fs", feature = "gltf"))]
#[test]
fn missing_asset_from_file_names_it() {
    let err = from_file("gltf/buffer_missing/cube.gltf");
    assert_eq!(err.kind(), ErrorKind::MissingAsset, "{err}");
    // The path is the missing buffer's, not the model's.
    let want = std::path::absolute(corpus("gltf/buffer_missing/absent.bin")).unwrap();
    assert_eq!(err.path(), Some(want.as_path()));
    let io = err
        .source()
        .and_then(|s| s.downcast_ref::<std::io::Error>())
        .expect("an io source");
    assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
}

#[cfg(all(feature = "fs", feature = "stl"))]
#[test]
fn io_error_names_the_missing_file() {
    let path = std::path::absolute(corpus("stl/no_such_file.stl")).unwrap();
    let Err(err) = polysmelt::open(&path) else {
        panic!("a missing file loaded");
    };
    assert_eq!(err.kind(), ErrorKind::Io, "{err}");
    assert_eq!(err.path(), Some(path.as_path()));
    let io = err
        .source()
        .and_then(|s| s.downcast_ref::<std::io::Error>())
        .expect("an io source");
    assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
    let text = err.to_string();
    assert!(text.starts_with(&format!("{}: ", path.display())), "{text}");
    assert_phrase(&err);
}

#[cfg(feature = "fs")]
#[test]
fn io_error_from_a_missing_file_of_unknown_format() {
    let path = std::path::absolute(corpus("no_such_file")).unwrap();
    let Err(err) = polysmelt::open(&path) else {
        panic!("a missing file loaded");
    };
    assert_eq!(err.kind(), ErrorKind::Io, "{err}");
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn invalid() {
    use polysmelt::gltf::{CompressOptions, compress_with};
    let options = CompressOptions::EXT.color_filter(true);
    let err = compress_with(b"{}", options).expect_err("the COLOR filter needs KHR");
    assert_eq!(err.kind(), ErrorKind::Invalid, "{err}");
    assert_eq!(err.path(), None);
    assert!(err.source().is_none());
}

#[cfg(all(feature = "fs", feature = "3mf"))]
#[test]
fn zip_error_is_the_source() {
    let err = from_file("3mf/not_a_zip/text.3mf");
    assert_eq!(err.kind(), ErrorKind::UnknownFormat, "{err}");
    assert_eq!(err.format(), Some("3mf"));
    let source = err.source().expect("a source");
    assert!(source.downcast_ref::<zip::result::ZipError>().is_some(), "{source:?}");
    // Display carries the source's message too, so one line says it all.
    assert!(err.to_string().ends_with(&source.to_string()), "{err}");
}

#[test]
fn error_is_small_and_thread_safe() {
    fn send_sync<T: Send + Sync + 'static>() {}
    send_sync::<Error>();
    assert_eq!(size_of::<Error>(), size_of::<usize>());
    assert_eq!(size_of::<polysmelt::Result<()>>(), size_of::<usize>());
}
