//! What can go wrong reading or writing an asset: [`Error`], and the [`ErrorKind`] it falls
//! under.

use std::borrow::Cow;
use std::error::Error as StdError;
use std::fmt;
use std::path::{Path, PathBuf};

/// What can go wrong reading or writing an asset.
///
/// Match on [`kind`](Self::kind) to tell failures apart; the message is for people and may
/// change. [`path`](Self::path) names the file being read, when the error came from one, and
/// [`format`](Self::format) the reader that failed. An error caused by another (an I/O error, a
/// zip archive that doesn't open, JSON that doesn't parse) returns it from
/// [`source`](std::error::Error::source).
///
/// The message is lowercase without a trailing period, prefixed by the path when there is one.
/// It ends with the source's own message, so printing the error alone says why it failed.
pub struct Error(Box<Inner>);

struct Inner {
    kind: ErrorKind,
    message: Cow<'static, str>,
    path: Option<PathBuf>,
    format: Option<&'static str>,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

/// The kind of an [`Error`]: what a caller can do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Neither the extension nor the bytes name a format any reader reads.
    UnknownFormat,
    /// A reader for the format exists but this build was compiled without its Cargo feature,
    /// which the message names.
    FeatureDisabled,
    /// A valid file that uses a version, variant or extension the reader doesn't read.
    Unsupported,
    /// The data is malformed or inconsistent: it doesn't parse, or it contradicts itself.
    Corrupt,
    /// The data ends early, or an offset or count points past its end.
    Truncated,
    /// The file is empty, or well formed but holds nothing to convert: no geometry, points,
    /// skeleton or motion.
    Empty,
    /// A file the model needs (a glTF buffer, a USD layer, a skeleton) isn't among the assets.
    /// Readers that can do without a file (most textures) skip it instead.
    MissingAsset,
    /// The model is over a safety cap: too many triangles, nodes, pixels, nesting levels or
    /// bytes to allocate.
    LimitExceeded,
    /// Reading a file failed.
    Io,
    /// The caller asked for something that can't be done: a bad argument or option.
    Invalid,
}

impl ErrorKind {
    fn describe(self) -> &'static str {
        match self {
            Self::UnknownFormat => "unknown format",
            Self::FeatureDisabled => "feature disabled",
            Self::Unsupported => "unsupported",
            Self::Corrupt => "corrupt data",
            Self::Truncated => "truncated data",
            Self::Empty => "nothing to convert",
            Self::MissingAsset => "missing asset",
            Self::LimitExceeded => "limit exceeded",
            Self::Io => "i/o error",
            Self::Invalid => "invalid argument",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.describe())
    }
}

impl Error {
    /// The kind of error this is.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// The file being read when this went wrong (the model, or a file it refers to), when the
    /// error came from a file named by path.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.0.path.as_deref()
    }

    /// The short name of the reader that failed, which is also the name of its Cargo feature
    /// (`"obj"`, `"gltf"`, `"usd"`), when the error came from loading a model.
    #[must_use]
    pub fn format(&self) -> Option<&'static str> {
        self.0.format
    }

    /// An error of `kind` that says `message`, for an [`Assets`](crate::assets::Assets)
    /// source or a [`pack_glb`](crate::gltf::pack_glb) fetch of your own to return.
    ///
    /// ```
    /// use polysmelt::{Error, ErrorKind};
    ///
    /// let err = Error::new(ErrorKind::MissingAsset, "tex/wood.png isn't in the pack");
    /// assert_eq!(err.kind(), ErrorKind::MissingAsset);
    /// assert_eq!(err.to_string(), "tex/wood.png isn't in the pack");
    /// ```
    #[cold]
    #[inline(never)]
    pub fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        let message = match message.into() {
            Cow::Owned(s) => Cow::Owned(printable(s)),
            m @ Cow::Borrowed(_) => m,
        };
        Self(Box::new(Inner {
            kind,
            message,
            path: None,
            format: None,
            source: None,
        }))
    }

    /// An error of `kind` with a message from [`format_args!`], which a constant message
    /// builds without allocating it.
    #[cold]
    #[inline(never)]
    pub(crate) fn from_args(kind: ErrorKind, args: fmt::Arguments<'_>) -> Self {
        match args.as_str() {
            Some(s) => Self::new(kind, s),
            None => Self::new(kind, fmt::format(args)),
        }
    }

    /// The data is malformed or inconsistent.
    #[allow(dead_code, reason = "which readers use it depends on the enabled features")]
    pub(crate) fn corrupt(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Corrupt, message)
    }

    /// The data ends early.
    #[allow(dead_code, reason = "which readers use it depends on the enabled features")]
    pub(crate) fn truncated(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Truncated, message)
    }

    /// A version or variant the reader doesn't read.
    #[allow(dead_code, reason = "which readers use it depends on the enabled features")]
    pub(crate) fn unsupported(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    /// A file the model needs isn't there.
    #[allow(dead_code, reason = "which readers use it depends on the enabled features")]
    pub(crate) fn missing(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::MissingAsset, message)
    }

    /// An error of `kind` that says `message`, caused by `source`, which
    /// [`source`](std::error::Error::source) returns.
    #[cold]
    #[inline(never)]
    pub fn caused(
        kind: ErrorKind,
        message: impl Into<Cow<'static, str>>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        let mut e = Self::new(kind, message);
        e.0.source = Some(Box::new(source));
        e
    }

    /// The error, saying it happened reading the file at `path` unless it names a file already.
    #[cold]
    #[must_use]
    pub(crate) fn at(mut self, path: &Path) -> Self {
        if self.0.path.is_none() {
            self.0.path = Some(path.to_path_buf());
        }
        self
    }

    /// The error, saying the reader named `format` failed unless it names one already.
    #[cold]
    #[must_use]
    #[allow(dead_code, reason = "only loading a model names its reader")]
    pub(crate) fn in_format(mut self, format: &'static str) -> Self {
        if self.0.format.is_none() {
            self.0.format = Some(format);
        }
        self
    }

    /// An error reading a file the model refers to, at that file's path: a missing file is
    /// [`ErrorKind::MissingAsset`], and any other failure is what the io error says.
    #[cold]
    #[inline(never)]
    #[allow(dead_code, reason = "only reading from the file system uses it")]
    pub(crate) fn asset(source: std::io::Error, path: &Path, what: impl Into<Cow<'static, str>>) -> Self {
        let kind = match source.kind() {
            std::io::ErrorKind::NotFound => ErrorKind::MissingAsset,
            k => io_kind(k),
        };
        Self::caused(kind, what, source).at(path)
    }

    /// The error with `what` before its message, saying where in the model it happened. The
    /// kind, path and source stay.
    #[cold]
    #[inline(never)]
    #[must_use]
    pub(crate) fn within(mut self, what: impl fmt::Display) -> Self {
        let inner = &mut *self.0;
        inner.message = Cow::Owned(printable(if inner.message.is_empty() {
            what.to_string()
        } else {
            format!("{what}: {}", inner.message)
        }));
        self
    }

    /// The message alone, without the path or the source's message.
    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.0.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &*self.0;
        if let Some(path) = &inner.path {
            write!(f, "{}: ", path.display())?;
        }
        f.write_str(&inner.message)?;
        if let Some(source) = &inner.source {
            if !inner.message.is_empty() {
                f.write_str(": ")?;
            }
            write!(f, "{source}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &*self.0;
        let mut d = f.debug_struct("Error");
        d.field("kind", &inner.kind).field("message", &inner.message);
        if let Some(path) = &inner.path {
            d.field("path", path);
        }
        if let Some(format) = inner.format {
            d.field("format", &format);
        }
        if let Some(source) = &inner.source {
            d.field("source", source);
        }
        d.finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source.as_ref().map(|s| &**s as &(dyn StdError + 'static))
    }
}

/// An I/O error, of kind [`ErrorKind::Io`] with the I/O error as its source; or, when it says
/// the data ended early ([`std::io::ErrorKind::UnexpectedEof`]), [`ErrorKind::Truncated`], and
/// when it says the data is invalid ([`std::io::ErrorKind::InvalidData`], as a decompressor
/// reports a bad stream), [`ErrorKind::Corrupt`].
impl From<std::io::Error> for Error {
    #[cold]
    fn from(e: std::io::Error) -> Self {
        Self::caused(io_kind(e.kind()), "", e)
    }
}

/// The kind of error an I/O error of `kind` is.
fn io_kind(kind: std::io::ErrorKind) -> ErrorKind {
    match kind {
        std::io::ErrorKind::UnexpectedEof => ErrorKind::Truncated,
        std::io::ErrorKind::InvalidData => ErrorKind::Corrupt,
        _ => ErrorKind::Io,
    }
}

/// A result with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors from the crates and std conversions readers parse with, kept as the source of the
/// [`Error`] that [`Context::context`] makes of them.
pub(crate) trait Foreign: StdError + Send + Sync + 'static {}

impl Foreign for std::io::Error {}
impl Foreign for std::num::ParseIntError {}
impl Foreign for std::num::ParseFloatError {}
impl Foreign for std::num::TryFromIntError {}
impl Foreign for std::array::TryFromSliceError {}
impl Foreign for std::str::Utf8Error {}
impl Foreign for std::string::FromUtf8Error {}
impl Foreign for std::collections::TryReserveError {}
impl Foreign for serde_json::Error {}
#[cfg(any(feature = "3mf", feature = "dae", feature = "usd", feature = "amf", feature = "sog"))]
impl Foreign for zip::result::ZipError {}
#[cfg(any(feature = "3mf", feature = "dae", feature = "x3d", feature = "amf"))]
impl Foreign for quick_xml::Error {}
#[cfg(feature = "e57")]
impl Foreign for e57::Error {}
#[cfg(feature = "laz")]
impl Foreign for laz::LasZipError {}
#[cfg(feature = "draco")]
impl Foreign for draco_core::DracoError {}
#[cfg(feature = "usd")]
impl Foreign for openusd::Error {}

/// Turns a missing value or a foreign error into an [`Error`] of a kind, saying what was being
/// done. A foreign error stays as the source.
pub(crate) trait Context<T> {
    /// The value, or an error of `kind` that says `what`.
    fn context(self, kind: ErrorKind, what: impl Into<Cow<'static, str>>) -> Result<T>;
    /// [`context`](Self::context), with a message made only on failure.
    fn with_context<M: Into<Cow<'static, str>>>(self, kind: ErrorKind, what: impl FnOnce() -> M) -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    #[inline]
    fn context(self, kind: ErrorKind, what: impl Into<Cow<'static, str>>) -> Result<T> {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::new(kind, what)),
        }
    }
    #[inline]
    fn with_context<M: Into<Cow<'static, str>>>(self, kind: ErrorKind, what: impl FnOnce() -> M) -> Result<T> {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::new(kind, what())),
        }
    }
}

impl<T, E: Foreign> Context<T> for std::result::Result<T, E> {
    #[inline]
    fn context(self, kind: ErrorKind, what: impl Into<Cow<'static, str>>) -> Result<T> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(Error::caused(kind, what, e)),
        }
    }
    #[inline]
    fn with_context<M: Into<Cow<'static, str>>>(self, kind: ErrorKind, what: impl FnOnce() -> M) -> Result<T> {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(Error::caused(kind, what(), e)),
        }
    }
}

/// Says where in the model an [`Error`] happened, keeping its kind, path and source.
pub(crate) trait Within<T> {
    /// The value, or the error with `what` before its message.
    #[allow(dead_code, reason = "which readers use it depends on the enabled features")]
    fn within(self, what: impl fmt::Display) -> Result<T>;
    /// [`within`](Self::within), with a message made only on failure.
    fn with_within<D: fmt::Display>(self, what: impl FnOnce() -> D) -> Result<T>;
}

impl<T> Within<T> for Result<T> {
    #[inline]
    fn within(self, what: impl fmt::Display) -> Result<T> {
        self.map_err(|e| e.within(what))
    }
    #[inline]
    fn with_within<D: fmt::Display>(self, what: impl FnOnce() -> D) -> Result<T> {
        self.map_err(|e| e.within(what()))
    }
}

/// Text from a file as an error message shows it: quoted, escaped as Rust writes a string
/// literal (so control characters and NULs come out printable), and cut to 32 characters.
#[allow(dead_code, reason = "which readers use it depends on the enabled features")]
pub(crate) fn shown(text: &[u8]) -> String {
    const MOST: usize = 32;
    let text = String::from_utf8_lossy(text);
    let mut cut: String = text.chars().take(MOST).collect();
    if text.chars().nth(MOST).is_some() {
        cut.push('…');
    }
    format!("{cut:?}")
}

/// An error message with any control characters escaped, so file content that slipped into
/// it can't garble a terminal or a report.
pub(crate) fn printable(message: String) -> String {
    if !message.chars().any(char::is_control) {
        return message;
    }
    message
        .chars()
        .map(|c| {
            if c.is_control() {
                c.escape_debug().to_string()
            } else {
                c.to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        const fn send_sync_static<T: Send + Sync + 'static>() {}
        send_sync_static::<Error>();
        assert!(size_of::<Error>() == size_of::<usize>());
        assert!(size_of::<Result<()>>() == size_of::<usize>());
    };

    #[test]
    fn constant_messages_are_borrowed() {
        let e = Error::from_args(ErrorKind::Corrupt, format_args!("no triangles"));
        assert!(matches!(e.0.message, Cow::Borrowed("no triangles")));
        let n = 3;
        let e = Error::from_args(ErrorKind::Corrupt, format_args!("{n} triangles"));
        assert_eq!(e.message(), "3 triangles");
    }

    #[test]
    fn display_has_the_path_message_and_source() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = Error::caused(ErrorKind::Io, "reading the MTL library", io).at(Path::new("a/b.mtl"));
        assert_eq!(e.to_string(), "a/b.mtl: reading the MTL library: denied");
        assert_eq!(e.path(), Some(Path::new("a/b.mtl")));
        let e = Error::from(std::io::Error::other("gone"));
        assert_eq!(e.to_string(), "gone");
        assert_eq!(e.kind(), ErrorKind::Io);
        assert!(e.source().is_some_and(|s| s.downcast_ref::<std::io::Error>().is_some()));
    }

    #[test]
    fn io_end_of_data_is_truncated() {
        let eof = std::io::Error::from(std::io::ErrorKind::UnexpectedEof);
        assert_eq!(Error::from(eof).kind(), ErrorKind::Truncated);
        let bad = std::io::Error::from(std::io::ErrorKind::InvalidData);
        assert_eq!(Error::from(bad).kind(), ErrorKind::Corrupt);
    }

    #[test]
    fn within_keeps_kind_path_and_source() {
        let e = Error::caused(ErrorKind::Truncated, "chunk", std::io::Error::other("eof"))
            .at(Path::new("m.3ds"))
            .within("reading mesh 2");
        assert_eq!(e.kind(), ErrorKind::Truncated);
        assert_eq!(e.to_string(), "m.3ds: reading mesh 2: chunk: eof");
        assert!(e.source().is_some());
    }

    #[test]
    fn control_characters_are_escaped() {
        let e = Error::new(ErrorKind::Corrupt, String::from("bad \u{1b}[31m name"));
        assert_eq!(e.message(), "bad \\u{1b}[31m name");
    }
}
