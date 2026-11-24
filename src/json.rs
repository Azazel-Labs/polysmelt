//! JSON text to the [`Value`] `serde_json::from_slice` gives, built faster: whitespace and
//! string bodies are skipped eight bytes at a time, arrays and objects gather their items on
//! shared stacks and are allocated once at their final size, and nothing goes through serde's
//! visitors.
//!
//! - Text this parser doesn't accept (malformed, or nested deeper than `serde_json` allows) goes
//!   to `serde_json`, so errors are `serde_json`'s own.
//! - Numbers follow `serde_json`'s default rules: an integer that fits is a `u64` (or an `i64`
//!   when negative), anything else an `f64` computed the way `serde_json` computes it.
//! - Some `serde_json` features change the values it makes (`arbitrary_precision`,
//!   `float_roundtrip`), and any crate in a build can turn them on. A check made once compares
//!   both parsers on numbers those features read differently; when they disagree, every parse
//!   goes to `serde_json`. Objects whose first key `serde_json` may treat as a number or raw JSON
//!   (`$serde_json::private::...`) go to `serde_json` too.

use std::sync::OnceLock;

use serde_json::{Map, Number, Value};

/// Parses `text` as `serde_json::from_slice::<Value>` does.
pub(crate) fn from_slice(text: &[u8]) -> serde_json::Result<Value> {
    if agrees_with_serde_json()
        && let Some(v) = parse(text)
    {
        return Ok(v);
    }
    serde_json::from_slice(text)
}

/// The deepest nesting of arrays and objects `serde_json` accepts.
const MAX_DEPTH: usize = 127;

/// `text` parsed, or `None` when it isn't JSON this parser accepts.
fn parse(text: &[u8]) -> Option<Value> {
    let mut p = Parser {
        text,
        at: 0,
        depth: 0,
        items: Vec::new(),
        entries: Vec::new(),
        scratch: Vec::new(),
    };
    p.skip_whitespace();
    let v = p.value()?;
    p.skip_whitespace();
    (p.at == text.len()).then_some(v)
}

/// Whether this parser makes the values the `serde_json` in this build makes: false when a
/// feature that changes them is on.
fn agrees_with_serde_json() -> bool {
    static AGREES: OnceLock<bool> = OnceLock::new();
    *AGREES.get_or_init(|| {
        // `1.10` keeps its digits with `arbitrary_precision`; the long numbers round differently
        // with `float_roundtrip`.
        let probe = br"[1.10, -0, 18446744073709551616, 154757138494460336e8, 104308740306854565e-4,
            156437686781252718e-17, 186777421129029881e-18, 1e-400, 123456789012345678901234567890e-10]";
        let (Some(ours), Ok(theirs)) = (parse(probe), serde_json::from_slice::<Value>(probe)) else {
            return false;
        };
        same(&ours, &theirs)
    })
}

/// Whether two values are the same down to each number's kind and each float's bits.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.as_u64() == y.as_u64()
                && x.as_i64() == y.as_i64()
                && x.is_f64() == y.is_f64()
                && x.as_f64().map(f64::to_bits) == y.as_f64().map(f64::to_bits)
                && x.to_string() == y.to_string()
        }
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|((kx, vx), (ky, vy))| kx == ky && same(vx, vy))
        }
        _ => a == b,
    }
}

/// Powers of ten from 1e0 to 1e308, each the nearest `f64`, as `serde_json`'s table holds them.
fn pow10() -> &'static [f64; 309] {
    static POW10: OnceLock<[f64; 309]> = OnceLock::new();
    POW10.get_or_init(|| std::array::from_fn(|i| format!("1e{i}").parse().expect("a power of ten parses")))
}

struct Parser<'t> {
    text: &'t [u8],
    at: usize,
    depth: usize,
    /// The items of the arrays being read, innermost last.
    items: Vec<Value>,
    /// The entries of the objects being read, innermost last.
    entries: Vec<(String, Value)>,
    /// A string with escapes, as it's decoded.
    scratch: Vec<u8>,
}

/// Eight copies of byte `b`.
const fn splat(b: u8) -> u64 {
    0x0101_0101_0101_0101 * b as u64
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    /// The eight bytes from `at`, when there are eight.
    fn word(&self) -> Option<u64> {
        let b = self.text.get(self.at..self.at + 8)?;
        Some(u64::from_le_bytes(b.try_into().expect("eight bytes")))
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.peek() {
            self.at += 1;
            // Indentation: runs of spaces, eight at a time.
            while let Some(w) = self.word() {
                let other = w ^ splat(b' ');
                if other != 0 {
                    self.at += (other.trailing_zeros() / 8) as usize;
                    break;
                }
                self.at += 8;
            }
        }
    }

    /// A value, at its first byte.
    fn value(&mut self) -> Option<Value> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => {
                self.at += 1;
                self.string().map(Value::String)
            }
            b't' => self.literal(b"true", Value::Bool(true)),
            b'f' => self.literal(b"false", Value::Bool(false)),
            b'n' => self.literal(b"null", Value::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn literal(&mut self, word: &[u8], v: Value) -> Option<Value> {
        if self.text.get(self.at..self.at + word.len()) != Some(word) {
            return None;
        }
        self.at += word.len();
        Some(v)
    }

    /// Enters an array or object, at its opening bracket.
    fn enter(&mut self) -> Option<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return None;
        }
        self.at += 1;
        self.skip_whitespace();
        Some(())
    }

    /// After an item: `true` at a comma (skipped, with the whitespace after it), `false` at the
    /// closing bracket `close` (skipped).
    fn more(&mut self, close: u8) -> Option<bool> {
        self.skip_whitespace();
        let c = self.peek()?;
        self.at += 1;
        if c == b',' {
            self.skip_whitespace();
            Some(true)
        } else {
            (c == close).then_some(false)
        }
    }

    fn array(&mut self) -> Option<Value> {
        self.enter()?;
        let start = self.items.len();
        if self.peek() == Some(b']') {
            self.at += 1;
        } else {
            loop {
                let v = self.value()?;
                self.items.push(v);
                if !self.more(b']')? {
                    break;
                }
            }
        }
        self.depth -= 1;
        Some(Value::Array(self.items.drain(start..).collect()))
    }

    fn object(&mut self) -> Option<Value> {
        self.enter()?;
        let start = self.entries.len();
        if self.peek() == Some(b'}') {
            self.at += 1;
        } else {
            loop {
                if self.peek() != Some(b'"') {
                    return None;
                }
                self.at += 1;
                let key = self.string()?;
                // An object that starts with one of these keys can be a number or raw JSON in
                // serde_json, depending on its features.
                if self.entries.len() == start && key.starts_with("$serde_json::private::") {
                    return None;
                }
                self.skip_whitespace();
                if self.peek() != Some(b':') {
                    return None;
                }
                self.at += 1;
                self.skip_whitespace();
                let v = self.value()?;
                self.entries.push((key, v));
                if !self.more(b'}')? {
                    break;
                }
            }
        }
        self.depth -= 1;
        // Inserted in order, so a repeated key keeps its last value, as in serde_json.
        let mut map = Map::new();
        for (k, v) in self.entries.drain(start..) {
            map.insert(k, v);
        }
        Some(Value::Object(map))
    }

    /// Moves past bytes that are neither a quote, a backslash nor a control character.
    fn skip_plain(&mut self) {
        while let Some(w) = self.word() {
            // A high bit in each byte below 0x20, equal to `"` or equal to `\`. Only bytes past
            // the first such byte can be marked wrongly, so the lowest mark is exact.
            let quote = w ^ splat(b'"');
            let backslash = w ^ splat(b'\\');
            let found = (w.wrapping_sub(splat(0x20)) & !w
                | quote.wrapping_sub(splat(1)) & !quote
                | backslash.wrapping_sub(splat(1)) & !backslash)
                & splat(0x80);
            if found != 0 {
                self.at += (found.trailing_zeros() / 8) as usize;
                return;
            }
            self.at += 8;
        }
        while let Some(c) = self.peek() {
            if c == b'"' || c == b'\\' || c < 0x20 {
                return;
            }
            self.at += 1;
        }
    }

    /// A string, after its opening quote, through its closing quote.
    fn string(&mut self) -> Option<String> {
        let start = self.at;
        self.skip_plain();
        match self.peek()? {
            b'"' => {
                let s = std::str::from_utf8(&self.text[start..self.at]).ok()?;
                self.at += 1;
                return Some(s.to_owned());
            }
            b'\\' => {}
            _ => return None,
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.text[start..self.at]);
        loop {
            self.at += 1;
            self.escape()?;
            let from = self.at;
            self.skip_plain();
            self.scratch.extend_from_slice(&self.text[from..self.at]);
            match self.peek()? {
                b'"' => {
                    self.at += 1;
                    return std::str::from_utf8(&self.scratch).ok().map(str::to_owned);
                }
                b'\\' => {}
                _ => return None,
            }
        }
    }

    /// An escape, after its backslash, decoded onto the scratch string.
    fn escape(&mut self) -> Option<()> {
        let c = self.peek()?;
        self.at += 1;
        let b = match c {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => {
                let n = self.hex4()?;
                let code = match n {
                    0xDC00..=0xDFFF => return None,
                    // A leading surrogate needs a trailing one, escaped, right after it.
                    0xD800..=0xDBFF => {
                        if self.text.get(self.at..self.at + 2) != Some(b"\\u") {
                            return None;
                        }
                        self.at += 2;
                        let low = self.hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return None;
                        }
                        0x1_0000 + ((n - 0xD800) << 10 | (low - 0xDC00))
                    }
                    _ => n,
                };
                let ch = char::from_u32(code)?;
                let mut buf = [0; 4];
                self.scratch.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                return Some(());
            }
            _ => return None,
        };
        self.scratch.push(b);
        Some(())
    }

    /// Four hex digits.
    fn hex4(&mut self) -> Option<u32> {
        let digits = self.text.get(self.at..self.at + 4)?;
        let mut n = 0;
        for &d in digits {
            n = n << 4 | char::from(d).to_digit(16)?;
        }
        self.at += 4;
        Some(n)
    }

    /// A number, at its sign or first digit, with `serde_json`'s rules and arithmetic.
    fn number(&mut self) -> Option<Value> {
        let positive = self.peek() != Some(b'-');
        if !positive {
            self.at += 1;
        }
        let first = self.peek()?;
        self.at += 1;
        let n = match first {
            b'0' => {
                // One leading zero only.
                if let Some(b'0'..=b'9') = self.peek() {
                    return None;
                }
                self.after_integer(positive, 0)?
            }
            b'1'..=b'9' => {
                let mut significand = u64::from(first - b'0');
                loop {
                    match self.peek() {
                        Some(c @ b'0'..=b'9') => {
                            let digit = u64::from(c - b'0');
                            if overflows(significand, digit, u64::MAX) {
                                break float(self.long_integer(positive, significand)?);
                            }
                            self.at += 1;
                            significand = significand * 10 + digit;
                        }
                        _ => break self.after_integer(positive, significand)?,
                    }
                }
            }
            _ => return None,
        };
        Some(Value::Number(n))
    }

    /// The rest of a number after its integer digits, `significand`.
    fn after_integer(&mut self, positive: bool, significand: u64) -> Option<Number> {
        Some(match self.peek() {
            Some(b'.') => float(self.decimal(positive, significand, 0)?),
            Some(b'e' | b'E') => float(self.exponent(positive, significand, 0)?),
            _ if positive => Number::from(significand),
            _ => {
                #[allow(clippy::cast_possible_wrap, reason = "wrapping as serde_json does, then checked")]
                let negated = (significand as i64).wrapping_neg();
                // Past `i64::MIN`, and `-0`, are floats.
                if negated >= 0 {
                    #[allow(clippy::cast_precision_loss, reason = "as serde_json converts it")]
                    float(-(significand as f64))
                } else {
                    Number::from(negated)
                }
            }
        })
    }

    /// The fraction, at its point, and any exponent after it.
    fn decimal(&mut self, positive: bool, mut significand: u64, exponent_before: i32) -> Option<f64> {
        self.at += 1;
        let mut exponent_after = 0;
        while let Some(c @ b'0'..=b'9') = self.peek() {
            let digit = u64::from(c - b'0');
            if overflows(significand, digit, u64::MAX) {
                // Further digits are ignored.
                while let Some(b'0'..=b'9') = self.peek() {
                    self.at += 1;
                }
                let exponent = exponent_before + exponent_after;
                return match self.peek() {
                    Some(b'e' | b'E') => self.exponent(positive, significand, exponent),
                    _ => from_parts(positive, significand, exponent),
                };
            }
            self.at += 1;
            significand = significand * 10 + digit;
            exponent_after -= 1;
        }
        if exponent_after == 0 {
            return None;
        }
        let exponent = exponent_before + exponent_after;
        match self.peek() {
            Some(b'e' | b'E') => self.exponent(positive, significand, exponent),
            _ => from_parts(positive, significand, exponent),
        }
    }

    /// The exponent, at its `e`.
    fn exponent(&mut self, positive: bool, significand: u64, starting: i32) -> Option<f64> {
        self.at += 1;
        let positive_exp = match self.peek() {
            Some(b'+') => {
                self.at += 1;
                true
            }
            Some(b'-') => {
                self.at += 1;
                false
            }
            _ => true,
        };
        let first = self.peek()?;
        self.at += 1;
        if !first.is_ascii_digit() {
            return None;
        }
        let mut exp = i32::from(first - b'0');
        while let Some(c @ b'0'..=b'9') = self.peek() {
            self.at += 1;
            let digit = i32::from(c - b'0');
            if exp >= i32::MAX / 10 && (exp > i32::MAX / 10 || digit > i32::MAX % 10) {
                // Too large to hold: infinity is an error, and any fraction is zero.
                if significand != 0 && positive_exp {
                    return None;
                }
                while let Some(b'0'..=b'9') = self.peek() {
                    self.at += 1;
                }
                return Some(if positive { 0.0 } else { -0.0 });
            }
            exp = exp * 10 + digit;
        }
        let exponent = if positive_exp {
            starting.saturating_add(exp)
        } else {
            starting.saturating_sub(exp)
        };
        from_parts(positive, significand, exponent)
    }

    /// The digits of an integer past what a `u64` holds: ignored, but counted into the exponent.
    fn long_integer(&mut self, positive: bool, significand: u64) -> Option<f64> {
        let mut exponent = 0;
        loop {
            match self.peek() {
                Some(b'0'..=b'9') => {
                    self.at += 1;
                    exponent += 1;
                }
                Some(b'.') => return self.decimal(positive, significand, exponent),
                Some(b'e' | b'E') => return self.exponent(positive, significand, exponent),
                _ => return from_parts(positive, significand, exponent),
            }
        }
    }
}

/// Whether `a * 10 + b` passes `max`.
fn overflows(a: u64, b: u64, max: u64) -> bool {
    a >= max / 10 && (a > max / 10 || b > max % 10)
}

/// A float as the number `serde_json` makes of it (always finite here).
fn float(f: f64) -> Number {
    Number::from_f64(f).expect("a parsed float is finite")
}

/// `significand` × 10^`exponent`, with `serde_json`'s default arithmetic: one multiplication or
/// division by a power of ten, with division by 1e308 first for exponents below -308. `None`
/// when it overflows.
fn from_parts(positive: bool, significand: u64, mut exponent: i32) -> Option<f64> {
    #[allow(clippy::cast_precision_loss, reason = "as serde_json converts it")]
    let mut f = significand as f64;
    let table = pow10();
    loop {
        if let Some(&pow) = table.get(exponent.unsigned_abs() as usize) {
            if exponent >= 0 {
                f *= pow;
                if f.is_infinite() {
                    return None;
                }
            } else {
                f /= pow;
            }
            break;
        }
        if f == 0.0 {
            break;
        }
        if exponent >= 0 {
            return None;
        }
        f /= 1e308;
        exponent += 308;
    }
    Some(if positive { f } else { -f })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Pcg32;

    /// Both parsers on `text`: the same value, or both an error.
    fn check(text: &[u8]) {
        let theirs = serde_json::from_slice::<Value>(text);
        // A build whose serde_json features read numbers differently never uses `parse`.
        match (parse(text).filter(|_| agrees_with_serde_json()), &theirs) {
            (Some(ours), Ok(theirs)) => assert!(
                same(&ours, theirs),
                "{:?}: {ours:?} vs {theirs:?}",
                String::from_utf8_lossy(text)
            ),
            (None, Err(_)) => {}
            (None, Ok(_)) if !agrees_with_serde_json() => {}
            (ours, theirs) => panic!(
                "{:?}: ours {ours:?}, serde_json {theirs:?}",
                String::from_utf8_lossy(text)
            ),
        }
        // What callers get is serde_json's answer either way.
        match (from_slice(text), theirs) {
            (Ok(a), Ok(b)) => assert!(same(&a, &b)),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            _ => panic!("{:?}", String::from_utf8_lossy(text)),
        }
    }

    #[test]
    fn this_build_agrees() {
        // The check says yes exactly when serde_json reads numbers its default way.
        let default = |t: &str| serde_json::from_str::<Value>(t).unwrap();
        let expected = serde_json::to_string(&default("1.10")).is_ok_and(|t| t == "1.1")
            && default("154757138494460336e8").as_f64().map(f64::to_bits)
                == Some(1.547_571_384_944_603_6e25_f64.to_bits());
        assert_eq!(agrees_with_serde_json(), expected);
    }

    #[test]
    fn numbers() {
        let cases = [
            "0",
            "-0",
            "1",
            "-1",
            "0.0",
            "-0.0",
            "1.5",
            "1e5",
            "1E5",
            "1e+5",
            "1e-5",
            "-1.5e-5",
            "0e0",
            "0e-0",
            "18446744073709551615",
            "18446744073709551616",
            "-9223372036854775808",
            "-9223372036854775809",
            "-18446744073709551615",
            "-18446744073709551616",
            "1844674407370955161.5",
            "18446744073709551616.25",
            "184467440737095516161844674407370955161.1844674407370955161",
            "1.7976931348623157e308",
            "1.7976931348623159e308",
            "1e309",
            "-1e309",
            "1e-400",
            "-1e-400",
            "0e99999999999",
            "1e99999999999",
            "1e-99999999999",
            "0.1e-2147483648",
            "1e2147483647",
            "1e-2147483648",
            "4.9e-324",
            "2.4e-324",
            "2.2250738585072014e-308",
            "0.30000000000000004441",
            "123456789012345678901234567890e-10",
            "9007199254740993",
            "9007199254740993.0",
            "0.000000000000000000000000000000000000001",
            "1.00000000000000011102230246251565404236316680908203125",
            "01",
            "-01",
            "1.",
            ".5",
            "-",
            "+1",
            "1e",
            "1e+",
            "1e-",
            "--1",
            "1.e5",
            "0x10",
            "1_0",
            "Infinity",
            "NaN",
            "-a",
            "1e5.5",
            "00",
            "-00",
        ];
        for c in cases {
            check(c.as_bytes());
            check(format!("[{c}]").as_bytes());
            check(format!("{{\"k\": {c} }}").as_bytes());
        }
    }

    #[test]
    fn strings() {
        let cases: [&[u8]; 28] = [
            br#""""#,
            br#""abc""#,
            br#""a\"b""#,
            br#""\\\/\b\f\n\r\t""#,
            br#""\u0041\u00e9\u20AC""#,
            br#""\ud83d\ude00""#,
            br#""\uD83D\uDE00x""#,
            br#""\ud83d""#,
            br#""\ude00""#,
            br#""\ud83dx""#,
            br#""\ud83d\u0041""#,
            br#""\ud83d\ud83d""#,
            br#""\u00""#,
            br#""\u00g0""#,
            br#""\x""#,
            b"\"a\x01b\"",
            b"\"a\x1fb\"",
            b"\"tab\tin\"",
            "\"caf\u{e9} \u{1F600}\"".as_bytes(),
            b"\"\xff\"",
            b"\"\xc3\"",
            b"\"\xc3\\u00a9\"",
            b"\"\xe2\x82\"",
            br#""abc"#,
            br#""abc\"#,
            br#""\u0000""#,
            b"\"\x7f\"",
            br#""0123456789abcdef0123456789\"abcdef""#,
        ];
        for c in cases {
            check(c);
            check(&[b"[", c, b"]"].concat());
            check(&[b"{", c, b": 1}"].concat());
        }
    }

    #[test]
    fn structure() {
        let cases = [
            "",
            " ",
            "{}",
            "[]",
            "[ ]",
            "{ }",
            "[1,2]",
            "[1,]",
            "[,1]",
            "[1 2]",
            "{\"a\":1,}",
            "{\"a\" 1}",
            "{\"a\":}",
            "{1:2}",
            "{\"a\":1 \"b\":2}",
            "{\"a\":1,\"a\":2}",
            "{\"b\":1,\"a\":2,\"b\":3}",
            "true",
            "false",
            "null",
            "tru",
            "nul",
            "nulll",
            "[true,false,null]",
            "\t\r\n [ 1 , { \"x\" : [ ] } ] \n",
            "[1] x",
            "[1]]",
            "{\"a\":1}}",
            "[",
            "{",
            "{\"a\"",
            "{\"a\":",
            "[1",
            "\u{feff}[]",
            "[1]\u{0}",
            "[\"a\"\u{0}]",
            "/**/1",
            "[1]   ",
        ];
        for c in cases {
            check(c.as_bytes());
        }
        // Nesting at serde_json's limit and one past it.
        for depth in [126, 127, 128, 129, 200] {
            check(format!("{}{}", "[".repeat(depth), "]".repeat(depth)).as_bytes());
            check(format!("{}1{}", "{\"a\":".repeat(depth), "}".repeat(depth)).as_bytes());
        }
    }

    /// Random JSON, with the kinds of number, string and spacing `serde_json` treats specially.
    fn random_value(r: &mut Pcg32, depth: u32, out: &mut String) {
        let space = |r: &mut Pcg32, out: &mut String| {
            for _ in 0..r.below(3) {
                out.push([' ', '\n', '\t', '\r'][r.below(4) as usize]);
            }
        };
        space(r, out);
        match r.below(if depth > 4 { 4 } else { 6 }) {
            0 => out.push_str(["true", "false", "null"][r.below(3) as usize]),
            1 => random_number(r, out),
            2 | 3 => random_string(r, out),
            4 => {
                out.push('[');
                let n = r.below(5);
                for i in 0..n {
                    if i > 0 {
                        out.push(',');
                    }
                    random_value(r, depth + 1, out);
                }
                space(r, out);
                out.push(']');
            }
            _ => {
                out.push('{');
                let n = r.below(5);
                for i in 0..n {
                    if i > 0 {
                        out.push(',');
                    }
                    space(r, out);
                    random_string(r, out);
                    space(r, out);
                    out.push(':');
                    random_value(r, depth + 1, out);
                }
                space(r, out);
                out.push('}');
            }
        }
        space(r, out);
    }

    fn random_number(r: &mut Pcg32, out: &mut String) {
        if r.below(2) == 0 {
            out.push('-');
        }
        let digits = |r: &mut Pcg32, n: u32, out: &mut String| {
            for _ in 0..n {
                out.push(char::from(b"0123456789"[r.below(10) as usize]));
            }
        };
        if r.below(3) == 0 {
            out.push('0');
        } else {
            out.push(char::from(b"123456789"[r.below(9) as usize]));
            let n = [r.below(4), r.below(20), r.below(40)][r.below(3) as usize];
            digits(r, n, out);
        }
        if r.below(2) == 0 {
            out.push('.');
            let n = 1 + [r.below(4), r.below(20), r.below(40)][r.below(3) as usize];
            digits(r, n, out);
        }
        if r.below(3) == 0 {
            out.push(['e', 'E'][r.below(2) as usize]);
            match r.below(3) {
                0 => out.push('+'),
                1 => out.push('-'),
                _ => {}
            }
            let n = 1 + [r.below(2), r.below(3), r.below(12)][r.below(3) as usize];
            digits(r, n, out);
        }
    }

    fn random_string(r: &mut Pcg32, out: &mut String) {
        const PIECES: [&str; 16] = [
            "a",
            "name",
            "translation",
            " ",
            "\\\"",
            "\\\\",
            "\\n",
            "\\u0041",
            "\\u00E9",
            "\\ud83d\\ude00",
            "\u{e9}",
            "\u{1F600}",
            "\\/",
            "\\t",
            "0123456789abcdef",
            "\\uFFFF",
        ];
        out.push('"');
        for _ in 0..r.below(6) {
            out.push_str(PIECES[r.below(16) as usize]);
        }
        out.push('"');
    }

    #[test]
    fn random_documents_and_their_damage() {
        let mut r = Pcg32::new(7);
        for _ in 0..20_000 {
            let mut s = String::new();
            random_value(&mut r, 0, &mut s);
            check(s.as_bytes());
            // Damaged: a byte changed, removed or cut off.
            let mut b = s.into_bytes();
            if b.is_empty() {
                continue;
            }
            let at = r.below(u32::try_from(b.len()).unwrap()) as usize;
            match r.below(4) {
                0 => b[at] = r.below(256).to_le_bytes()[0],
                1 => b[at] = b"{}[]\":,\\-.eE0 \x01\xff"[r.below(16) as usize],
                2 => {
                    b.remove(at);
                }
                _ => b.truncate(at),
            }
            check(&b);
        }
    }

    #[test]
    fn corpus_json() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
        let mut stack = vec![root];
        let mut n = 0;
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase());
                let bytes = std::fs::read(&p).unwrap();
                let text = match ext.as_deref() {
                    Some("gltf" | "json") => &bytes[..],
                    Some("glb" | "vrm") if bytes.len() >= 20 && &bytes[16..20] == b"JSON" => {
                        let len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
                        &bytes[20..(20 + len).min(bytes.len())]
                    }
                    _ => continue,
                };
                let text = text.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(text);
                check(text);
                n += 1;
            }
        }
        assert!(n > 50, "only {n} JSON texts");
    }
}
