//! Just enough of OGC well-known text (WKT 1 and 2) for a point cloud's coordinate system: what
//! kind it is, its EPSG code, and the size of its linear units in meters, across and up.

use super::stream::f64_from;

/// A coordinate system as far as a point cloud needs it.
#[derive(Debug, PartialEq)]
pub(crate) struct Crs {
    /// `projected`, `geographic`, `geocentric` or `other`.
    pub kind: &'static str,
    /// The horizontal system's EPSG code.
    pub epsg: Option<u32>,
    /// The vertical system's EPSG code, in a compound system.
    pub vertical_epsg: Option<u32>,
    /// The horizontal linear unit's name and meters, for a projected or geocentric system.
    pub horizontal: Option<(String, f64)>,
    /// The vertical linear unit's name and meters.
    pub vertical: Option<(String, f64)>,
}

/// The deepest nesting read.
const DEPTH: usize = 32;

#[derive(Debug)]
enum Arg {
    Text(String),
    Num(f64),
    Node(Node),
}

#[derive(Debug)]
struct Node {
    key: String,
    args: Vec<Arg>,
}

impl Node {
    fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.args.iter().filter_map(|a| match a {
            Arg::Node(n) => Some(n),
            _ => None,
        })
    }

    fn child(&self, keys: &[&str]) -> Option<&Node> {
        self.nodes()
            .find(|n| keys.iter().any(|k| n.key.eq_ignore_ascii_case(k)))
    }

    fn is(&self, keys: &[&str]) -> bool {
        keys.iter().any(|k| self.key.eq_ignore_ascii_case(k))
    }

    /// The EPSG code in an `AUTHORITY` or `ID` child.
    fn epsg(&self) -> Option<u32> {
        let id = self.child(&["AUTHORITY", "ID"])?;
        match id.args.as_slice() {
            [Arg::Text(a), code, ..] if a.eq_ignore_ascii_case("EPSG") => match code {
                Arg::Text(t) => t.trim().parse().ok(),
                Arg::Num(n) if n.fract() == 0.0 && *n >= 0.0 && *n < 4e9 => {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "checked whole and in range"
                    )]
                    let c = *n as u32;
                    Some(c)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The linear unit: a `UNIT` or `LENGTHUNIT` child, or one in an `AXIS`.
    fn unit(&self) -> Option<(String, f64)> {
        let of = |n: &Node| match n.args.as_slice() {
            [Arg::Text(name), Arg::Num(f), ..] if f.is_finite() && *f > 0.0 => Some((name.clone(), *f)),
            _ => None,
        };
        if let Some(u) = self.child(&["UNIT", "LENGTHUNIT"]).and_then(of) {
            return Some(u);
        }
        self.nodes()
            .filter(|n| n.is(&["AXIS"]))
            .find_map(|a| a.child(&["LENGTHUNIT", "UNIT"]).and_then(of))
    }
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self.s.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn node(&mut self, depth: usize) -> Option<Node> {
        if depth > DEPTH {
            return None;
        }
        self.space();
        let start = self.at;
        while self
            .s
            .get(self.at)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            self.at += 1;
        }
        let key = std::str::from_utf8(&self.s[start..self.at]).ok()?.to_string();
        self.space();
        let mut args = Vec::new();
        if !matches!(self.s.get(self.at), Some(b'[' | b'(')) {
            return (!key.is_empty()).then_some(Node { key, args });
        }
        self.at += 1;
        loop {
            self.space();
            match *self.s.get(self.at)? {
                b']' | b')' => {
                    self.at += 1;
                    break;
                }
                b',' => self.at += 1,
                b'"' => {
                    self.at += 1;
                    let mut t = Vec::new();
                    loop {
                        let c = *self.s.get(self.at)?;
                        self.at += 1;
                        if c == b'"' {
                            if self.s.get(self.at) == Some(&b'"') {
                                self.at += 1;
                            } else {
                                break;
                            }
                        }
                        t.push(c);
                    }
                    args.push(Arg::Text(crate::bytes::text(&t).into_owned()));
                }
                c if c == b'-' || c == b'+' || c == b'.' || c.is_ascii_digit() => {
                    let start = self.at;
                    while self
                        .s
                        .get(self.at)
                        .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E'))
                    {
                        self.at += 1;
                    }
                    args.push(Arg::Num(f64_from(&self.s[start..self.at])?));
                }
                _ => {
                    let before = self.at;
                    let n = self.node(depth + 1)?;
                    if self.at == before {
                        return None;
                    }
                    args.push(Arg::Node(n));
                }
            }
        }
        Some(Node { key, args })
    }
}

const PROJECTED: &[&str] = &["PROJCS", "PROJCRS", "PROJECTEDCRS"];
const GEOGRAPHIC: &[&str] = &[
    "GEOGCS",
    "GEOGCRS",
    "GEOGRAPHICCRS",
    "GEODCRS",
    "GEODETICCRS",
    "BASEGEOGCRS",
];
const GEOCENTRIC: &[&str] = &["GEOCCS"];
const VERTICAL: &[&str] = &["VERT_CS", "VERTCRS", "VERTICALCRS"];
const COMPOUND: &[&str] = &["COMPD_CS", "COMPOUNDCRS"];

impl Crs {
    /// The coordinate system `wkt` describes, or `None` when it doesn't parse.
    pub fn parse(wkt: &str) -> Option<Crs> {
        let root = Parser {
            s: wkt.as_bytes(),
            at: 0,
        }
        .node(0)?;
        let (horizontal, vertical) = if root.is(COMPOUND) {
            let h = root
                .nodes()
                .find(|n| n.is(PROJECTED) || n.is(GEOGRAPHIC) || n.is(GEOCENTRIC));
            (h, root.nodes().find(|n| n.is(VERTICAL)))
        } else if root.is(VERTICAL) {
            (None, Some(&root))
        } else {
            (Some(&root), None)
        };
        let kind = match horizontal {
            Some(h) if h.is(PROJECTED) => "projected",
            Some(h) if h.is(GEOGRAPHIC) => "geographic",
            Some(h) if h.is(GEOCENTRIC) => "geocentric",
            _ => "other",
        };
        // A geographic system's unit is an angle, not a length.
        let linear = matches!(kind, "projected" | "geocentric");
        Some(Crs {
            kind,
            epsg: horizontal.and_then(Node::epsg),
            vertical_epsg: vertical.and_then(Node::epsg),
            horizontal: horizontal.filter(|_| linear).and_then(Node::unit),
            vertical: vertical.and_then(Node::unit),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wkt1_projected_in_feet() {
        let c = Crs::parse(
            r#"COMPD_CS["x",PROJCS["NAD83 / Ohio North (ftUS)",GEOGCS["NAD83",DATUM["d",SPHEROID["s",6378137,298.25]],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert"],UNIT["US survey foot",0.3048006096012192,AUTHORITY["EPSG","9003"]],AUTHORITY["EPSG","3734"]],VERT_CS["NAVD88 height (ftUS)",VERT_DATUM["v",2005],UNIT["US survey foot",0.3048006096012192],AUTHORITY["EPSG","6360"]]]"#,
        )
        .unwrap();
        assert_eq!(c.kind, "projected");
        assert_eq!(c.epsg, Some(3734));
        assert_eq!(c.vertical_epsg, Some(6360));
        assert!((c.horizontal.unwrap().1 - 0.304_800_609_6).abs() < 1e-9);
        assert!(c.vertical.is_some());
    }

    #[test]
    fn wkt2_units_on_the_axes() {
        let c = Crs::parse(
            r#"PROJCRS["WGS 84 / UTM zone 17N",BASEGEOGCRS["WGS 84",ANGLEUNIT["degree",0.0174532925199433]],CONVERSION["UTM"],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",32617]]"#,
        )
        .unwrap();
        assert_eq!(c.epsg, Some(32617));
        assert_eq!(c.horizontal, Some(("metre".into(), 1.0)));
    }

    #[test]
    fn geographic_has_no_linear_unit_and_junk_doesnt_parse() {
        let c = Crs::parse(r#"GEOGCS["WGS 84",UNIT["degree",0.01745],AUTHORITY["EPSG","4326"]]"#).unwrap();
        assert_eq!((c.kind, c.epsg, c.horizontal), ("geographic", Some(4326), None));
        assert!(Crs::parse("PROJCS[\"x\",UNIT[").is_none());
        assert!(Crs::parse(&"A[".repeat(10_000)).is_none());
        assert!(Crs::parse("[[[").is_none());
    }
}
