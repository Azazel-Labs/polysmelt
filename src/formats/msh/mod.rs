//! Gmsh `.msh` files: versions 1.0, 2.0 to 2.2, 4.0 and 4.1, ASCII and binary.
//!
//! A file is a run of sections, each from a `$Name` line to its `$EndName`. `$MeshFormat` gives the version,
//! whether the sections are text or binary, and (version 4) how wide a `size_t` is, 4 or 8 bytes; a binary
//! file then holds the integer one, which tells its byte order. `$Nodes` and `$Elements` are read in each
//! version's layout: one list in versions 1 and 2, blocks by entity in version 4 (tags and coordinates apart in
//! 4.1, together in 4.0). `$PhysicalNames` names the physical groups and `$Entities` (version 4) says which
//! groups each entity is in, and `$PartitionedEntities` (4.1) the same for the parts of a partitioned mesh.
//! Sections not known here, `$Periodic` and `$Parametrizations` among them, are passed over. Version 3, which
//! Gmsh never released as a default, isn't read.
//!
//! Every element type is drawn by its corner nodes, which each type lists first: points as points, lines as
//! lines, triangles and quadrangles as triangles, and tetrahedra, hexahedra, prisms and pyramids as the faces
//! only one of them has, wound to face outward. Polygons, polyhedra and the one-node elements aren't drawn;
//! the source notes count them. Nodes no drawn element uses are left out, and a file with nodes and no
//! elements draws its nodes as points.
//!
//! Each physical group is a mesh and a node of its own, named by `$PhysicalNames` or else `Physical Surface 3`
//! (`Point`, `Curve`, `Surface` or `Volume` by its dimension), with its tag and dimension in the mesh's
//! `extras`. A volume group draws its own boundary, so two groups that meet both draw the faces between them.
//! Elements in no group are one mesh with no name, and an entity in several groups goes to the first. A file
//! with more than 1024 groups is one mesh, the tags in a `_PHYSICAL` attribute.
//!
//! `$NodeData`, `$ElementData` and `$ElementNodeData` become float custom attributes named `_` and the view's
//! name in capitals, as VTK's arrays do: `SCALAR` or `VEC3`, and a tensor's nine values an attribute each. Node
//! data follows its nodes; element data splits a vertex where the elements around it differ; element-node data
//! gives each element its own vertices. Nodes or elements a view leaves out are zero. Only a view's first time
//! step is kept, and a view covering less than a sixteenth of the nodes or elements is left out.
//!
//! Gmsh names no units and no up axis, so both are unknown, and geometry is kept as written.

#[cfg(test)]
mod tests;

use serde_json::{Map, json};

use super::Input;
use super::cells::{self, Array, BAD, Cursor, Elements, Group, Model, Role, int_of, real_of, tag_of, word_of};
use super::stream;
use crate::document::Document;
use crate::mesh::FastMap;
use crate::num::{u32_of, usize_of};
use crate::{Context as _, ErrorKind, Result};

/// What an element's corner nodes make.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Base {
    /// Not drawn.
    X,
    /// A point.
    V,
    /// A line.
    L,
    /// A triangle.
    T,
    /// A quadrangle.
    Q,
    /// A tetrahedron.
    E,
    /// A hexahedron.
    H,
    /// A prism: VTK's wedge.
    W,
    /// A pyramid.
    P,
}

impl Base {
    /// The corner nodes, the VTK cell type they draw as, and the dimension.
    fn drawn(self) -> Option<(usize, u8, u32)> {
        Some(match self {
            Base::X => return None,
            Base::V => (1, 1, 0),
            Base::L => (2, 3, 1),
            Base::T => (3, 5, 2),
            Base::Q => (4, 9, 2),
            Base::E => (4, 10, 3),
            Base::H => (8, 12, 3),
            Base::W => (6, 13, 3),
            Base::P => (5, 14, 3),
        })
    }
}

/// Each element type from 1: how many nodes it has (0 for a type with no fixed count, or none known here) and
/// what its corners make. The counts are Gmsh's own (`gmsh.model.mesh.getElementProperties`).
const TYPES: [(u16, Base); 140] = {
    use Base::{E, H, L, P, Q, T, V, W, X};
    [
        (2, L),
        (3, T),
        (4, Q),
        (4, E),
        (8, H),
        (6, W),
        (5, P),
        (3, L),
        (6, T),
        (9, Q),
        (10, E),
        (27, H),
        (18, W),
        (14, P),
        (1, V),
        (8, Q),
        (20, H),
        (15, W),
        (13, P),
        (9, T),
        (10, T),
        (12, T),
        (15, T),
        (15, T),
        (21, T),
        (4, L),
        (5, L),
        (6, L),
        (20, E),
        (35, E),
        (56, E),
        (22, E),
        (28, E),
        (0, X),
        (0, X),
        (16, Q),
        (25, Q),
        (36, Q),
        (12, Q),
        (16, Q),
        (20, Q),
        (28, T),
        (36, T),
        (45, T),
        (55, T),
        (66, T),
        (49, Q),
        (64, Q),
        (81, Q),
        (100, Q),
        (121, Q),
        (18, T),
        (21, T),
        (24, T),
        (27, T),
        (30, T),
        (24, Q),
        (28, Q),
        (32, Q),
        (36, Q),
        (40, Q),
        (7, L),
        (8, L),
        (9, L),
        (10, L),
        (11, L),
        (0, X),
        (0, X),
        (0, X),
        (0, X),
        (84, E),
        (120, E),
        (165, E),
        (220, E),
        (286, E),
        (0, X),
        (0, X),
        (0, X),
        (34, E),
        (40, E),
        (46, E),
        (52, E),
        (58, E),
        (1, X),
        (1, X),
        (1, X),
        (1, X),
        (1, X),
        (1, X),
        (40, W),
        (75, W),
        (64, H),
        (125, H),
        (216, H),
        (343, H),
        (512, H),
        (729, H),
        (1000, H),
        (32, H),
        (44, H),
        (56, H),
        (68, H),
        (80, H),
        (92, H),
        (104, H),
        (126, W),
        (196, W),
        (288, W),
        (405, W),
        (550, W),
        (24, W),
        (33, W),
        (42, W),
        (51, W),
        (60, W),
        (69, W),
        (78, W),
        (30, P),
        (55, P),
        (91, P),
        (140, P),
        (204, P),
        (285, P),
        (385, P),
        (21, P),
        (29, P),
        (37, P),
        (45, P),
        (53, P),
        (61, P),
        (69, P),
        (1, X),
        (0, X),
        (0, X),
        (0, X),
        (0, X),
        (16, E),
        (4, T),
        (5, E),
        (4, X),
    ]
};

/// Element type `ty`'s node count and what it draws as; `None` for a type with no fixed count.
fn element_type(ty: i64) -> Option<(usize, Base)> {
    let &(nodes, base) = TYPES.get(usize::try_from(ty.checked_sub(1)?).ok()?)?;
    (nodes > 0).then_some((usize::from(nodes), base))
}

/// What Gmsh calls a group of each dimension.
const KINDS: [&str; 4] = ["Point", "Curve", "Surface", "Volume"];

/// The dimension a physical name has in a version 2.0 file, which gives none.
const ANY: u32 = u32::MAX;

/// A view is kept when it has values for at least one in this many of the nodes or elements.
const COVERAGE: usize = 16;

/// The most views kept.
const MAX_VIEWS: usize = 64;

/// The layouts a file's sections come in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Version {
    V1,
    V2,
    V40,
    V41,
}

/// What a view's values are on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum On {
    Nodes,
    Elements,
    ElementNodes,
}

/// One `$NodeData`, `$ElementData` or `$ElementNodeData` section.
struct View {
    on: On,
    name: String,
    comps: usize,
    /// The node or element each record is for.
    tags: Vec<u32>,
    /// With [`On::ElementNodes`], how many nodes each record has values for.
    nodes: Vec<u32>,
    values: Vec<f32>,
}

/// The index of each tag among the nodes or elements a file lists.
enum Lookup {
    /// Tags from 1 in order, as most files have them.
    Identity(u32),
    Dense(Vec<u32>),
    Sparse(FastMap<u32, u32>),
}

impl Lookup {
    fn of(tags: &[u32]) -> Self {
        let n = tags.len();
        if tags.iter().enumerate().all(|(i, &t)| t as usize == i + 1) {
            return Lookup::Identity(u32_of(n));
        }
        let most = tags.iter().copied().filter(|&t| t != BAD).max().unwrap_or(0) as usize;
        let known = tags.iter().enumerate().filter(|(_, t)| **t != BAD);
        if most <= n.saturating_mul(4).saturating_add(1024) {
            let mut table = vec![BAD; most + 1];
            for (i, &t) in known {
                table[t as usize] = u32_of(i);
            }
            Lookup::Dense(table)
        } else {
            Lookup::Sparse(known.map(|(i, &t)| (t, u32_of(i))).collect())
        }
    }

    /// The index of `tag`, or [`BAD`].
    #[inline]
    fn get(&self, tag: u32) -> u32 {
        match self {
            Lookup::Identity(n) => {
                let i = tag.wrapping_sub(1);
                if i < *n { i } else { BAD }
            }
            Lookup::Dense(table) => table.get(tag as usize).copied().unwrap_or(BAD),
            Lookup::Sparse(map) => map.get(&tag).copied().unwrap_or(BAD),
        }
    }
}

struct Reader<'a> {
    c: Cursor<'a>,
    version: Version,
    binary: bool,
    big: bool,
    /// The bytes of a `size_t` in a version 4 binary file.
    wide: usize,
    node_tags: Vec<u32>,
    /// The elements; their nodes are tags until the end.
    el: Elements,
    element_tags: Vec<u32>,
    /// Each physical group's place in the elements' groups, by dimension and tag.
    groups: FastMap<(u32, i64), u32>,
    names: FastMap<(u32, i64), String>,
    /// Each entity's first physical group, by dimension and tag.
    entities: FastMap<(u32, i64), i64>,
    views: Vec<View>,
    /// Time steps after a view's first, and views past [`MAX_VIEWS`].
    skipped_views: usize,
    /// Elements of types that aren't drawn.
    skipped: usize,
}

pub(crate) fn load(input: Input<'_>, keep: crate::Keep) -> Result<Document> {
    let data = input.read()?;
    let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&data);
    ensure!(data.iter().any(|&b| !stream::space(b)), Empty, "Gmsh file is empty");
    let mut r = Reader {
        c: Cursor::new(data, "Gmsh"),
        version: Version::V1,
        binary: false,
        big: false,
        wide: 8,
        node_tags: Vec::new(),
        el: Elements::new("physical"),
        element_tags: Vec::new(),
        groups: FastMap::default(),
        names: FastMap::default(),
        entities: FastMap::default(),
        views: Vec::new(),
        skipped_views: 0,
        skipped: 0,
    };
    let mut source = Map::new();
    let mut version = String::from("1.0");
    let mut first = true;
    while let Some(name) = r.section() {
        match name {
            b"MeshFormat" => version = r.format()?,
            _ if first && !matches!(name, b"NOD" | b"ELM") => {
                bail!(UnknownFormat, "not a Gmsh mesh: it starts with ${}", crate::shown(name))
            }
            b"PhysicalNames" => r.physical_names()?,
            b"Entities" if r.version != Version::V2 => r.entities()?,
            b"PartitionedEntities" if r.version == Version::V41 => r.partitioned_entities()?,
            b"Nodes" | b"NOD" => r.nodes()?,
            b"Elements" | b"ELM" => r.elements()?,
            b"NodeData" => r.view(On::Nodes)?,
            b"ElementData" => r.view(On::Elements)?,
            b"ElementNodeData" => r.view(On::ElementNodes)?,
            _ => {}
        }
        first = false;
        r.end(name);
    }
    ensure!(!first, UnknownFormat, "not a Gmsh mesh: it has no sections");
    source.insert("encoding".into(), json!(if r.binary { "binary" } else { "ascii" }));
    if r.big {
        source.insert("byteOrder".into(), json!("big"));
    }
    source.insert("nodes".into(), json!(r.node_tags.len()));
    source.insert("elements".into(), json!(r.el.len() + r.skipped));
    if r.skipped > 0 {
        source.insert("skippedElements".into(), json!(r.skipped));
    }
    if r.skipped_views > 0 {
        source.insert("skippedViews".into(), json!(r.skipped_views));
    }
    let mut model = Model {
        source,
        ..Model::default()
    };
    model.info.version = Some(version);
    let (el, left_out) = r.finish();
    if !left_out.is_empty() {
        model.source.insert("skippedArrays".into(), json!(left_out));
    }
    model.pieces = el.pieces();
    cells::build(model, "Gmsh", input.limit, keep.crease_angle())
}

impl<'a> Reader<'a> {
    /// The next section's name, the cursor just past its line; `None` at the end of the file.
    fn section(&mut self) -> Option<&'a [u8]> {
        let d = self.c.d;
        loop {
            self.c.skip_space();
            match d.get(self.c.at) {
                None => return None,
                Some(b'$') => {
                    let line = stream::next_line(d, &mut self.c.at).unwrap_or_default();
                    return Some(stream::trim(&line[1..]));
                }
                // Something between sections: on to the next line that starts one.
                Some(_) => self.c.skip_line(),
            }
        }
    }

    /// Moves to the end marker of section `name`, wherever reading it stopped.
    fn end(&mut self, name: &[u8]) {
        let d = self.c.d;
        let mut marker = b"$End".to_vec();
        if matches!(name, b"NOD" | b"ELM") {
            marker = b"$END".to_vec();
        }
        marker.extend_from_slice(name);
        let mut at = self.c.at.min(d.len());
        while let Some(n) = d[at..].iter().position(|&b| b == b'$') {
            at += n;
            if d[at..].starts_with(&marker) {
                self.c.at = at + marker.len();
                return;
            }
            at += 1;
        }
        self.c.at = d.len();
    }

    /// The bytes left.
    fn left(&self) -> usize {
        self.c.d.len() - self.c.at.min(self.c.d.len())
    }

    /// `n` when the file has room for that many items of at least `each` bytes.
    fn count(&self, n: usize, each: usize, what: &str) -> Result<usize> {
        ensure!(
            n <= self.left() / each.max(1) + usize::from(!self.binary),
            Truncated,
            "Gmsh file is cut short: it claims {n} {what}, more than it holds"
        );
        Ok(n)
    }

    /// The next `n` bytes.
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let end = self.c.at.checked_add(n);
        let Some(b) = end.and_then(|end| self.c.d.get(self.c.at..end)) else {
            bail!(Truncated, "Gmsh file is cut short in its {what}");
        };
        self.c.at += n;
        Ok(b)
    }

    fn token(&mut self, what: &str) -> Result<&'a [u8]> {
        self.c
            .token()
            .with_context(ErrorKind::Truncated, || format!("Gmsh file is cut short in its {what}"))
    }

    /// The next integer: a token of text, or four bytes.
    fn int(&mut self, what: &str) -> Result<i64> {
        if self.binary {
            return Ok(i64::from(int_of(self.take(4, what)?, self.big)));
        }
        let t = self.token(what)?;
        stream::int::<i64>(t).with_context(ErrorKind::Corrupt, || {
            format!("Gmsh {what}: {} is not a whole number", crate::shown(t))
        })
    }

    /// The next count: a token of text, or a `size_t`.
    fn size(&mut self, what: &str) -> Result<usize> {
        if self.binary {
            return Ok(usize_of(word_of(self.take(self.wide, what)?, self.big)));
        }
        let t = self.token(what)?;
        stream::int::<usize>(t).with_context(ErrorKind::Corrupt, || {
            format!("Gmsh {what}: {} is not a count", crate::shown(t))
        })
    }

    /// Passes over `n` doubles.
    fn skip_reals(&mut self, n: usize, what: &str) -> Result<()> {
        if self.binary {
            self.take(n.saturating_mul(8), what)?;
        } else {
            for _ in 0..n {
                self.token(what)?;
            }
        }
        Ok(())
    }

    /// The whole number alone on the next line that isn't blank, the cursor past the line's end.
    fn line_int(&mut self, what: &str) -> Result<i64> {
        let line = self
            .c
            .line()
            .with_context(ErrorKind::Truncated, || format!("Gmsh file is cut short in its {what}"))?;
        let t = stream::tokens(line).next().unwrap_or_default();
        stream::int::<i64>(t).with_context(ErrorKind::Corrupt, || {
            format!("Gmsh {what}: {} is not a whole number", crate::shown(t))
        })
    }

    /// A count alone on a line.
    fn line_count(&mut self, what: &str) -> Result<usize> {
        let n = self.line_int(what)?;
        usize::try_from(n).with_context(ErrorKind::Corrupt, || format!("Gmsh {what} count is {n}"))
    }

    /// `$MeshFormat`: the version, the encoding and the data size, then a binary file's one. Returns the version
    /// as written.
    fn format(&mut self) -> Result<String> {
        let line = self.c.line().unwrap_or_default();
        let mut words = stream::tokens(line);
        let written = words.next().unwrap_or_default();
        let number = stream::f64_from(written).with_context(ErrorKind::Corrupt, || {
            format!("Gmsh version {} is not a number", crate::shown(written))
        })?;
        self.version = if (2.0..3.0).contains(&number) {
            Version::V2
        } else if (4.0..4.1).contains(&number) {
            Version::V40
        } else if (4.1..5.0).contains(&number) {
            Version::V41
        } else if (1.0..2.0).contains(&number) {
            Version::V1
        } else {
            bail!(
                Unsupported,
                "Gmsh files of version {} aren't read",
                crate::bytes::text(written)
            );
        };
        let kind = words.next().and_then(stream::int::<i64>);
        let size = words.next().and_then(stream::int::<usize>);
        let (Some(kind), Some(size)) = (kind, size) else {
            bail!(Corrupt, "Gmsh $MeshFormat doesn't give a file type and a data size");
        };
        self.binary = kind == 1;
        ensure!(kind == 0 || kind == 1, Unsupported, "Gmsh file type {kind} isn't read");
        if self.version == Version::V2 {
            // Version 2's data size is a double's.
            ensure!(
                size == 8,
                Unsupported,
                "Gmsh data size {size} isn't read: doubles are 8 bytes"
            );
        } else {
            ensure!(
                size == 4 || size == 8,
                Unsupported,
                "Gmsh data size {size} isn't read: a size_t is 4 or 8 bytes"
            );
            self.wide = size;
        }
        if self.binary {
            let one = self.take(4, "header")?;
            self.big = match (int_of(one, false), int_of(one, true)) {
                (1, _) => false,
                (_, 1) => true,
                _ => bail!(
                    Corrupt,
                    "Gmsh binary file doesn't hold the one that tells its byte order"
                ),
            };
        }
        Ok(crate::bytes::text(written).into_owned())
    }

    /// `$PhysicalNames`, text in every version: a dimension (from version 2.1), a tag and a quoted name a line.
    fn physical_names(&mut self) -> Result<()> {
        let n = self.line_count("physical names")?;
        for _ in 0..n {
            let Some(line) = self.c.line() else {
                bail!(Truncated, "Gmsh file is cut short in its physical names");
            };
            if line.first() == Some(&b'$') {
                bail!(Truncated, "Gmsh file is cut short in its physical names");
            }
            let quote = line.iter().position(|&b| b == b'"').unwrap_or(line.len());
            let numbers: Vec<i64> = stream::tokens(&line[..quote]).filter_map(stream::int::<i64>).collect();
            let name = line.get(quote + 1..).unwrap_or_default();
            let name = &name[..name.iter().rposition(|&b| b == b'"').unwrap_or(name.len())];
            let key = match numbers[..] {
                [dim, tag] => (u32::try_from(dim).unwrap_or(ANY), tag),
                [tag] => (ANY, tag),
                _ => continue,
            };
            if !name.is_empty() {
                self.names.insert(key, crate::bytes::text(name).into_owned());
            }
        }
        Ok(())
    }

    /// Version 4's `$Entities`: each point, curve, surface and volume with the physical groups it's in.
    fn entities(&mut self) -> Result<()> {
        let mut counts = [0usize; 4];
        for n in &mut counts {
            *n = self.size("entities")?;
        }
        for (dim, &n) in (0u32..).zip(&counts) {
            // A point has its place, and in 4.0 a box like the rest.
            let reals = if dim == 0 && self.version == Version::V41 { 3 } else { 6 };
            self.count(
                n,
                if self.binary {
                    4 + 8 * reals + self.wide
                } else {
                    2 * (reals + 2)
                },
                "entities",
            )?;
            for _ in 0..n {
                let tag = self.int("entities")?;
                self.skip_reals(reals, "entities")?;
                self.physicals(dim, tag)?;
            }
        }
        Ok(())
    }

    /// The end of an entity's line: the physical groups it's in, and for all but a point the entities that
    /// bound it.
    fn physicals(&mut self, dim: u32, tag: i64) -> Result<()> {
        let physicals = self.size("entities")?;
        self.count(physicals, if self.binary { 4 } else { 2 }, "physical tags")?;
        for k in 0..physicals {
            // A negative tag is the group with the entity turned around.
            let physical = self.int("entities")?.abs();
            if k == 0 && physical != 0 {
                self.entities.insert((dim, tag), physical);
            }
        }
        if dim > 0 {
            let bounds = self.size("entities")?;
            self.count(bounds, if self.binary { 4 } else { 2 }, "bounding entities")?;
            for _ in 0..bounds {
                self.int("entities")?;
            }
        }
        Ok(())
    }

    /// 4.1's `$PartitionedEntities`: the parts a partitioned mesh's entities are cut into, which its elements
    /// are on. Each is listed with its parent, its partitions and the physical groups it's in.
    fn partitioned_entities(&mut self) -> Result<()> {
        self.size("partitions")?;
        let ghosts = self.size("partitions")?;
        self.count(ghosts, if self.binary { 8 } else { 4 }, "ghost entities")?;
        for _ in 0..2 * ghosts {
            self.int("partitions")?;
        }
        let mut counts = [0usize; 4];
        for n in &mut counts {
            *n = self.size("partitions")?;
        }
        for (dim, &n) in (0u32..).zip(&counts) {
            let reals = if dim == 0 { 3 } else { 6 };
            self.count(
                n,
                if self.binary {
                    12 + 8 * reals + 2 * self.wide
                } else {
                    2 * (reals + 5)
                },
                "entities",
            )?;
            for _ in 0..n {
                let tag = self.int("partitions")?;
                // The parent's dimension and tag.
                self.int("partitions")?;
                self.int("partitions")?;
                let partitions = self.size("partitions")?;
                self.count(partitions, if self.binary { 4 } else { 2 }, "partition tags")?;
                for _ in 0..partitions {
                    self.int("partitions")?;
                }
                self.skip_reals(reals, "partitions")?;
                self.physicals(dim, tag)?;
            }
        }
        Ok(())
    }

    /// Checks the nodes stay few enough to index.
    fn room_for_nodes(&self, more: usize) -> Result<()> {
        ensure!(
            self.node_tags.len().saturating_add(more) < BAD as usize,
            LimitExceeded,
            "Gmsh file has too many nodes"
        );
        Ok(())
    }

    fn nodes(&mut self) -> Result<()> {
        match self.version {
            Version::V1 | Version::V2 => self.nodes_v2(),
            Version::V40 | Version::V41 => self.nodes_v4(),
        }
    }

    /// Versions 1 and 2: a count, then each node's tag and place.
    fn nodes_v2(&mut self) -> Result<()> {
        let n = self.line_count("nodes")?;
        self.count(n, if self.binary { 28 } else { 8 }, "nodes")?;
        self.room_for_nodes(n)?;
        self.node_tags.reserve(n);
        self.el.points.reserve(n);
        if self.binary {
            let big = self.big;
            for record in self.take(n * 28, "nodes")?.chunks_exact(28) {
                self.node_tags.push(tag_of(int_of(record, big)));
                self.el.points.push([4, 12, 20].map(|at| real_of(&record[at..], big)));
            }
        } else {
            for _ in 0..n {
                let tag = self.text_tag()?;
                let p = self.text_point()?;
                self.node_tags.push(tag);
                self.el.points.push(p);
            }
        }
        Ok(())
    }

    /// A node's tag from the text.
    #[inline]
    fn text_tag(&mut self) -> Result<u32> {
        match self.c.id("node")? {
            Some(tag) => Ok(tag),
            None => bail!(Truncated, "Gmsh file is cut short in its nodes"),
        }
    }

    /// A node's place from the text.
    #[inline]
    fn text_point(&mut self) -> Result<[f32; 3]> {
        let mut p = [0.0; 3];
        for v in &mut p {
            let Some(x) = self.c.float("node")? else {
                bail!(Truncated, "Gmsh file is cut short in its nodes");
            };
            *v = x;
        }
        Ok(p)
    }

    /// Version 4: blocks of nodes by entity. 4.1 lists a block's tags and then its coordinates; 4.0 gives each
    /// node's tag with its coordinates. A parametric block has its entity's coordinates after each node's.
    fn nodes_v4(&mut self) -> Result<()> {
        let v41 = self.version == Version::V41;
        let blocks = self.size("nodes")?;
        let total = self.size("nodes")?;
        if v41 {
            // The least and the greatest tag.
            self.size("nodes")?;
            self.size("nodes")?;
        }
        self.count(total, if self.binary { 24 + self.wide.min(4) } else { 8 }, "nodes")?;
        self.room_for_nodes(total)?;
        self.node_tags.reserve(total);
        self.el.points.reserve(total);
        for _ in 0..blocks {
            let (a, b) = (self.int("nodes")?, self.int("nodes")?);
            let dim = if v41 { a } else { b };
            let parametric = self.int("nodes")? != 0;
            let n = self.size("nodes")?;
            let extra = if parametric {
                usize::try_from(dim.clamp(0, 3)).unwrap_or(0)
            } else {
                0
            };
            let tag_bytes = if v41 { self.wide } else { 4 };
            let each = if self.binary { tag_bytes + 8 * (3 + extra) } else { 8 };
            self.count(n, each, "nodes")?;
            self.room_for_nodes(n)?;
            let big = self.big;
            if self.binary && v41 {
                for tag in self.take(n * tag_bytes, "nodes")?.chunks_exact(tag_bytes) {
                    self.node_tags.push(tag_of(word_of(tag, big)));
                }
                let stride = 8 * (3 + extra);
                for record in self.take(n * stride, "nodes")?.chunks_exact(stride) {
                    self.el.points.push([0, 8, 16].map(|at| real_of(&record[at..], big)));
                }
            } else if self.binary {
                let stride = 4 + 8 * (3 + extra);
                for record in self.take(n * stride, "nodes")?.chunks_exact(stride) {
                    self.node_tags.push(tag_of(int_of(record, big)));
                    self.el.points.push([4, 12, 20].map(|at| real_of(&record[at..], big)));
                }
            } else {
                if v41 {
                    for _ in 0..n {
                        let tag = self.text_tag()?;
                        self.node_tags.push(tag);
                    }
                }
                for _ in 0..n {
                    if !v41 {
                        let tag = self.text_tag()?;
                        self.node_tags.push(tag);
                    }
                    let p = self.text_point()?;
                    self.el.points.push(p);
                    if parametric {
                        self.c.skip_line();
                    }
                }
            }
        }
        Ok(())
    }

    /// The elements' group for physical group `tag` of dimension `dim`: [`BAD`] for none.
    fn group(&mut self, dim: u32, tag: i64) -> u32 {
        if tag == 0 {
            return BAD;
        }
        let next = u32_of(self.el.groups.len());
        let at = *self.groups.entry((dim, tag)).or_insert(next);
        if at == next {
            self.el.groups.push(Group {
                name: String::new(),
                extras: json!({ "physical": tag, "dimension": dim }),
                tag,
            });
        }
        at
    }

    /// Checks the elements' nodes stay few enough to index.
    fn room_for_elements(&self) -> Result<()> {
        ensure!(
            self.el.conn.len() < BAD as usize && self.el.len() < BAD as usize,
            LimitExceeded,
            "Gmsh file has too many elements"
        );
        Ok(())
    }

    fn elements(&mut self) -> Result<()> {
        match (self.version, self.binary) {
            (Version::V1 | Version::V2, false) => self.elements_v2_text(),
            (Version::V1 | Version::V2, true) => self.elements_v2_binary(),
            (Version::V40 | Version::V41, _) => self.elements_v4(),
        }
    }

    /// An element's corner tags from the text, its other nodes passed over.
    #[inline]
    fn text_nodes(&mut self, nodes: usize, corners: usize) -> Result<()> {
        for k in 0..nodes {
            let Some(tag) = self.c.id("element")? else {
                bail!(Truncated, "Gmsh file is cut short in its elements");
            };
            if k < corners {
                self.el.conn.push(tag);
            }
        }
        Ok(())
    }

    /// Versions 1 and 2 as text: a count, then an element a line. Version 2 gives its tag, its type, a count
    /// of tags and the tags (the physical group first), then its nodes; version 1 its tag, its type, its
    /// physical group, its entity, a node count and its nodes.
    fn elements_v2_text(&mut self) -> Result<()> {
        let n = self.line_count("elements")?;
        self.count(n, 6, "elements")?;
        self.element_tags.reserve(n);
        let v1 = self.version == Version::V1;
        for _ in 0..n {
            let Some(tag) = self.c.id("element")? else {
                bail!(Truncated, "Gmsh file is cut short in its elements");
            };
            let ty = self.int("element")?;
            let mut physical = 0;
            let mut listed = None;
            if v1 {
                physical = self.int("element")?;
                self.int("element")?;
                listed = Some(usize_of(self.int("element")?.unsigned_abs()));
            } else {
                let tags = self.int("element")?;
                for k in 0..tags {
                    let t = self.int("element")?;
                    if k == 0 {
                        physical = t;
                    }
                }
            }
            let known = element_type(ty).and_then(|(nodes, base)| Some((nodes, base.drawn()?)));
            let Some((nodes, (corners, vtk, dim))) = known else {
                // A type that isn't drawn: the rest of its line is its nodes.
                self.c.skip_line();
                self.skipped += 1;
                continue;
            };
            let nodes = listed.unwrap_or(nodes);
            ensure!(
                nodes >= corners,
                Corrupt,
                "Gmsh element {tag} has too few nodes for its type"
            );
            self.text_nodes(nodes, corners)?;
            let group = self.group(dim, physical.abs());
            self.el.close(vtk, group);
            self.element_tags.push(tag);
            self.room_for_elements()?;
        }
        Ok(())
    }

    /// Version 2 as binary: a count as text, then runs of elements of one type: the type, how many follow and
    /// how many tags each has, and for each its tag, its tags and its nodes, all four-byte integers.
    fn elements_v2_binary(&mut self) -> Result<()> {
        let n = self.line_count("elements")?;
        self.count(n, 8, "elements")?;
        self.element_tags.reserve(n);
        let big = self.big;
        let mut read = 0;
        while read < n {
            let head = self.take(12, "elements")?;
            let (ty, follow, tags) = (int_of(head, big), int_of(&head[4..], big), int_of(&head[8..], big));
            let Some((nodes, base)) = element_type(i64::from(ty)) else {
                bail!(Unsupported, "Gmsh element type {ty} isn't read from a binary file");
            };
            let (Ok(follow), Ok(tags)) = (usize::try_from(follow), usize::try_from(tags)) else {
                bail!(Corrupt, "Gmsh elements have a negative count");
            };
            ensure!(follow > 0, Corrupt, "Gmsh elements come in a run of none");
            ensure!(tags <= 64, Corrupt, "Gmsh elements have {tags} tags each");
            let stride = 4 * (1 + tags + nodes);
            self.count(follow, stride, "elements")?;
            let records = self.take(follow * stride, "elements")?;
            read += follow;
            let Some((corners, vtk, dim)) = base.drawn() else {
                self.skipped += follow;
                continue;
            };
            for record in records.chunks_exact(stride) {
                let physical = if tags > 0 {
                    i64::from(int_of(&record[4..], big))
                } else {
                    0
                };
                let first = 4 * (1 + tags);
                self.el.conn.extend(
                    record[first..first + 4 * corners]
                        .chunks_exact(4)
                        .map(|b| tag_of(int_of(b, big))),
                );
                let group = self.group(dim, physical.abs());
                self.el.close(vtk, group);
                self.element_tags.push(tag_of(int_of(record, big)));
            }
            self.room_for_elements()?;
        }
        Ok(())
    }

    /// Version 4: blocks of elements of one type on one entity, each element its tag and its nodes. The tags
    /// are a `size_t` each in 4.1 and an integer in 4.0.
    fn elements_v4(&mut self) -> Result<()> {
        let v41 = self.version == Version::V41;
        let blocks = self.size("elements")?;
        let total = self.size("elements")?;
        if v41 {
            self.size("elements")?;
            self.size("elements")?;
        }
        self.count(total, if self.binary { 8 } else { 4 }, "elements")?;
        self.element_tags.reserve(total);
        for _ in 0..blocks {
            let (a, b) = (self.int("elements")?, self.int("elements")?);
            let (entity_dim, entity) = if v41 { (a, b) } else { (b, a) };
            let ty = self.int("elements")?;
            let n = self.size("elements")?;
            let physical = u32::try_from(entity_dim)
                .ok()
                .and_then(|dim| self.entities.get(&(dim, entity)).copied())
                .unwrap_or(0);
            let known = element_type(ty);
            let drawn = known.and_then(|(_, base)| base.drawn());
            let group = match drawn {
                Some((_, _, dim)) => self.group(dim, physical),
                None => BAD,
            };
            if self.binary {
                let Some((nodes, _)) = known else {
                    bail!(Unsupported, "Gmsh element type {ty} isn't read from a binary file");
                };
                let word = if v41 { self.wide } else { 4 };
                let stride = word * (1 + nodes);
                self.count(n, stride, "elements")?;
                let records = self.take(n * stride, "elements")?;
                let Some((corners, vtk, _)) = drawn else {
                    self.skipped += n;
                    continue;
                };
                let big = self.big;
                for record in records.chunks_exact(stride) {
                    let mut words = record.chunks_exact(word).map(|b| tag_of(word_of(b, big)));
                    self.element_tags.push(words.next().unwrap_or(BAD));
                    self.el.conn.extend(words.take(corners));
                    self.el.close(vtk, group);
                }
            } else {
                self.count(n, 4, "elements")?;
                // The block's line ends before its first element's begins.
                self.c.skip_line();
                let Some(((nodes, _), (corners, vtk, _))) = known.zip(drawn) else {
                    for _ in 0..n {
                        self.c.skip_line();
                    }
                    self.skipped += n;
                    continue;
                };
                for _ in 0..n {
                    let Some(tag) = self.c.id("element")? else {
                        bail!(Truncated, "Gmsh file is cut short in its elements");
                    };
                    self.text_nodes(nodes, corners)?;
                    self.el.close(vtk, group);
                    self.element_tags.push(tag);
                }
            }
            self.room_for_elements()?;
        }
        Ok(())
    }

    /// A `$NodeData`, `$ElementData` or `$ElementNodeData` section: string tags (the view's name first), real
    /// tags (the time) and integer tags (the time step, the components of a value, and how many records
    /// follow), each a count and then a line apiece, as text in every file. Then the records: a node or
    /// element tag and its values, and for element-node data how many nodes the values are for.
    fn view(&mut self, on: On) -> Result<()> {
        let strings = self.line_count("data")?;
        let mut name = String::new();
        for k in 0..strings {
            let line = self.c.line().unwrap_or_default();
            ensure!(
                line.first() != Some(&b'$'),
                Truncated,
                "Gmsh file is cut short in its data"
            );
            if k == 0 {
                let text = crate::bytes::text(line);
                text.trim().trim_matches('"').clone_into(&mut name);
            }
        }
        let reals = self.line_count("data")?;
        for _ in 0..reals {
            let line = self.c.line().unwrap_or_default();
            ensure!(
                line.first() != Some(&b'$'),
                Truncated,
                "Gmsh file is cut short in its data"
            );
        }
        let integers = self.line_count("data")?;
        let mut tags = [0i64; 3];
        for k in 0..integers {
            let v = self.line_int("data")?;
            if let Some(t) = tags.get_mut(k) {
                *t = v;
            }
        }
        let [_, comps, records] = tags;
        let (Ok(comps), Ok(records)) = (usize::try_from(comps), usize::try_from(records)) else {
            bail!(Corrupt, "Gmsh data has a negative count");
        };
        ensure!(
            integers >= 3 && comps > 0,
            Corrupt,
            "Gmsh data doesn't say how many values it has"
        );
        ensure!(comps <= 1 << 16, Corrupt, "Gmsh data has {comps} components a value");
        let per_node = on == On::ElementNodes;
        let least = if self.binary {
            4 + 8 * comps + 4 * usize::from(per_node)
        } else {
            2 * (1 + comps + usize::from(per_node))
        };
        self.count(records, least, "data values")?;
        // Only a view's first time step is kept.
        let keep = self.views.len() < MAX_VIEWS && !self.views.iter().any(|v| v.on == on && v.name == name);
        let mut view = View {
            on,
            name,
            comps,
            tags: Vec::new(),
            nodes: Vec::new(),
            values: Vec::new(),
        };
        if keep {
            view.tags.reserve(records);
            if !per_node {
                view.values.reserve(records * comps);
            }
        }
        let big = self.big;
        for _ in 0..records {
            let mut count = comps;
            let tag;
            if self.binary {
                tag = tag_of(int_of(self.take(4, "data")?, big));
                if per_node {
                    let nodes = usize::try_from(int_of(self.take(4, "data")?, big)).unwrap_or(0);
                    count = self.count(nodes, 8 * comps, "data values")? * comps;
                }
                let values = self.take(count * 8, "data")?;
                if keep {
                    view.values.extend(values.chunks_exact(8).map(|b| real_of(b, big)));
                }
            } else {
                let Some(t) = self.c.id("data")? else {
                    bail!(Truncated, "Gmsh file is cut short in its data");
                };
                tag = t;
                if per_node {
                    let nodes = usize_of(self.int("data")?.unsigned_abs());
                    count = self.count(nodes, 2 * comps, "data values")? * comps;
                }
                for _ in 0..count {
                    let Some(v) = self.c.float("data")? else {
                        bail!(Truncated, "Gmsh file is cut short in its data");
                    };
                    if keep {
                        view.values.push(v);
                    }
                }
            }
            if keep {
                view.tags.push(tag);
                if per_node {
                    view.nodes.push(u32_of((count / comps).min(BAD as usize - 1)));
                }
            }
        }
        if keep {
            self.views.push(view);
        } else {
            self.skipped_views += 1;
        }
        Ok(())
    }

    /// Turns tags into indices, names the groups and lays the views out over the nodes and elements. Returns
    /// the elements and the names of the views left out.
    fn finish(mut self) -> (Elements, Vec<String>) {
        let nodes = Lookup::of(&self.node_tags);
        for tag in &mut self.el.conn {
            *tag = nodes.get(*tag);
        }
        for (&(dim, tag), &at) in &self.groups {
            let name = self.names.get(&(dim, tag)).or_else(|| self.names.get(&(ANY, tag)));
            self.el.groups[at as usize].name = match name {
                Some(name) => name.clone(),
                None => format!("Physical {} {tag}", KINDS[dim.min(3) as usize]),
            };
        }
        let mut left_out = Vec::new();
        if self.views.is_empty() {
            return (self.el, left_out);
        }
        let elements = Lookup::of(&self.element_tags);
        let (np, ne) = (self.node_tags.len(), self.el.len());
        for view in std::mem::take(&mut self.views) {
            let comps = view.comps;
            let (lookup, of, slots) = match view.on {
                On::Nodes => (&nodes, np, np),
                On::Elements => (&elements, ne, ne),
                On::ElementNodes => (&elements, ne, self.el.conn.len()),
            };
            // Laid out over the nodes or elements, a view is at most so many times the values the file gave.
            if view.values.len().saturating_mul(COVERAGE) < slots.saturating_mul(comps) || comps > 9 {
                left_out.push(view.name);
                continue;
            }
            let mut values = vec![0.0; slots * comps];
            let mut from = 0;
            for (k, &tag) in view.tags.iter().enumerate() {
                let listed = view.nodes.get(k).map_or(1, |&n| n as usize);
                let record = &view.values[from..from + listed * comps];
                from += listed * comps;
                let at = lookup.get(tag) as usize;
                if at >= of {
                    continue;
                }
                let (first, room) = match view.on {
                    On::ElementNodes => {
                        let first = self.el.offsets[at] as usize;
                        (first, self.el.offsets[at + 1] as usize - first)
                    }
                    _ => (at, 1),
                };
                let n = listed.min(room) * comps;
                values[first * comps..first * comps + n].copy_from_slice(&record[..n]);
            }
            let array = Array::new(&view.name, comps, Role::Other, values);
            match view.on {
                On::Nodes => self.el.point_data.push(array),
                On::Elements => self.el.cell_data.push(array),
                On::ElementNodes => self.el.corner_data.push(array),
            }
        }
        (self.el, left_out)
    }
}
