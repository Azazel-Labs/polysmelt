//! A finite-element mesh as its file lists it, and its elements sorted into the groups the file names.
//!
//! Gmsh's physical groups and Medit's references each tag elements. A group draws as a mesh of its own, named
//! after it, so a volume group shows its own boundary and can be hidden apart from the rest. Elements in no
//! group are one mesh with no name. Each group's piece holds only the points its elements use, in the file's
//! order, with the data that goes with them.

use serde_json::Value;

use super::{Array, BAD, Cells, Kind, Piece, Points, Role};
use crate::num::u32_of;

/// The most groups drawn as meshes of their own. A file with more is one mesh, each element's tag in a custom
/// attribute instead.
const MAX_GROUPS: usize = 1024;

/// A group of elements.
pub(in crate::formats) struct Group {
    /// The mesh's and the node's name.
    pub name: String,
    /// The mesh's `extras`.
    pub extras: Value,
    /// The number the file knows the group by.
    pub tag: i64,
}

/// A mesh's nodes and elements, and the data on them.
#[derive(Default)]
pub(in crate::formats) struct Elements {
    pub points: Vec<[f32; 3]>,
    /// Each element's VTK cell type.
    pub types: Vec<u8>,
    /// Element `i`'s nodes are `conn[offsets[i]..offsets[i + 1]]`: indices into the points, [`BAD`] for none.
    pub offsets: Vec<u32>,
    pub conn: Vec<u32>,
    /// Each element's group, an index into `groups`, or [`BAD`] for none.
    pub group: Vec<u32>,
    pub groups: Vec<Group>,
    pub point_data: Vec<Array>,
    /// A value for each element.
    pub cell_data: Vec<Array>,
    /// A value for each node of each element, in `conn`'s order.
    pub corner_data: Vec<Array>,
    /// The attribute the groups' tags go to when there are too many groups to draw apart.
    pub tags_as: &'static str,
}

impl Elements {
    /// Nothing yet, with the offsets' leading zero.
    pub fn new(tags_as: &'static str) -> Self {
        Self {
            offsets: vec![0],
            tags_as,
            ..Self::default()
        }
    }

    /// How many elements there are.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Adds an element of VTK type `ty` in `group`; its nodes are what was pushed to `conn` since the last.
    #[inline]
    pub fn close(&mut self, ty: u8, group: u32) {
        self.types.push(ty);
        self.group.push(group);
        self.offsets.push(u32_of(self.conn.len()));
    }

    /// The pieces to draw: one for the elements in no group, and one for each group that has elements.
    pub fn pieces(mut self) -> Vec<Piece> {
        let n = self.len();
        let mut counts = vec![0u32; self.groups.len() + 1];
        for &g in &self.group {
            counts[group_slot(g, self.groups.len())] += 1;
        }
        let used = counts.iter().filter(|&&c| c > 0).count();
        if used > MAX_GROUPS {
            // Too many to draw apart: the tags are data on the elements.
            #[allow(
                clippy::cast_precision_loss,
                reason = "tags past 2^24 round, as other whole numbers do"
            )]
            let tags = self
                .group
                .iter()
                .map(|&g| self.groups.get(g as usize).map_or(0.0, |g| g.tag as f32))
                .collect();
            self.cell_data.insert(0, Array::new(self.tags_as, 1, Role::Other, tags));
        }
        if used <= 1 || used > MAX_GROUPS {
            let only = match self.group.first() {
                Some(&g) if used == 1 => self.groups.get_mut(g as usize),
                _ => None,
            };
            let (name, extras) = match only {
                Some(g) => (Some(std::mem::take(&mut g.name)), Some(g.extras.take())),
                None => (None, None),
            };
            return vec![Piece {
                points: Points::List(self.points),
                dims: None,
                cells: vec![Cells {
                    kind: Kind::Mixed,
                    conn: self.conn,
                    offsets: self.offsets,
                    types: self.types,
                }],
                point_data: self.point_data,
                cell_data: self.cell_data,
                corner_data: self.corner_data,
                name,
                extras,
            }];
        }
        // The elements in each group's order: those in no group first.
        let mut starts = Vec::with_capacity(counts.len() + 1);
        let mut total = 0u32;
        for &c in &counts {
            starts.push(total);
            total += c;
        }
        starts.push(total);
        let mut next = starts.clone();
        let mut order = vec![0u32; n];
        for (e, &g) in self.group.iter().enumerate() {
            let slot = group_slot(g, self.groups.len());
            order[next[slot] as usize] = u32_of(e);
            next[slot] += 1;
        }
        let np = self.points.len();
        // Each point's place among a group's points, while that group is gathered.
        let mut local = vec![BAD; np];
        let mut pieces = Vec::with_capacity(used);
        for slot in 0..counts.len() {
            let elements = &order[starts[slot] as usize..starts[slot + 1] as usize];
            if elements.is_empty() {
                continue;
            }
            let nodes = |e: u32| self.offsets[e as usize] as usize..self.offsets[e as usize + 1] as usize;
            let mut points: Vec<u32> = Vec::new();
            let mut slots = 0;
            for &e in elements {
                slots += nodes(e).len();
                for &p in &self.conn[nodes(e)] {
                    if let Some(l) = local.get_mut(p as usize)
                        && *l == BAD
                    {
                        *l = 0;
                        points.push(p);
                    }
                }
            }
            points.sort_unstable();
            for (k, &p) in points.iter().enumerate() {
                local[p as usize] = u32_of(k);
            }
            let mut conn = Vec::with_capacity(slots);
            let mut offsets = Vec::with_capacity(elements.len() + 1);
            offsets.push(0);
            for &e in elements {
                conn.extend(
                    self.conn[nodes(e)]
                        .iter()
                        .map(|&p| local.get(p as usize).copied().unwrap_or(BAD)),
                );
                offsets.push(u32_of(conn.len()));
            }
            // Arrays that don't cover what they're data on are left behind.
            let gather = |arrays: &[Array], of: usize, from: &mut dyn Iterator<Item = usize>| -> Vec<Array> {
                let from: Vec<usize> = from.collect();
                let count = from.len();
                arrays
                    .iter()
                    .filter(|a| a.comps > 0 && of.checked_mul(a.comps) == Some(a.values.len()))
                    .map(|a| {
                        let mut values = Vec::with_capacity(count * a.comps);
                        for &i in &from {
                            values.extend_from_slice(&a.values[i * a.comps..][..a.comps]);
                        }
                        Array::new(&a.name, a.comps, a.role, values)
                    })
                    .collect()
            };
            let point_data = gather(&self.point_data, np, &mut points.iter().map(|&p| p as usize));
            let cell_data = gather(&self.cell_data, n, &mut elements.iter().map(|&e| e as usize));
            let corner_data = gather(
                &self.corner_data,
                self.conn.len(),
                &mut elements.iter().flat_map(|&e| nodes(e)),
            );
            let group = slot.checked_sub(1).map(|g| &mut self.groups[g]);
            let (name, extras) = match group {
                Some(g) => (Some(std::mem::take(&mut g.name)), Some(g.extras.take())),
                None => (None, None),
            };
            pieces.push(Piece {
                points: Points::List(points.iter().map(|&p| self.points[p as usize]).collect()),
                dims: None,
                cells: vec![Cells {
                    kind: Kind::Mixed,
                    conn,
                    offsets,
                    types: elements.iter().map(|&e| self.types[e as usize]).collect(),
                }],
                point_data,
                cell_data,
                corner_data,
                name,
                extras,
            });
            for &p in &points {
                local[p as usize] = BAD;
            }
        }
        pieces
    }
}

/// Where group `g` of `groups` is counted: 0 for none, then each group in turn.
fn group_slot(g: u32, groups: usize) -> usize {
    if (g as usize) < groups { g as usize + 1 } else { 0 }
}
