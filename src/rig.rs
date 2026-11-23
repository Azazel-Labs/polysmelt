//! Typed views of a [`Document`](crate::document::Document)'s node tree, skins and
//! animations, for code that plays or retargets motion rather than drawing triangles:
//! [`Document::rig`](crate::document::Document::rig).
//!
//! Every format's reader converts into glTF, so an FBX armature, a `UsdSkel` skeleton and a
//! glTF skin all come out the same way: nodes with their rest transforms, skins naming their
//! joints and inverse bind matrices, and animations as keyed channels on nodes. Accessors are
//! read with every component type, normalized and sparse, as the flat mesh reads them.

/// A transform as translation, rotation (a unit quaternion, `[x, y, z, w]`) and scale, applied
/// scale first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trs {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Trs {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };
}

impl Default for Trs {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// One node of the document.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: Option<String>,
    /// The node that lists it as a child, if any. glTF doesn't order parents before children.
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Its own transform, relative to its parent. A node written as a matrix is decomposed.
    pub rest: Trs,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
}

/// The joints a mesh is skinned to.
#[derive(Debug, Clone, PartialEq)]
pub struct Skin {
    pub name: Option<String>,
    /// The joints' nodes.
    pub joints: Vec<usize>,
    /// Each joint's inverse bind matrix, column-major: identity for a skin that gives none.
    pub inverse_bind: Vec<[f32; 16]>,
    /// The node the skin names as its skeleton's root, if any.
    pub skeleton: Option<usize>,
}

/// What a channel animates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    Translation,
    Rotation,
    Scale,
    /// Morph target weights.
    Weights,
}

/// How a channel's values run between keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    /// Hermite splines: each key holds an in-tangent, a value and an out-tangent.
    CubicSpline,
}

/// One animated property of one node.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    pub node: usize,
    pub path: Path,
    pub interpolation: Interpolation,
    /// Key times in seconds, ascending.
    pub times: Vec<f32>,
    /// The keys' values, flat: each key's element is `values.len() / times.len()` wide (three
    /// elements per key, in-tangent, value and out-tangent, for a cubic spline).
    pub values: Vec<f32>,
}

impl Channel {
    /// The value at `t` seconds: the first key's before it, the last's after. Rotations come
    /// out unit length. `None` for a channel without keys or with values that don't fit them.
    #[must_use]
    pub fn sample(&self, t: f32) -> Option<Vec<f32>> {
        let interpolation = match self.interpolation {
            Interpolation::Linear => "LINEAR",
            Interpolation::Step => "STEP",
            Interpolation::CubicSpline => "CUBICSPLINE",
        };
        crate::flatten::sample(&self.times, &self.values, interpolation, t, self.path == Path::Rotation)
            .map(|v| v.into_iter().map(crate::num::f32_of).collect())
    }
}

/// An animation: channels keyed on nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Animation {
    pub name: Option<String>,
    pub channels: Vec<Channel>,
}

impl Animation {
    /// When its first key is, and its last: `(0, 0)` for one without keys.
    #[must_use]
    pub fn span(&self) -> (f32, f32) {
        let mut first = f32::INFINITY;
        let mut last = f32::NEG_INFINITY;
        for c in &self.channels {
            if let (Some(a), Some(b)) = (c.times.first(), c.times.last()) {
                first = first.min(*a);
                last = last.max(*b);
            }
        }
        if first <= last { (first, last) } else { (0.0, 0.0) }
    }

    /// Every key time any channel has, ascending, each once.
    #[must_use]
    pub fn key_times(&self) -> Vec<f32> {
        let mut t: Vec<f32> = self.channels.iter().flat_map(|c| c.times.iter().copied()).collect();
        t.sort_by(f32::total_cmp);
        t.dedup();
        t
    }
}

/// A document's node tree, skins and animations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rig {
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
}

impl Rig {
    /// The first node with this name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name.as_deref() == Some(name))
    }

    /// Every node's transform relative to its parent in animation `animation` at `time`
    /// seconds: its rest transform, with what the animation's channels set in place of it.
    /// `None` for the rest pose.
    ///
    /// # Panics
    ///
    /// If there's no animation `animation`.
    #[must_use]
    pub fn pose(&self, animation: Option<usize>, time: f32) -> Vec<Trs> {
        let mut out: Vec<Trs> = self.nodes.iter().map(|n| n.rest).collect();
        let Some(a) = animation else { return out };
        for c in &self.animations[a].channels {
            let (Some(o), Some(v)) = (out.get_mut(c.node), c.sample(time)) else {
                continue;
            };
            match (c.path, v.as_slice()) {
                (Path::Translation, &[x, y, z]) => o.translation = [x, y, z],
                (Path::Scale, &[x, y, z]) => o.scale = [x, y, z],
                (Path::Rotation, &[x, y, z, w]) => o.rotation = [x, y, z, w],
                _ => {}
            }
        }
        out
    }

    /// The nodes in an order where each comes after its parent, and so every joint after the
    /// joints above it; nodes in a cycle (which glTF forbids) are left out.
    #[must_use]
    pub fn parents_first(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<usize> = (0..self.nodes.len())
            .rev()
            .filter(|&i| self.nodes[i].parent.is_none())
            .collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            out.push(i);
            stack.extend(self.nodes[i].children.iter().rev().filter(|&&c| c < seen.len()));
        }
        out
    }
}
