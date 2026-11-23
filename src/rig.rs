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

impl Interpolation {
    /// Its glTF name.
    const fn name(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::Step => "STEP",
            Self::CubicSpline => "CUBICSPLINE",
        }
    }
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
        crate::flatten::sample(
            &self.times,
            &self.values,
            self.interpolation.name(),
            t,
            self.path == Path::Rotation,
        )
        .map(|v| v.into_iter().map(crate::num::f32_of).collect())
    }

    /// [`Channel::sample`] for a channel at most four wide, without allocating, given how many of its keys are at or
    /// before `t`.
    fn sample_small(&self, t: f32, k: usize) -> Option<([f32; 4], usize)> {
        let interpolation = self.interpolation.name();
        let w = crate::flatten::width(&self.times, &self.values, interpolation)?;
        let mut v = [0.0f64; 4];
        let rotation = self.path == Path::Rotation;
        if w > 4 || !crate::flatten::sample_after(&self.times, &self.values, interpolation, t, rotation, k, &mut v[..w])
        {
            return None;
        }
        Some((v.map(crate::num::f32_of), w))
    }

    /// Sets what this channel animates of `o` to its value at `t`, given how many of its keys are at or before it.
    fn pose(&self, t: f32, k: usize, o: &mut Trs) {
        if self.path == Path::Weights {
            return;
        }
        let Some((v, w)) = self.sample_small(t, k) else {
            return;
        };
        match (self.path, w) {
            (Path::Translation, 3) => o.translation = [v[0], v[1], v[2]],
            (Path::Scale, 3) => o.scale = [v[0], v[1], v[2]],
            (Path::Rotation, 4) => o.rotation = v,
            _ => {}
        }
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
        // Channels mostly share their times, every bone keyed on the same frames: each set is gathered once.
        let mut sets: Vec<&[f32]> = Vec::new();
        for c in &self.channels {
            if !sets.contains(&c.times.as_slice()) {
                sets.push(&c.times);
            }
        }
        let mut t: Vec<f32> = sets.concat();
        if sets.len() > 1 || !t.is_sorted_by(|a, b| a.total_cmp(b).is_le()) {
            t.sort_unstable_by(f32::total_cmp);
        }
        t.dedup();
        t
    }
}

/// Poses one animation at time after time, as [`Rig::pose_into`] does, each channel keeping its place in its keys:
/// times in order, as when a clip is played or baked, step to their keys rather than searching for them. Times in any
/// order pose the same, just no faster.
#[derive(Debug, Clone)]
pub struct Poser<'a> {
    rig: &'a Rig,
    channels: &'a [Channel],
    /// For each channel, how many of its keys were at or before the last time posed.
    keys: Vec<usize>,
}

impl Poser<'_> {
    /// Every node's transform relative to its parent at `time` seconds, into `out`, which is cleared first.
    pub fn pose_into(&mut self, time: f32, out: &mut Vec<Trs>) {
        out.clear();
        out.extend(self.rig.nodes.iter().map(|n| n.rest));
        for (c, k) in self.channels.iter().zip(&mut self.keys) {
            let times = &c.times;
            if *k > 0 && times.get(*k - 1).is_none_or(|&x| x > time) {
                *k = times.partition_point(|&x| x <= time);
            } else {
                // Forward a few keys at most at a time in order; a search if it's far ahead.
                let mut steps = 0;
                while times.get(*k).is_some_and(|&x| x <= time) {
                    *k += 1;
                    steps += 1;
                    if steps == 4 {
                        *k += times[*k..].partition_point(|&x| x <= time);
                        break;
                    }
                }
            }
            if let Some(o) = out.get_mut(c.node) {
                c.pose(time, *k, o);
            }
        }
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
        let mut out = Vec::with_capacity(self.nodes.len());
        self.pose_into(animation, time, &mut out);
        out
    }

    /// [`Rig::pose`] into `out`, which is cleared first: posing frame after frame into the same
    /// vector allocates nothing after the first.
    ///
    /// # Panics
    ///
    /// If there's no animation `animation`.
    pub fn pose_into(&self, animation: Option<usize>, time: f32, out: &mut Vec<Trs>) {
        out.clear();
        out.extend(self.nodes.iter().map(|n| n.rest));
        let Some(a) = animation else { return };
        for c in &self.animations[a].channels {
            if let Some(o) = out.get_mut(c.node) {
                c.pose(time, c.times.partition_point(|&x| x <= time), o);
            }
        }
    }

    /// A [`Poser`] for animation `animation`, to pose it at one time after another.
    ///
    /// # Panics
    ///
    /// If there's no animation `animation`.
    #[must_use]
    pub fn poser(&self, animation: usize) -> Poser<'_> {
        let channels = &self.animations[animation].channels;
        Poser {
            rig: self,
            channels,
            keys: vec![0; channels.len()],
        }
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

#[cfg(test)]
mod tests {
    use super::{Animation, Channel, Interpolation, Node, Path, Rig, Trs};

    fn rig() -> Rig {
        let node = |name: &str, parent| Node {
            name: Some(name.into()),
            parent,
            children: Vec::new(),
            rest: Trs::IDENTITY,
            mesh: None,
            skin: None,
        };
        let channel = |node, path, interpolation, times: &[f32], values: &[f32]| Channel {
            node,
            path,
            interpolation,
            times: times.to_vec(),
            values: values.to_vec(),
        };
        Rig {
            nodes: vec![node("root", None), node("arm", Some(0))],
            skins: Vec::new(),
            animations: vec![Animation {
                name: None,
                channels: vec![
                    // Written twice too long at its first key.
                    channel(
                        0,
                        Path::Rotation,
                        Interpolation::Linear,
                        &[0.0, 1.0],
                        &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                    ),
                    channel(
                        0,
                        Path::Translation,
                        Interpolation::Step,
                        &[0.0, 1.0],
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ),
                    channel(
                        1,
                        Path::Scale,
                        Interpolation::CubicSpline,
                        &[0.0, 1.0],
                        &[
                            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0,
                        ],
                    ),
                    channel(
                        1,
                        Path::Weights,
                        Interpolation::Linear,
                        &[0.0],
                        &[0.5, 0.5, 0.5, 0.5, 0.5],
                    ),
                ],
            }],
        }
    }

    #[test]
    fn pose_into_reuses_its_vector_and_poses_as_pose_does() {
        let rig = rig();
        let mut out = Vec::new();
        for t in [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0] {
            rig.pose_into(Some(0), t, &mut out);
            assert_eq!(out, rig.pose(Some(0), t), "at {t}");
        }
        let at = out.as_ptr();
        rig.pose_into(Some(0), 0.5, &mut out);
        assert_eq!(out.as_ptr(), at, "allocated again");
        rig.pose_into(None, 0.5, &mut out);
        assert_eq!(out, vec![Trs::IDENTITY; 2]);
    }

    #[test]
    fn key_times_holds_every_channels_keys_once_in_order() {
        let mut rig = rig();
        assert_eq!(rig.animations[0].key_times(), [0.0, 1.0]);
        let channel = rig.animations[0].channels[0].clone();
        for times in [vec![0.5, 1.0, 3.0], vec![0.0, 1.0], vec![-1.0, 0.25]] {
            let values = channel.values[..4].repeat(times.len());
            rig.animations[0].channels.push(Channel {
                times,
                values,
                ..channel.clone()
            });
        }
        assert_eq!(rig.animations[0].key_times(), [-1.0, 0.0, 0.25, 0.5, 1.0, 3.0]);
        assert_eq!(
            Animation {
                name: None,
                channels: Vec::new()
            }
            .key_times(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn a_poser_poses_as_pose_does_at_times_in_any_order() {
        let mut rig = rig();
        // A long channel, for steps of one key, several and many.
        let times: Vec<f32> = (0..40u8).map(|k| f32::from(k) * 0.05).collect();
        let values: Vec<f32> = (0..40u8).flat_map(|k| [f32::from(k), 0.0, -f32::from(k)]).collect();
        rig.animations[0].channels.push(Channel {
            node: 1,
            path: Path::Translation,
            interpolation: Interpolation::Linear,
            times,
            values,
        });
        let mut poser = rig.poser(0);
        let mut out = Vec::new();
        let mut check = |t: f32| {
            poser.pose_into(t, &mut out);
            assert_eq!(out, rig.pose(Some(0), t), "at {t}");
        };
        let forward = (-2..50i8).map(|k| f32::from(k) * 0.05);
        let jumps = [0.0, 0.01, 0.9, 0.9, 1.95, 2.5, 0.3, -1.0, 1.0, 0.05, 0.0];
        let backward = (-2..50i8).rev().map(|k| f32::from(k) * 0.037);
        for t in forward.chain(jumps).chain(backward) {
            check(t);
        }
    }

    #[test]
    fn rotations_come_out_unit_length_at_the_first_and_last_keys_too() {
        let rig = rig();
        for t in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            let q = rig.pose(Some(0), t)[0].rotation;
            let len = q.iter().map(|c| c * c).sum::<f32>().sqrt();
            assert!((len - 1.0).abs() < 1e-6, "{q:?} at {t}");
        }
        assert_eq!(rig.pose(Some(0), 0.0)[0].rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(rig.pose(Some(0), 0.5)[0].translation, [1.0, 2.0, 3.0]);
        assert_eq!(rig.pose(Some(0), 0.5)[1].scale, [2.0, 2.0, 2.0]);
    }
}
