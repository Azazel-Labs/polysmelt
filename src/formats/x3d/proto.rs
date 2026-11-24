//! Prototypes and their instances.
//!
//! An instance is replaced by a copy of its prototype's first body node, and of the nodes under
//! it. A field the body ties to the interface (`IS`) takes the instance's value, or else the
//! interface's default; one with neither is left out. Instances in the copy, and in the values
//! it was given, are expanded in turn, to a nesting depth and a node budget, so a prototype that
//! instances itself stops.
//!
//! An external prototype is read from the first of its URLs that can be read, through the
//! assets: `file.wrl#Name` picks the prototype named, a URL without a name the file's first.
//! It's copied into the scene's graph with the prototypes it uses from its file, and instanced
//! like one declared there.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::{Graph, MAX_INLINE, MAX_NODES, Node, Proto, Val, parse};
use crate::assets::Assets;

/// Instances inside instances expanded, at most.
const MAX_NESTING: usize = 32;

/// Files external prototypes may be read from, in all.
const MAX_FILES: usize = 64;

/// External prototype files read, by path: `None` for one that couldn't be read, or is being
/// read.
pub(super) type Library = HashMap<PathBuf, Option<Rc<Graph>>>;

/// Gives graph `g` the external prototypes it declares, each as a prototype of its own that can
/// be instanced; `dir` is its file's folder, `depth` how many files led to it.
pub(super) fn resolve(g: &mut Graph, assets: &dyn Assets, dir: &Path, lib: &mut Library, depth: usize) {
    for (name, urls) in std::mem::take(&mut g.externs) {
        let before = g.protos.len();
        for url in &urls {
            let (path, pick) = match url.split_once('#') {
                Some((path, pick)) => (path, Some(pick)),
                None => (url.as_str(), None),
            };
            if path.is_empty() {
                // One of this file's own.
                if let Some(i) = pick.and_then(|p| g.proto(p)) {
                    let p = g.protos[i].clone();
                    g.protos.push(Proto {
                        name: name.clone(),
                        ..p
                    });
                    break;
                }
                continue;
            }
            let Some(src) = library(assets, path, dir, lib, depth) else {
                continue;
            };
            let found = match pick {
                Some(p) => src.proto(p),
                None => (!src.protos.is_empty()).then_some(0),
            };
            if let Some(i) = found {
                import(g, &src, i, &name);
                break;
            }
        }
        if g.protos.len() == before {
            g.notes.insert("an external prototype wasn't found or read");
        }
    }
}

/// The graph of the file `url` names, its own external prototypes resolved.
fn library(assets: &dyn Assets, url: &str, dir: &Path, lib: &mut Library, depth: usize) -> Option<Rc<Graph>> {
    if depth >= MAX_INLINE {
        return None;
    }
    let path = assets.find(url, dir)?;
    if let Some(g) = lib.get(&path) {
        return g.clone();
    }
    if lib.len() >= MAX_FILES {
        return None;
    }
    lib.insert(path.clone(), None);
    let (mut g, _) = parse(&assets.read(&path)?).ok()?;
    let sub = path.parent().map(Path::to_path_buf).unwrap_or_default();
    resolve(&mut g, assets, &sub, lib, depth + 1);
    let g = Rc::new(g);
    lib.insert(path, Some(Rc::clone(&g)));
    Some(g)
}

/// Copies prototype `i` of `src` into `dst` as `name`, with the prototypes of `src` its nodes
/// are instances of that `dst` has none of.
fn import(dst: &mut Graph, src: &Graph, i: usize, name: &str) {
    let mut map = HashMap::new();
    let mut done = HashSet::new();
    let mut todo = vec![(i, name.to_owned())];
    while let Some((i, name)) = todo.pop() {
        if !done.insert(i) {
            continue;
        }
        let p = &src.protos[i];
        let mut roots = p.body.clone();
        for (_, v) in &p.fields {
            if let Some(Val::Nodes(list)) = v {
                roots.extend(list);
            }
        }
        for s in copy(dst, src, &roots, &mut map) {
            let kind = src.kind(s);
            if dst.proto(kind).is_none()
                && let Some(j) = src.proto(kind)
            {
                todo.push((j, kind.to_owned()));
            }
        }
        let fields = p
            .fields
            .iter()
            .map(|(n, v)| (n.clone(), v.as_ref().map(|v| remap(v, &map))))
            .collect();
        let body = p.body.iter().filter_map(|b| map.get(b).copied()).collect();
        dst.protos.push(Proto { name, fields, body });
    }
}

/// Copies the nodes of `src` under `roots` into `dst`, those already copied (in `map`) left as
/// they are; the nodes of `src` it copied.
fn copy(dst: &mut Graph, src: &Graph, roots: &[usize], map: &mut HashMap<usize, usize>) -> Vec<usize> {
    let mut stack = roots.to_vec();
    let mut new = Vec::new();
    while let Some(s) = stack.pop() {
        if map.contains_key(&s) || s >= src.nodes.len() {
            continue;
        }
        if dst.nodes.len() >= MAX_NODES {
            dst.notes.insert("prototype nodes past the node limit were skipped");
            break;
        }
        map.insert(s, dst.add("", None));
        new.push(s);
        stack.extend(children(&src.nodes[s]));
    }
    for &s in &new {
        let n = &src.nodes[s];
        let fields = n.fields.iter().map(|(k, v)| (k.clone(), remap(v, map))).collect();
        dst.nodes[map[&s]] = Node {
            kind: n.kind.clone(),
            name: n.name.clone(),
            fields,
        };
    }
    new
}

/// The nodes a node's fields hold.
fn children(n: &Node) -> impl Iterator<Item = usize> + '_ {
    n.fields
        .iter()
        .flat_map(|(_, v)| match v {
            Val::Nodes(list) => list.as_slice(),
            _ => &[],
        })
        .copied()
}

/// A value with its nodes renumbered by `map`, those not in it left out.
fn remap(v: &Val, map: &HashMap<usize, usize>) -> Val {
    match v {
        Val::Nodes(list) => Val::Nodes(list.iter().filter_map(|n| map.get(n).copied()).collect()),
        v => v.clone(),
    }
}

/// Expands the prototype instances in graph `g`'s scene, from its roots.
pub(super) fn expand(g: &mut Graph) {
    if g.protos.is_empty() {
        return;
    }
    let index: HashMap<String, usize> = g.protos.iter().enumerate().map(|(i, p)| (p.name.clone(), i)).collect();
    // The nodes a copy of each prototype makes.
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    let mut seen = Vec::new();
    let mut stack: Vec<(usize, usize)> = g.roots.iter().rev().map(|&r| (r, 0)).collect();
    while let Some((id, depth)) = stack.pop() {
        if id >= g.nodes.len() {
            continue;
        }
        if seen.len() < g.nodes.len() {
            seen.resize(g.nodes.len(), false);
        }
        if seen[id] {
            continue;
        }
        if let Some(&p) = index.get(&g.nodes[id].kind) {
            let size = *sizes
                .entry(p)
                .or_insert_with(|| reach(g, g.protos[p].body.first().copied()));
            if depth >= MAX_NESTING {
                g.notes.insert("prototype instances nested too deep were skipped");
            } else if g.nodes.len() + size > MAX_NODES {
                g.notes.insert("prototype instances past the node limit were skipped");
            } else if instantiate(g, id, p) {
                stack.push((id, depth + 1));
                continue;
            }
            seen[id] = true;
            continue;
        }
        seen[id] = true;
        let at = stack.len();
        stack.extend(children(&g.nodes[id]).map(|c| (c, depth)));
        stack[at..].reverse();
    }
}

/// How many nodes are under `root`, itself included.
fn reach(g: &Graph, root: Option<usize>) -> usize {
    let mut seen = HashSet::new();
    let mut stack: Vec<usize> = root.into_iter().collect();
    while let Some(n) = stack.pop() {
        if n < g.nodes.len() && seen.insert(n) {
            stack.extend(children(&g.nodes[n]));
        }
    }
    seen.len()
}

/// Makes node `id`, an instance of prototype `p`, a copy of the prototype's first body node;
/// whether it could be. A prototype with an empty body makes an empty group.
fn instantiate(g: &mut Graph, id: usize, p: usize) -> bool {
    let Some(&first) = g.protos[p].body.first() else {
        g.nodes[id] = Node {
            kind: "Group".into(),
            name: g.nodes[id].name.take(),
            fields: Vec::new(),
        };
        return true;
    };
    // The nodes to copy, the first becoming the instance and the rest new.
    let mut order = Vec::new();
    let mut map = HashMap::new();
    let mut stack = vec![first];
    while let Some(s) = stack.pop() {
        if s >= g.nodes.len() || map.contains_key(&s) {
            continue;
        }
        if s == id {
            // The instance is inside its own prototype's body.
            return false;
        }
        let new = if s == first {
            id
        } else {
            g.nodes.len() + order.len() - 1
        };
        map.insert(s, new);
        order.push(s);
        stack.extend(children(&g.nodes[s]));
    }
    let given = std::mem::take(&mut g.nodes[id].fields);
    let proto = &g.protos[p];
    let value = |name: &str| {
        given
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .or_else(|| {
                proto
                    .fields
                    .iter()
                    .find(|(n, _)| n == name)
                    .and_then(|(_, v)| v.clone())
            })
    };
    let copies: Vec<Node> = order
        .iter()
        .map(|&s| {
            let n = &g.nodes[s];
            let fields = n
                .fields
                .iter()
                .filter_map(|(k, v)| {
                    let v = match v {
                        Val::Is(f) => value(f)?,
                        v => remap(v, &map),
                    };
                    Some((k.clone(), v))
                })
                .collect();
            Node {
                kind: n.kind.clone(),
                name: n.name.clone(),
                fields,
            }
        })
        .collect();
    let mut copies = copies.into_iter();
    if let Some(mut head) = copies.next() {
        if let Some(name) = g.nodes[id].name.take() {
            head.name = Some(name);
        }
        g.nodes[id] = head;
    }
    g.nodes.extend(copies);
    true
}
