//! [`Format`]: the formats Polysmelt reads, by name and extension.

/// A model format Polysmelt reads, whether or not this build has its cargo feature.
///
/// [`open`](crate::open) and [`read`](crate::read) tell a model's format from its extension and
/// then from its first bytes; [`Loader::format`](crate::Loader::format) names it outright.
///
/// ```
/// use polysmelt::Format;
/// assert_eq!(Format::from_extension("GLB"), Some(Format::Gltf));
/// assert_eq!(Format::detect(b"Kaydara FBX Binary  \0"), Some(Format::Fbx));
/// assert_eq!(Format::Collada.extensions(), ["dae", "zae", "kmz"]);
/// assert_eq!(Format::Collada.name(), "COLLADA");
/// ```
///
/// Each is behind a cargo feature of its name: `3ds` and `3mf` for those two, `dae` for
/// COLLADA, `xyz` for the text point clouds, `ac3d` for `.ac`, `mmd` for PMX, PMD and VMD,
/// `directx` for `.x`, `asf` for ASF and AMC, `irr` for Irrlicht's two, `md5` for MD5's two,
/// `smd` for SMD and VTA, `gltf` for glTF and VRM, `las` for LAS and LAZ (`laz` adds LAZ's
/// decompression), `vtk` for VTK's legacy and XML files.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Format {
    /// STL, binary and ASCII (`.stl`).
    Stl,
    /// PLY, ASCII and binary, with gaussian splats, plain or compressed (`.ply`).
    Ply,
    /// `MagicaVoxel` (`.vox`).
    Vox,
    /// Wavefront OBJ, with its MTL materials (`.obj`).
    Obj,
    /// FBX, binary and ASCII (`.fbx`).
    Fbx,
    /// 3D Studio (`.3ds`).
    ThreeDs,
    /// 3D Manufacturing Format (`.3mf`).
    ThreeMf,
    /// Universal Scene Description: text, binary and zipped (`.usd`, `.usda`, `.usdc`, `.usdz`).
    Usd,
    /// glTF 2.0 (and 1.0), JSON and binary (`.gltf`, `.glb`).
    Gltf,
    /// VRM avatars and their animations (`.vrm`, `.vrma`).
    Vrm,
    /// `SketchUp` (`.skp`).
    SketchUp,
    /// BVH motion capture (`.bvh`).
    Bvh,
    /// COLLADA, and the `.zae` and `.kmz` archives that hold it (`.dae`, `.zae`, `.kmz`).
    Collada,
    /// Object File Format, ASCII and binary (`.off`).
    Off,
    /// X3D and VRML97, and their gzipped forms (`.x3d`, `.x3dv`, `.x3dz`, `.wrl`, `.wrz`).
    X3d,
    /// DXF meshes (`.dxf`).
    Dxf,
    /// Gaussian splats in the 32-byte layout (`.splat`).
    Splat,
    /// Gaussian splats, gzipped (`.spz`).
    Spz,
    /// SOG gaussian splats: a zip of a `meta.json` and its WebP images, or that `meta.json`
    /// beside them (`.sog`).
    Sog,
    /// LAS and LAZ lidar point clouds (`.las`, `.laz`).
    Las,
    /// PCD point clouds (`.pcd`).
    Pcd,
    /// Text point clouds, a point per line (`.xyz`).
    Xyz,
    /// Text point clouds, a point per line (`.pts`).
    Pts,
    /// Text point clouds of scans, each placed by its pose (`.ptx`).
    Ptx,
    /// Text point clouds, a point per line, or ESRI ASCII grids (`.asc`).
    Asc,
    /// E57 3D scans (`.e57`).
    E57,
    /// Additive Manufacturing File Format (`.amf`).
    Amf,
    /// `LightWave` objects (`.lwo`).
    Lwo,
    /// AC3D (`.ac`).
    Ac3d,
    /// `MikuMikuDance` PMX models (`.pmx`).
    Pmx,
    /// `MikuMikuDance` PMD models (`.pmd`).
    Pmd,
    /// `MikuMikuDance` VMD motions (`.vmd`).
    Vmd,
    /// `DirectX` `.x`: text, binary and compressed (`.x`).
    DirectX,
    /// Quake II models (`.md2`).
    Md2,
    /// Quake III models (`.md3`).
    Md3,
    /// Doom 3 skeletal meshes (`.md5mesh`).
    Md5Mesh,
    /// Doom 3 skeletal animations (`.md5anim`).
    Md5Anim,
    /// Inter-Quake Model (`.iqm`).
    Iqm,
    /// Valve SMD reference and animation files (`.smd`).
    Smd,
    /// Valve VTA vertex animation (`.vta`).
    Vta,
    /// `MilkShape` 3D (`.ms3d`).
    Ms3d,
    /// `Blitz3D` (`.b3d`).
    B3d,
    /// Acclaim ASF skeletons (`.asf`).
    Asf,
    /// Acclaim AMC motions (`.amc`).
    Amc,
    /// C3D motion capture (`.c3d`).
    C3d,
    /// TRC marker motion capture (`.trc`).
    Trc,
    /// HTR motion capture (`.htr`).
    Htr,
    /// OGRE meshes and skeletons, binary and XML, with `.material` scripts (`.mesh`,
    /// `.skeleton`; `.mesh.xml` and `.skeleton.xml` are told from their root element).
    Ogre,
    /// `OpenFlight` (`.flt`).
    Flt,
    /// `OpenGEX` (`.ogex`).
    Ogex,
    /// Irrlicht meshes (`.irrmesh`).
    IrrMesh,
    /// Irrlicht scenes (`.irr`).
    Irr,
    /// XGL, and zipped ZGL (`.xgl`, `.zgl`).
    Xgl,
    /// VTK: legacy files, and XML poly data and unstructured grids (`.vtk`, `.vtp`, `.vtu`).
    Vtk,
}

/// Each format with its name and extensions, the first of them the one it's dispatched under.
const TABLE: &[(Format, &str, &[&str])] = &[
    (Format::Stl, "STL", &["stl"]),
    (Format::Ply, "PLY", &["ply"]),
    (Format::Vox, "MagicaVoxel", &["vox"]),
    (Format::Obj, "OBJ", &["obj"]),
    (Format::Fbx, "FBX", &["fbx"]),
    (Format::ThreeDs, "3DS", &["3ds"]),
    (Format::ThreeMf, "3MF", &["3mf"]),
    (Format::Usd, "USD", &["usd", "usda", "usdc", "usdz"]),
    (Format::Gltf, "glTF", &["gltf", "glb"]),
    (Format::Vrm, "VRM", &["vrm", "vrma"]),
    (Format::SketchUp, "SketchUp", &["skp"]),
    (Format::Bvh, "BVH", &["bvh"]),
    (Format::Collada, "COLLADA", &["dae", "zae", "kmz"]),
    (Format::Off, "OFF", &["off"]),
    (Format::X3d, "X3D", &["x3d", "x3dv", "x3dz", "wrl", "wrz"]),
    (Format::Dxf, "DXF", &["dxf"]),
    (Format::Splat, "splat", &["splat"]),
    (Format::Spz, "SPZ", &["spz"]),
    (Format::Sog, "SOG", &["sog"]),
    (Format::Las, "LAS", &["las", "laz"]),
    (Format::Pcd, "PCD", &["pcd"]),
    (Format::Xyz, "XYZ", &["xyz"]),
    (Format::Pts, "PTS", &["pts"]),
    (Format::Ptx, "PTX", &["ptx"]),
    (Format::Asc, "ASC", &["asc"]),
    (Format::E57, "E57", &["e57"]),
    (Format::Amf, "AMF", &["amf"]),
    (Format::Lwo, "LightWave", &["lwo"]),
    (Format::Ac3d, "AC3D", &["ac"]),
    (Format::Pmx, "PMX", &["pmx"]),
    (Format::Pmd, "PMD", &["pmd"]),
    (Format::Vmd, "VMD", &["vmd"]),
    (Format::DirectX, "DirectX", &["x"]),
    (Format::Md2, "MD2", &["md2"]),
    (Format::Md3, "MD3", &["md3"]),
    (Format::Md5Mesh, "MD5 mesh", &["md5mesh"]),
    (Format::Md5Anim, "MD5 animation", &["md5anim"]),
    (Format::Iqm, "IQM", &["iqm"]),
    (Format::Smd, "SMD", &["smd"]),
    (Format::Vta, "VTA", &["vta"]),
    (Format::Ms3d, "MilkShape 3D", &["ms3d"]),
    (Format::B3d, "Blitz3D", &["b3d"]),
    (Format::Asf, "ASF", &["asf"]),
    (Format::Amc, "AMC", &["amc"]),
    (Format::C3d, "C3D", &["c3d"]),
    (Format::Trc, "TRC", &["trc"]),
    (Format::Htr, "HTR", &["htr"]),
    (Format::Ogre, "OGRE", &["mesh", "skeleton"]),
    (Format::Flt, "OpenFlight", &["flt"]),
    (Format::Ogex, "OpenGEX", &["ogex"]),
    (Format::IrrMesh, "Irrlicht mesh", &["irrmesh"]),
    (Format::Irr, "Irrlicht scene", &["irr"]),
    (Format::Xgl, "XGL", &["xgl", "zgl"]),
    (Format::Vtk, "VTK", &["vtk", "vtp", "vtu"]),
];

impl Format {
    /// Every format, in the order they're listed here.
    pub fn all() -> impl ExactSizeIterator<Item = Format> + Clone {
        TABLE.iter().map(|&(f, ..)| f)
    }

    /// The format whose extension is `ext` (in any case, with or without its dot); `None` for
    /// one that names none, `xml` among them: an `.xml` file is told by its root element.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Format> {
        let ext = ext.strip_prefix('.').unwrap_or(ext);
        TABLE
            .iter()
            .find(|(.., exts)| exts.iter().any(|e| e.eq_ignore_ascii_case(ext)))
            .map(|&(f, ..)| f)
    }

    /// The format of a model whose bytes (all of them, or its start) are `bytes`, told from its
    /// signature or first words; `None` when they don't say. A binary STL, which has no
    /// signature, is told only from all of its bytes.
    #[must_use]
    pub fn detect(bytes: &[u8]) -> Option<Format> {
        Self::from_extension(super::detect(bytes)?)
    }

    /// The format's extensions, without their dots, the usual one first.
    #[must_use]
    pub fn extensions(self) -> &'static [&'static str] {
        self.entry().2
    }

    /// The format's name, as people write it: `"glTF"`, `"COLLADA"`, `"MagicaVoxel"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.entry().1
    }

    fn entry(self) -> &'static (Format, &'static str, &'static [&'static str]) {
        // Every variant has an entry: `every_format_has_an_entry` checks.
        TABLE.iter().find(|(f, ..)| *f == self).unwrap_or(&TABLE[0])
    }

    /// The extension a model of this format is read under, for one whose own extension is
    /// `ext`: `ext` itself when it's one of the format's, else the format's own (when `sniffed`,
    /// the extension its bytes told, isn't one of the format's either).
    pub(crate) fn extension_for(self, ext: &str, sniffed: impl FnOnce() -> Option<&'static str>) -> &'static str {
        let exts = self.extensions();
        exts.iter()
            .find(|e| e.eq_ignore_ascii_case(ext))
            .or_else(|| sniffed().and_then(|s| exts.iter().find(|e| **e == s)))
            .unwrap_or(&exts[0])
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_has_an_entry() {
        for (i, &(f, ..)) in TABLE.iter().enumerate() {
            assert_eq!(f.entry().0, f);
            assert!(TABLE[..i].iter().all(|(g, ..)| *g != f), "{f:?} listed twice");
        }
        let exts: Vec<&str> = Format::all().flat_map(Format::extensions).copied().collect();
        let mut all: Vec<&str> = super::super::EXTENSIONS
            .iter()
            .copied()
            .filter(|e| *e != "xml")
            .collect();
        let mut sorted = exts.clone();
        sorted.sort_unstable();
        all.sort_unstable();
        assert_eq!(sorted, all, "the formats' extensions are the ones read");
    }

    #[test]
    fn formats_from_extensions() {
        assert_eq!(Format::from_extension("Glb"), Some(Format::Gltf));
        assert_eq!(Format::from_extension(".usdz"), Some(Format::Usd));
        assert_eq!(Format::from_extension("xml"), None);
        assert_eq!(Format::from_extension(""), None);
        assert_eq!(Format::Usd.extension_for("USDZ", || None), "usdz");
        assert_eq!(Format::Usd.extension_for("bin", || Some("usdz")), "usdz");
        assert_eq!(Format::Usd.extension_for("bin", || Some("glb")), "usd");
    }
}
