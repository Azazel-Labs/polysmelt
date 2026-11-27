use crate::Document;

fn obj_doc() -> Document {
    crate::read(
        &b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\nf 1/1 2/2 4/4 3/3\n"[..],
        "t.obj",
    )
    .unwrap()
}

#[test]
fn writes_a_file_the_reader_reads_back() {
    let doc = obj_doc();
    let fbx = doc.to_fbx().unwrap();
    assert_eq!(fbx, doc.to_fbx().unwrap(), "the same document writes the same bytes");
    let back = crate::read(fbx.clone(), "t.fbx").unwrap();
    let mesh = back.flatten(crate::document::Pose::Rest).unwrap().unwrap();
    assert_eq!(mesh.triangles(), 2);
}

#[test]
fn fbxcel_reads_what_is_written() {
    let fbx = obj_doc().to_fbx().unwrap();
    let fbxcel::tree::any::AnyTree::V7400(version, tree, footer) =
        fbxcel::tree::any::AnyTree::from_seekable_reader(std::io::Cursor::new(fbx)).unwrap()
    else {
        panic!("not FBX 7.4")
    };
    assert_eq!(version, fbxcel::low::FbxVersion::V7_4);
    assert!(footer.is_ok(), "the footer is valid");
    let names: Vec<String> = tree.root().children().map(|n| n.name().to_owned()).collect();
    assert_eq!(
        names,
        [
            "FBXHeaderExtension",
            "FileId",
            "CreationTime",
            "Creator",
            "GlobalSettings",
            "Documents",
            "References",
            "Definitions",
            "Objects",
            "Connections",
            "Takes"
        ]
    );
}
