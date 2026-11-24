// The evidence binaries' window on assimp: read a model (from its file, or from bytes in memory
// with the files it refers to found beside its path), count what the scene holds, and write the
// scene as a GLB (the `glb2` exporter, which `assimp export x.glb` uses). No C++ exception
// crosses into Rust: each entry point catches everything.

#include <assimp/DefaultIOSystem.h>
#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/MemoryIOWrapper.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

namespace {

void set_err(char* err, size_t len, const char* msg) {
    if (err && len) {
        std::snprintf(err, len, "%s", msg ? msg : "");
    }
}

std::string normalized(const char* p) {
    std::string s(p ? p : "");
    for (auto& c : s) {
        if (c == '\\') c = '/';
    }
    return s;
}

// The file system, except that the model itself is served from memory. (DefaultIOSystem is final,
// so this wraps one rather than deriving from it.)
class Overlay : public Assimp::IOSystem {
public:
    Overlay(const char* path, const uint8_t* data, size_t len) : main_(normalized(path)), data_(data), len_(len) {}
    bool Exists(const char* file) const override {
        if (normalized(file) == main_) return true;
        return disk_.Exists(file);
    }
    char getOsSeparator() const override { return disk_.getOsSeparator(); }
    Assimp::IOStream* Open(const char* file, const char* mode) override {
        if (normalized(file) == main_) return new Assimp::MemoryIOStream(data_, len_, false);
        return disk_.Open(file, mode);
    }
    void Close(Assimp::IOStream* f) override { delete f; }

private:
    mutable Assimp::DefaultIOSystem disk_;
    std::string main_;
    const uint8_t* data_;
    size_t len_;
};

uint64_t count_nodes(const aiNode* n, int depth) {
    if (!n || depth > 100000) return 0;
    uint64_t k = 1;
    for (unsigned i = 0; i < n->mNumChildren; i++) k += count_nodes(n->mChildren[i], depth + 1);
    return k;
}

}  // namespace

extern "C" {

struct ShimStats {
    uint64_t meshes, vertices, triangles, lines, points, polygons, materials, nodes, joints, animations, channels,
        textures, cameras, lights, incomplete;
};

// A new importer holding the scene read from `path` (from `data` when not null), or null with
// the reason in `err`.
void* shim_read(const char* path, const uint8_t* data, size_t len, unsigned flags, char* err, size_t errlen) {
    Assimp::Importer* imp = nullptr;
    try {
        imp = new Assimp::Importer();
        if (data) imp->SetIOHandler(new Overlay(path, data, len));
        const aiScene* s = imp->ReadFile(path, flags);
        if (!s) {
            set_err(err, errlen, imp->GetErrorString());
            delete imp;
            return nullptr;
        }
        return imp;
    } catch (const std::exception& e) {
        set_err(err, errlen, e.what());
    } catch (...) {
        set_err(err, errlen, "unknown C++ exception");
    }
    delete imp;
    return nullptr;
}

int shim_stats(void* h, ShimStats* out) {
    try {
        std::memset(out, 0, sizeof(*out));
        const aiScene* s = static_cast<Assimp::Importer*>(h)->GetScene();
        if (!s) return 1;
        out->meshes = s->mNumMeshes;
        std::set<std::string> bones;
        for (unsigned m = 0; m < s->mNumMeshes; m++) {
            const aiMesh* mesh = s->mMeshes[m];
            if (!mesh) continue;
            out->vertices += mesh->mNumVertices;
            for (unsigned f = 0; f < mesh->mNumFaces; f++) {
                unsigned n = mesh->mFaces[f].mNumIndices;
                if (n == 3) out->triangles++;
                else if (n == 2) out->lines++;
                else if (n == 1) out->points++;
                else if (n > 3) out->polygons++;
            }
            for (unsigned b = 0; b < mesh->mNumBones; b++) {
                if (mesh->mBones[b]) bones.insert(std::string(mesh->mBones[b]->mName.C_Str()));
            }
        }
        out->joints = bones.size();
        out->materials = s->mNumMaterials;
        out->nodes = count_nodes(s->mRootNode, 0);
        out->animations = s->mNumAnimations;
        for (unsigned a = 0; a < s->mNumAnimations; a++) {
            if (s->mAnimations[a]) out->channels += s->mAnimations[a]->mNumChannels + s->mAnimations[a]->mNumMorphMeshChannels;
        }
        out->textures = s->mNumTextures;
        out->cameras = s->mNumCameras;
        out->lights = s->mNumLights;
        out->incomplete = (s->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ? 1 : 0;
        return 0;
    } catch (...) {
        return 2;
    }
}

void shim_free(void* h) {
    try {
        delete static_cast<Assimp::Importer*>(h);
    } catch (...) {
    }
}

// The scene as a GLB: written to `out_path`, or (null) to a blob in memory and dropped. The
// GLB's size, or -1 with the reason in `err`.
int64_t shim_export_glb(void* h, const char* out_path, char* err, size_t errlen) {
    try {
        const aiScene* s = static_cast<Assimp::Importer*>(h)->GetScene();
        if (!s) {
            set_err(err, errlen, "no scene");
            return -1;
        }
        Assimp::Exporter exp;
        if (out_path) {
            if (exp.Export(s, "glb2", out_path, 0) != AI_SUCCESS) {
                set_err(err, errlen, exp.GetErrorString());
                return -1;
            }
            FILE* f = std::fopen(out_path, "rb");
            if (!f) return 0;
            std::fseek(f, 0, SEEK_END);
            long n = std::ftell(f);
            std::fclose(f);
            return n;
        }
        const aiExportDataBlob* blob = exp.ExportToBlob(s, "glb2", 0);
        if (!blob) {
            set_err(err, errlen, exp.GetErrorString());
            return -1;
        }
        int64_t n = 0;
        for (const aiExportDataBlob* b = blob; b; b = b->next) n += static_cast<int64_t>(b->size);
        return n;
    } catch (const std::exception& e) {
        set_err(err, errlen, e.what());
    } catch (...) {
        set_err(err, errlen, "unknown C++ exception");
    }
    return -1;
}

// Every extension assimp reads, as "*.3ds;*.obj;...".
const char* shim_extensions() {
    static std::string list;
    try {
        if (list.empty()) {
            Assimp::Importer imp;
            imp.GetExtensionList(list);
        }
    } catch (...) {
    }
    return list.c_str();
}

const char* shim_version() {
    static char buf[64];
    std::snprintf(buf, sizeof buf, "%u.%u.%u (rev %x)", aiGetVersionMajor(), aiGetVersionMinor(), aiGetVersionPatch(),
                  aiGetVersionRevision());
    return buf;
}

}  // extern "C"
