// Validates every .glb under a folder with the Khronos glTF validator; exits non-zero when any
// has an error. The conformance tests write their GLBs to target/tmp/conformance:
//
//   cargo test --test conformance && node tools/validate/validate.mjs target/tmp/conformance
//
// A case that's invalid on purpose lists the error codes it's meant to raise under
// "validator_errors" in its tests/corpus/<format>/<case>/expect.json; only those are allowed.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import validator from "gltf-validator";

const root = process.argv[2] ?? "target/tmp/conformance";
const corpus = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "corpus");

/** The error codes a case's expect.json allows, for the GLB written from it. */
function allowed(path) {
  const expect = join(corpus, relative(root, path).replace(/(\.mesh)?\.glb$/, ""), "expect.json");
  if (!existsSync(expect)) return new Set();
  return new Set(JSON.parse(readFileSync(expect, "utf8")).validator_errors ?? []);
}

/**
 * Whether an error is the validator not knowing KHR_gaussian_splatting (2.0.0-dev.3.10 predates
 * it): the attribute names that extension defines, which carry its prefix, are all it refuses.
 */
function unknownSplatAttribute(m) {
  return m.code === "MESH_PRIMITIVE_INVALID_ATTRIBUTE" && /\/attributes\/KHR_gaussian_splatting:[A-Z0-9_]+$/.test(m.pointer ?? "");
}

function* glbs(dir) {
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* glbs(p);
    else if (name.endsWith(".glb")) yield p;
  }
}

let checked = 0;
let failed = 0;
let warned = 0;
for (const path of glbs(root)) {
  const report = await validator.validateBytes(new Uint8Array(readFileSync(path)), {
    uri: path,
    maxIssues: 0,
    // A GLB the tests write stands alone; an image whose file wasn't found keeps its URI.
    externalResourceFunction: (uri) => Promise.reject(new Error(`external resource ${uri}`)),
  });
  checked++;
  const issues = report.issues.messages;
  const ok = allowed(path);
  const errors = issues.filter(
    (m) => m.severity === 0 && m.code !== "IO_ERROR" && !ok.has(m.code) && !unknownSplatAttribute(m),
  );
  if (issues.some((m) => m.severity === 1)) warned++;
  if (errors.length > 0) {
    failed++;
    console.log(`${relative(root, path)}:`);
    for (const m of errors) console.log(`  ${m.code} ${m.pointer ?? ""}: ${m.message}`);
  }
}
console.log(`${checked} GLBs validated: ${failed} with errors, ${warned} with warnings`);
process.exit(failed > 0 || checked === 0 ? 1 : 0);
