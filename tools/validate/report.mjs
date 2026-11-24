// Runs the Khronos glTF validator over every .gltf and .glb under a folder and writes a JSON report:
// for each file, its error and warning codes (with counts). Unlike validate.mjs it loads a model's
// external buffers and images: from beside it, or, when a `<model>.src` file names one, from that
// folder (a converted GLB that still points at its source's textures).
//
//   node tools/validate/report.mjs <folder> <report.json> [--max-mb 200]
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import validator from "gltf-validator";

const [root, out] = process.argv.slice(2);
const maxArg = process.argv.indexOf("--max-mb");
const maxBytes = (maxArg > 0 ? Number(process.argv[maxArg + 1]) : 200) * 1024 * 1024;
if (!root || !out) {
  console.error("usage: node report.mjs <folder> <report.json> [--max-mb 200]");
  process.exit(2);
}

function* models(dir) {
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* models(p);
    else if (/\.(gltf|glb)$/i.test(name)) yield p;
  }
}

const report = {};
let n = 0;
let withErrors = 0;
for (const path of models(root)) {
  const key = relative(root, path).split(sep).join("/");
  const size = statSync(path).size;
  if (size > maxBytes) {
    report[key] = { skipped: `${(size / 1048576).toFixed(0)} MB` };
    continue;
  }
  const side = `${path}.src`;
  const base = existsSync(side) ? resolve(readFileSync(side, "utf8").trim()) : dirname(resolve(path));
  let result;
  try {
    result = await validator.validateBytes(new Uint8Array(readFileSync(path)), {
      uri: path,
      maxIssues: 0,
      externalResourceFunction: (uri) => {
        const p = resolve(base, decodeURIComponent(uri));
        // Only files under the model's own folder.
        if (!p.startsWith(base + sep)) return Promise.reject(new Error(`outside: ${uri}`));
        try {
          return Promise.resolve(new Uint8Array(readFileSync(p)));
        } catch (e) {
          return Promise.reject(e);
        }
      },
    });
  } catch (e) {
    report[key] = { crash: String(e).slice(0, 300) };
    continue;
  }
  const errors = {};
  const warns = {};
  for (const m of result.issues.messages) {
    if (m.severity === 0) errors[m.code] = (errors[m.code] ?? 0) + 1;
    else if (m.severity === 1) warns[m.code] = (warns[m.code] ?? 0) + 1;
  }
  const count = Object.values(errors).reduce((a, b) => a + b, 0);
  const warnings = Object.values(warns).reduce((a, b) => a + b, 0);
  if (count > 0) withErrors++;
  n++;
  report[key] = { errors, error_count: count, warnings, warning_codes: warns };
  if (n % 200 === 0) console.log(`${n} validated`);
}
writeFileSync(out, JSON.stringify(report, null, 1));
console.log(`${n} files validated: ${withErrors} with errors`);
