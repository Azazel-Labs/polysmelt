// The page around the viewer: files in (drop, pick, folders, samples), pointer input, the frame loop and the panel.

import init, { Viewer } from "./pkg/polysmelt_demo.js";

// Samples from the crate's test corpus, copied beside the page by scripts/build.sh (which lists their files in
// samples/index.json).
const SAMPLES = [
  { dir: "gltf/skin_animated", main: "skin.glb", label: "Skinned glTF", note: "bones on the GPU" },
  { dir: "fbx/blender_armature_animated", main: "armature_animated.fbx", label: "FBX armature", note: "Blender export" },
  { dir: "fbx/blender_shape_key_animated", main: "shape_key_animated.fbx", label: "FBX shape keys", note: "morph animation" },
  { dir: "md2/morph_frame_groups", main: "quad.md2", label: "Quake II MD2", note: "vertex frames, skin" },
  { dir: "bvh/humanoid", main: "humanoid.bvh", label: "BVH mocap", note: "a skeleton moving" },
  { dir: "dae/skin_twenty_influences", main: "skinned.dae", label: "COLLADA", note: "20 bones a vertex" },
  { dir: "obj/blender_suzanne_smooth", main: "suzanne_smooth.obj", label: "Wavefront OBJ", note: "with its MTL" },
  { dir: "pcd/pcl_car6_compressed", main: "car6.pcd", label: "PCL point cloud", note: "compressed PCD" },
];

// What polysmelt reads, by extension (formats::EXTENSIONS), and which of those this build can't.
const EXTENSIONS = "stl ply vox obj fbx 3ds 3mf usd usda usdc usdz gltf glb skp bvh dae zae kmz off vrm vrma x3d x3dv x3dz wrl wrz dxf splat spz sog las laz pcd xyz pts ptx asc e57 amf lwo ac pmx pmd vmd x md2 md3 md5mesh md5anim iqm smd vta ms3d b3d asf amc c3d trc htr mesh skeleton xml flt ogex vtk vtp vtu b3dm i3dm pnts cmpt".split(" ");
const UNSUPPORTED = new Set(["usd", "usda", "usdc", "usdz"]);
// Folders not worth reading when a big tree is dropped.
const SKIP_DIRS = new Set([".git", "node_modules", "__MACOSX", ".svn"]);
const MAX_FILES = 4000;

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const stage = $("stage");
let viewer = null;
let module = null;
let info = null;
let busy = false;
let scrubbing = false;

// Startup

async function start() {
  renderFormats();
  renderSamples();
  try {
    module = await init();
    viewer = await Viewer.create(canvas);
  } catch (e) {
    console.error(e);
    showError(
      "This browser can't run the viewer: it needs WebGPU or WebGL 2.\n" + String(e?.message ?? e),
    );
    return;
  }
  $("backend").textContent = viewer.backend();
  fitCanvas();
  new ResizeObserver(fitCanvas).observe(stage);
  requestAnimationFrame(loop);
  wireInput();
  reportCrash();
  const sample = new URLSearchParams(location.search).get("sample");
  if (sample) {
    const s = SAMPLES.find((s) => s.main === sample || s.dir === sample);
    if (s) loadSample(s);
  }
}

function loop(now) {
  if (viewer && !busy) {
    try {
      viewer.frame(now);
      if (info && info.animations.length && !scrubbing) showTime(viewer.time());
    } catch (e) {
      crashed(e);
    }
  }
  requestAnimationFrame(loop);
}

function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(stage.clientWidth * dpr));
  const h = Math.max(1, Math.round(stage.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  viewer?.resize(w, h);
}

// A panic in the module leaves it unusable: start the page afresh, once, saying why. A second crash soon after
// (one at startup, say) stays on the page with the message instead of reloading over and over.
const CRASH_KEY = "polysmelt-crash";
const CRASH_MESSAGE = "polysmelt hit a bug reading this file. Please report it on GitHub, with the file if you can.";
function crashed(e) {
  console.error(e);
  if (!(e instanceof WebAssembly.RuntimeError) || viewer === null) return;
  viewer = null;
  info = null;
  resetPanel();
  let last = 0;
  try {
    last = Number(sessionStorage.getItem(CRASH_KEY)) || 0;
    sessionStorage.setItem(CRASH_KEY, String(Date.now()));
  } catch {}
  if (Date.now() - last < 15000) {
    showError(`${CRASH_MESSAGE} (${e.message})`);
    return;
  }
  location.reload();
}

// Says why the page started afresh, when it did after a crash.
function reportCrash() {
  try {
    if (Number(sessionStorage.getItem(CRASH_KEY)) > Date.now() - 15000) showError(CRASH_MESSAGE);
  } catch {}
}

// Files in

function wireInput() {
  $("files").addEventListener("change", (e) => {
    const files = [...e.target.files].map((f) => ({ path: f.name, file: f }));
    e.target.value = "";
    loadFiles(files);
  });
  $("folder").addEventListener("change", (e) => {
    const files = [...e.target.files].map((f) => ({ path: f.webkitRelativePath || f.name, file: f }));
    e.target.value = "";
    loadFiles(files);
  });
  let depth = 0;
  stage.addEventListener("dragenter", (e) => {
    e.preventDefault();
    depth++;
    stage.classList.add("dragging");
  });
  stage.addEventListener("dragover", (e) => e.preventDefault());
  stage.addEventListener("dragleave", () => {
    if (--depth <= 0) {
      depth = 0;
      stage.classList.remove("dragging");
    }
  });
  stage.addEventListener("drop", async (e) => {
    e.preventDefault();
    depth = 0;
    stage.classList.remove("dragging");
    const files = await droppedFiles(e.dataTransfer);
    loadFiles(files);
  });
  // A drop outside the stage would open the file in the tab.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("download").addEventListener("click", download);
  $("reset").addEventListener("click", () => viewer?.reset_view());
  $("new").addEventListener("click", () => $("files").click());
  toggle($("floor"), (on) => viewer?.set_floor(on));
  toggle($("skeleton"), (on) => viewer?.set_skeleton(on));
  wirePlayer();
  wirePointer();
}

function toggle(button, set) {
  button.addEventListener("click", () => {
    const on = !button.classList.contains("on");
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", String(on));
    set(on);
  });
}

// Every file of a drop, folders walked, with paths relative to what was dropped.
async function droppedFiles(transfer) {
  const entries = [...transfer.items]
    .filter((i) => i.kind === "file")
    .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  if (!entries.length || entries.some((e) => !e)) {
    return [...transfer.files].map((f) => ({ path: f.name, file: f }));
  }
  const out = [];
  const walk = async (entry) => {
    if (out.length >= MAX_FILES) return;
    if (entry.isFile) {
      const file = await new Promise((ok, fail) => entry.file(ok, fail));
      out.push({ path: entry.fullPath.replace(/^\/+/, ""), file });
    } else if (entry.isDirectory && !SKIP_DIRS.has(entry.name)) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((ok, fail) => reader.readEntries(ok, fail));
        if (!batch.length) break;
        for (const e of batch) await walk(e);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

async function loadFiles(files) {
  if (!viewer || busy || !files.length) return;
  await withBusy(`Reading ${files.length === 1 ? files[0].path : files.length + " files"}...`, async () => {
    viewer.clear_files();
    for (const { path, file } of files) {
      viewer.add_file(path, new Uint8Array(await file.arrayBuffer()));
    }
    const main = viewer.pick_main();
    if (!main) {
      viewer.clear_files();
      const names = files.slice(0, 4).map((f) => f.path).join(", ") + (files.length > 4 ? ", ..." : "");
      throw new Error(`None of these is a model polysmelt reads: ${names}.\nOpen "Formats this page reads" to see what it takes.`);
    }
    await show(main);
  });
}

async function loadSample(sample) {
  if (!viewer || busy) return;
  await withBusy(`Fetching ${sample.main}...`, async () => {
    const index = await (await fetch("samples/index.json")).json();
    const files = index[sample.dir] ?? [sample.main];
    viewer.clear_files();
    await Promise.all(
      files.map(async (name) => {
        const response = await fetch(`samples/${sample.dir}/${name}`);
        if (!response.ok) throw new Error(`Couldn't fetch the sample (${response.status}).`);
        viewer.add_file(name, new Uint8Array(await response.arrayBuffer()));
      }),
    );
    await show(sample.main);
  });
}

async function show(main) {
  $("busy-text").textContent = `Smelting ${main.split("/").pop()}...`;
  // Let the page show that before the work starts.
  await new Promise((ok) => requestAnimationFrame(() => setTimeout(ok, 0)));
  const json = viewer.load(main);
  info = JSON.parse(json);
  hideError();
  $("drop").classList.add("loaded");
  $("toolbar").hidden = false;
  showHint();
  renderInfo(info);
  renderPlayer(info);
  const skeleton = $("skeleton");
  skeleton.hidden = !info.skeleton;
  skeleton.classList.toggle("on", info.bonesOnly);
  skeleton.setAttribute("aria-pressed", String(info.bonesOnly));
  decodeImages(info.generation);
}

// The model's images, decoded by the browser and handed to the GPU as they're ready.
async function decodeImages(generation) {
  const count = viewer.image_count();
  const max = viewer.max_texture_size();
  const options = { premultiplyAlpha: "none", colorSpaceConversion: "none" };
  await Promise.all(
    Array.from({ length: count }, async (_, i) => {
      try {
        const blob = new Blob([viewer.image_bytes(i)], { type: viewer.image_mime(i) });
        let bitmap = await createImageBitmap(blob, options);
        if (bitmap.width > max || bitmap.height > max) {
          const k = max / Math.max(bitmap.width, bitmap.height);
          const resized = await createImageBitmap(bitmap, {
            ...options,
            resizeWidth: Math.max(1, Math.floor(bitmap.width * k)),
            resizeHeight: Math.max(1, Math.floor(bitmap.height * k)),
            resizeQuality: "high",
          });
          bitmap.close();
          bitmap = resized;
        }
        if (viewer && viewer.generation() === generation) viewer.set_image(generation, i, bitmap);
        bitmap.close();
      } catch (e) {
        console.warn(`image ${i} couldn't be decoded`, e);
      }
    }),
  );
}

async function withBusy(text, work) {
  busy = true;
  $("busy-text").textContent = text;
  $("busy").hidden = false;
  try {
    await work();
  } catch (e) {
    if (e instanceof WebAssembly.RuntimeError) {
      crashed(e);
    } else {
      console.warn(e);
      showError(String(e?.message ?? e));
    }
  } finally {
    busy = false;
    $("busy").hidden = true;
  }
}

function download() {
  if (!viewer || !info) return;
  try {
    const glb = viewer.to_glb();
    const name = info.file.split("/").pop().replace(/\.[^.]*$/, "") + ".glb";
    const url = URL.createObjectURL(new Blob([glb], { type: "model/gltf-binary" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) {
    showError(`Couldn't write GLB: ${e?.message ?? e}`);
  }
}

// Pointer input: drag turns, right drag or shift drag pans, two fingers pan and pinch, the wheel zooms.

function wirePointer() {
  const pointers = new Map();
  let pinch = null;
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 1 || e.button === 2 || e.shiftKey });
    pinch = null;
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = pointers.get(e.pointerId);
    if (!p || !viewer) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    if (pointers.size === 1) {
      if (p.pan) viewer.pan(dx * dpr(), dy * dpr());
      else viewer.orbit(dx, dy);
    }
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const span = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) {
        viewer.pan((mid.x - pinch.mid.x) * dpr(), (mid.y - pinch.mid.y) * dpr());
        if (span > 0 && pinch.span > 0) viewer.zoom(pinch.span / span);
      }
      pinch = { mid, span };
    }
  });
  const up = (e) => {
    pointers.delete(e.pointerId);
    pinch = null;
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const scale = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
      viewer?.zoom(Math.exp(e.deltaY * scale * 0.0012));
    },
    { passive: false },
  );
  canvas.addEventListener("dblclick", () => viewer?.reset_view());
}

// The animation player

function wirePlayer() {
  $("play").addEventListener("click", () => setPlaying(!playing));
  $("animation").addEventListener("change", (e) => {
    const i = Number(e.target.value);
    viewer?.set_animation(i);
    setRange(i);
    setPlaying(i >= 0);
  });
  const scrub = $("scrub");
  scrub.addEventListener("pointerdown", () => (scrubbing = true));
  scrub.addEventListener("pointerup", () => (scrubbing = false));
  scrub.addEventListener("input", () => {
    viewer?.set_time(Number(scrub.value));
    showTime(Number(scrub.value));
  });
  $("speed").addEventListener("change", (e) => viewer?.set_speed(Number(e.target.value)));
}

let playing = false;

function setPlaying(on) {
  playing = on;
  viewer?.set_playing(on);
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  $("play-icon").innerHTML = on
    ? '<path d="M7 5h3v14H7zM14 5h3v14h-3z" fill="currentColor"/>'
    : '<path d="M7 4.5v15l12-7.5z" fill="currentColor"/>';
}

function setRange(i) {
  const a = info?.animations[i];
  const scrub = $("scrub");
  scrub.disabled = !a;
  scrub.min = a ? a.start : 0;
  scrub.max = a ? Math.max(a.end, a.start + 0.001) : 1;
  scrub.value = scrub.min;
  showTime(Number(scrub.min));
}

function showTime(t) {
  $("scrub").value = t;
  $("time").textContent = `${t.toFixed(2)} s`;
}

function renderPlayer(info) {
  const player = $("player");
  const list = info.animations;
  player.hidden = !list.length;
  if (!list.length) return;
  const select = $("animation");
  select.replaceChildren(
    ...list.map((a, i) => new Option(a.name + (a.morph ? " (morph)" : ""), i)),
    new Option("Rest pose", -1),
  );
  select.value = "0";
  setRange(0);
  setPlaying(true);
  viewer.set_speed(Number($("speed").value));
}

// The panel

function renderSamples() {
  const grid = $("samples");
  for (const s of SAMPLES) {
    const button = document.createElement("button");
    button.className = "sample";
    const ext = s.main.split(".").pop();
    button.innerHTML = `<b>${s.label}</b><span><span class="ext">.${ext}</span> &middot; ${s.note}</span>`;
    button.addEventListener("click", () => loadSample(s));
    grid.append(button);
  }
}

function renderFormats() {
  $("format-list").innerHTML = EXTENSIONS.map(
    (e) => `<code class="${UNSUPPORTED.has(e) ? "no" : ""}" title="${UNSUPPORTED.has(e) ? "native only" : ""}">.${e}</code>`,
  ).join("");
}

const number = (n) => Number(n).toLocaleString("en-US");

function meters(m) {
  const a = Math.abs(m);
  if (a === 0) return "0 m";
  if (a < 0.01) return `${(m * 1000).toPrecision(3)} mm`;
  if (a < 1) return `${(m * 100).toPrecision(3)} cm`;
  if (a < 1000) return `${m.toPrecision(3)} m`;
  return `${(m / 1000).toPrecision(3)} km`;
}

function row(dl, label, value, tag) {
  if (value === null || value === undefined || value === "") return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (tag) {
    const span = document.createElement("span");
    span.className = `tag ${tag}`;
    span.textContent = tag;
    dd.append(span);
  }
  dl.append(dt, dd);
}

function fact(dl, label, f, show = (v) => v) {
  if (!f) return;
  row(dl, label, show(f.value), f.declared ? "declared" : "assumed");
}

function renderInfo(info) {
  $("info").hidden = false;
  $("info-file").textContent = info.file.split("/").pop();
  $("info-format").textContent = [info.format, info.version && `version ${info.version}`].filter(Boolean).join(", ");
  const stats = $("stats");
  stats.replaceChildren();
  if (info.triangles) row(stats, "Triangles", number(info.triangles));
  if (info.vertices) row(stats, "Vertices", number(info.vertices));
  if (info.points) row(stats, "Points", number(info.points));
  if (info.lines) row(stats, "Lines", number(info.lines));
  if (info.size) {
    const [x, y, z] = info.size;
    row(stats, info.unitsKnown ? "Size" : "Size (units)", info.unitsKnown ? `${meters(x)} × ${meters(y)} × ${meters(z)}` : `${x.toPrecision(3)} × ${y.toPrecision(3)} × ${z.toPrecision(3)}`);
  }
  row(stats, "Meshes", info.meshes ? number(info.meshes) : null);
  row(stats, "Materials", info.materials ? number(info.materials) : null);
  row(stats, "Textures", info.textures ? number(info.textures) : null);
  row(stats, "Nodes", info.nodes ? number(info.nodes) : null);
  row(stats, "Skins", info.skins ? `${info.skins} (${info.bones} bones)` : null);
  row(stats, "Animations", info.animations.length ? number(info.animations.length) : null);
  row(stats, "Cameras", info.cameras ? number(info.cameras) : null);
  row(stats, "Lights", info.lights ? number(info.lights) : null);
  if (info.animations.length) row(stats, "Posed", info.motion === "gpu" ? "on the GPU (skinning)" : "by flattening each frame");
  row(stats, "Read in", `${info.parseMs.toFixed(1)} ms`);
  row(stats, "Flattened in", `${info.flattenMs.toFixed(1)} ms`);

  const facts = $("facts");
  facts.replaceChildren();
  fact(facts, "Units", info.metersPerUnit, (v) => {
    const unit = info.unit ? `${info.unit}, ` : "";
    return `${unit}${Number(v)} m per unit`;
  });
  fact(facts, "Up axis", info.upAxis);
  fact(facts, "Handedness", info.handedness);
  fact(facts, "Frame rate", info.frameRate, (v) => `${Number(Number(v).toFixed(3))} fps`);
  const converted = Object.entries(info.converted).filter(([, v]) => v).map(([k]) => ({ units: "to meters", upAxis: "to +Y up", handedness: "to right-handed" })[k]);
  row(facts, "Converted", converted.length ? converted.join(", ") : null);
  row(facts, "Tool", info.tool);
  row(facts, "Author", info.author);
  row(facts, "Title", info.title);
  row(facts, "Copyright", info.copyright);
  row(facts, "Created", info.created);
}

function resetPanel() {
  $("info").hidden = true;
  $("player").hidden = true;
  $("toolbar").hidden = true;
  $("drop").classList.remove("loaded");
}

let hintShown = false;
function showHint() {
  if (hintShown) return;
  hintShown = true;
  const hint = $("hint");
  hint.hidden = false;
  setTimeout(() => hint.classList.add("fade"), 4000);
  setTimeout(() => (hint.hidden = true), 5000);
}

let errorTimer = 0;
function showError(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  $("hint").hidden = true;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(hideError, 9000);
}

function hideError() {
  $("toast").hidden = true;
}

start();
