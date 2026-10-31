/**
 * homotope_cli.mjs — Homotope 矢量化 CLI（供 Python 侧调用）
 *
 *   node homotope_cli.mjs --in 图片 --preset line-art [--out out.svg] [--json stats.json]
 *
 * 纯本地：像素不上传任何服务器（Homotope 引擎是纯 TS，无网络依赖）。
 * 输出 SVG 到 stdout（--out 时写文件），统计信息以 JSON 写在最后一行 `__STATS__ {...}`。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { extname } from "node:path";
import { vectorize, presetOptions } from "@homotope/engine";
import { decodePng, decodeJpeg } from "@homotope/engine/node";

function parseArgs(argv) {
  const a = { preset: "line-art" };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--in") a.in = argv[++i];
    else if (k === "--out") a.out = argv[++i];
    else if (k === "--preset") a.preset = argv[++i];
    else if (k === "--json") a.json = argv[++i];
  }
  return a;
}

async function decode(path) {
  const buf = new Uint8Array(readFileSync(path));
  const ext = extname(path).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return decodeJpeg(buf);
  return decodePng(buf);
}

const args = parseArgs(process.argv);
if (!args.in) {
  process.stdout.write(JSON.stringify({ error: "missing --in" }));
  process.exit(2);
}

try {
  const image = await decode(args.in);
  const opts = presetOptions(args.preset);
  const t0 = Date.now();
  const res = vectorize(image, opts);
  const svg = res.svg || "";
  const stats = {
    ok: true,
    preset: args.preset,
    width: image.width,
    height: image.height,
    paths: res.stats?.paths ?? null,
    colors: res.palette?.colors?.length ?? null,
    svg_bytes: svg.length,
    ms: Date.now() - t0,
  };
  if (args.out) writeFileSync(args.out, svg, "utf8");
  if (args.json) writeFileSync(args.json, JSON.stringify(stats), "utf8");
  process.stdout.write(svg);
  process.stderr.write("__STATS__ " + JSON.stringify(stats));
} catch (e) {
  process.stderr.write("__STATS__ " + JSON.stringify({ ok: false, error: String(e && e.message || e) }));
  process.exit(1);
}
