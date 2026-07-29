"""Local web UI for the archaeology vector search.

Usage:
    uv run python scripts/search_ui.py
    open http://localhost:8000
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from search_milvus import search

HOST = "127.0.0.1"
PORT = 8000

CHUNK_TYPES = [
    "",
    "墓葬形制",
    "随葬器物",
    "葬式葬具",
    "年代分期",
    "发掘方法",
    "遗址背景",
    "图表数据",
    "小件器物",
    "遗物",
    "其他",
]


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Archaeology DataLab</title>
  <style>
    :root {
      --bg: #f6f8fc;
      --panel: rgba(255, 255, 255, 0.86);
      --panel-strong: #ffffff;
      --line: rgba(23, 48, 92, 0.12);
      --text: #102033;
      --muted: #64748b;
      --blue: #2367ff;
      --cyan: #00a7c7;
      --violet: #7658ff;
      --green: #087f5b;
      --danger: #d92d4b;
      --shadow: 0 24px 70px rgba(30, 58, 138, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 10% 8%, rgba(35, 103, 255, .13), transparent 26rem),
        radial-gradient(circle at 88% 4%, rgba(118, 88, 255, .12), transparent 30rem),
        linear-gradient(180deg, #fbfdff 0%, #f4f7fc 54%, #eef3fb 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(35,103,255,.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(35,103,255,.055) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0 64px; position: relative; }
    .nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 44px; }
    .brand { display: flex; gap: 12px; align-items: center; font-weight: 700; letter-spacing: .02em; }
    .mark {
      width: 36px; height: 36px; border-radius: 12px;
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      box-shadow: 0 0 32px rgba(84, 240, 255, .35);
    }
    .nav-links { display: flex; gap: 18px; color: var(--muted); font-size: 14px; }
    .hero { display: grid; grid-template-columns: 1.05fr .95fr; gap: 28px; align-items: stretch; }
    .hero-copy {
      padding: 44px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: linear-gradient(135deg, rgba(255, 255, 255, .95), rgba(239, 246, 255, .72));
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .eyebrow { color: var(--green); font-size: 13px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 18px 0 16px; font-size: clamp(40px, 7vw, 78px); line-height: .92; letter-spacing: -.06em; }
    .lead { max-width: 680px; color: #52657d; font-size: 18px; line-height: 1.75; }
    .stats { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 30px; }
    .stat { border: 1px solid var(--line); border-radius: 18px; padding: 14px 18px; background: rgba(35,103,255,.045); min-width: 138px; }
    .stat b { display:block; font-size: 24px; }
    .stat span { color: var(--muted); font-size: 13px; }
    .search-panel {
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: rgba(255, 255, 255, .88);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .panel-head { display:flex; justify-content:space-between; align-items:center; margin-bottom: 18px; }
    .pill { color: var(--cyan); border: 1px solid rgba(84,240,255,.28); background: rgba(84,240,255,.08); padding: 7px 10px; border-radius: 999px; font-size: 12px; }
    label { display: block; margin: 14px 0 8px; color: #263b57; font-size: 13px; font-weight: 700; }
    textarea, input, select {
      width: 100%; border: 1px solid rgba(23, 48, 92, .16); color: var(--text);
      background: rgba(255,255,255,.8); border-radius: 15px; padding: 13px 14px;
      outline: none; font: inherit;
    }
    textarea { min-height: 116px; resize: vertical; }
    textarea:focus, input:focus, select:focus { border-color: var(--cyan); box-shadow: 0 0 0 4px rgba(84,240,255,.1); }
    .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    button {
      width: 100%; margin-top: 18px; border: 0; border-radius: 16px; padding: 15px 18px;
      color: #04111e; font-weight: 800; cursor: pointer;
      background: linear-gradient(135deg, var(--cyan), var(--blue) 55%, var(--violet));
      box-shadow: 0 18px 42px rgba(57,168,255,.27);
    }
    button:disabled { opacity: .55; cursor: wait; }
    .results { margin-top: 28px; display:grid; gap: 16px; }
    .result {
      border: 1px solid var(--line); border-radius: 24px; padding: 20px;
      background: rgba(255, 255, 255, .92);
      box-shadow: 0 18px 52px rgba(30,58,138,.10);
    }
    .result-top { display:flex; gap: 10px; align-items:center; flex-wrap:wrap; margin-bottom: 12px; }
    .score { color: var(--green); font-variant-numeric: tabular-nums; font-weight: 800; }
    .tag { color: #d9e7ff; background: rgba(143,124,255,.15); border: 1px solid rgba(143,124,255,.28); padding: 5px 9px; border-radius: 999px; font-size: 12px; }
    .title { font-size: 18px; font-weight: 800; margin-bottom: 6px; }
    .meta { color: var(--muted); font-size: 13px; line-height: 1.65; margin-bottom: 14px; }
    .text { color: #263b57; line-height: 1.8; white-space: pre-wrap; max-height: 9.8em; overflow: hidden; }
    details[open] .text { max-height: none; }
    summary { cursor: pointer; color: var(--blue); font-weight: 700; margin-top: 12px; }
    .empty, .error { border: 1px dashed var(--line); border-radius: 20px; padding: 22px; color: var(--muted); background: rgba(255,255,255,.7); }
    .error { color: var(--danger); border-color: rgba(255,114,138,.34); }
    @media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-copy { padding: 28px; } .nav-links { display:none; } }
    @media (max-width: 560px) { .grid { grid-template-columns: 1fr; } .shell { width: min(100% - 20px, 1180px); } h1 { font-size: 43px; } }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="nav">
      <div class="brand"><div class="mark"></div><span>Archaeology DataLab</span></div>
      <div class="nav-links"><span>Milvus Lite</span><span>BAAI/bge-m3</span><span>SiliconFlow</span></div>
    </nav>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">OPEN ARCHAEOLOGY RETRIEVAL</div>
        <h1>让考古报告变成可检索的数据层。</h1>
        <p class="lead">输入自然语言问题，从本地向量索引中召回墓葬形制、随葬器物、分期断代等原文证据。先查证据，再做 RAG。</p>
        <div class="stats">
          <div class="stat"><b>84</b><span>文档</span></div>
          <div class="stat"><b>9674</b><span>目标 chunks</span></div>
          <div class="stat"><b>1024</b><span>向量维度</span></div>
        </div>
      </div>
      <form class="search-panel" id="form">
        <div class="panel-head"><strong>语义检索</strong><span class="pill">local index</span></div>
        <label for="query">问题</label>
        <textarea id="query" name="query" placeholder="例如：竖穴土坑墓 典型随葬品">竖穴土坑墓 典型随葬品</textarea>
        <div class="grid">
          <div><label for="top_k">返回条数</label><input id="top_k" name="top_k" type="number" min="1" max="20" value="5" /></div>
          <div><label for="chunk_type">内容类型</label><select id="chunk_type" name="chunk_type"></select></div>
          <div><label for="corpus">语料</label><select id="corpus" name="corpus"><option value="">全部</option><option value="facts">facts</option><option value="methods">methods</option></select></div>
          <div><label for="region">地区</label><input id="region" name="region" placeholder="可留空" /></div>
          <div><label for="period">时期</label><input id="period" name="period" placeholder="如：汉" /></div>
        </div>
        <button id="submit" type="submit">检索证据</button>
      </form>
    </section>
    <section class="results" id="results"><div class="empty">等待查询。索引重建完成后即可检索。</div></section>
  </main>
  <script>
    const chunkTypes = __CHUNK_TYPES__;
    const select = document.querySelector('#chunk_type');
    chunkTypes.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v || '全部';
      select.appendChild(opt);
    });
    const form = document.querySelector('#form');
    const results = document.querySelector('#results');
    const button = document.querySelector('#submit');
    const esc = s => String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const clean = s => String(s || '').replace(/\\s+/g, ' ').trim();
    const preview = s => clean(s).slice(0, 360) + (clean(s).length > 360 ? '…' : '');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const params = new URLSearchParams(new FormData(form));
      button.disabled = true;
      button.textContent = '检索中...';
      results.innerHTML = '<div class="empty">正在连接 Milvus 和 embedding API...</div>';
      try {
        const res = await fetch('/api/search?' + params.toString());
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || '检索失败');
        if (!data.results.length) {
          results.innerHTML = '<div class="empty">没有命中。换一个问题或取消过滤条件。</div>';
          return;
        }
        results.innerHTML = data.results.map((hit, i) => {
          const e = hit.entity || {};
          const fullText = clean(e.text);
          return `<article class="result">
            <div class="result-top"><span class="score">#${i + 1} · ${Number(hit.distance).toFixed(4)}</span></div>
            <div class="title">${esc(e.title || '未命名文档')}</div>
            <div class="meta"><strong>章节</strong>：${esc(e.heading || '无')}<br><strong>来源</strong>：${esc(e.source_file || '')}<br><strong>属性</strong>：${esc([e.region, e.period, e.chunk_topics].filter(Boolean).join(' · ') || '无')}</div>
            <div class="text">${esc(preview(fullText))}</div>
            ${fullText.length > 360 ? `<details><summary>展开全文</summary><div class="text">${esc(fullText)}</div></details>` : ''}
          </article>`;
        }).join('');
      } catch (err) {
        results.innerHTML = `<div class="error">${esc(err.message)}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = '检索证据';
      }
    });
  </script>
</body>
</html>
""".replace("__CHUNK_TYPES__", json.dumps(CHUNK_TYPES, ensure_ascii=False))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(HTML)
            return
        if parsed.path == "/api/search":
            self.handle_search(parsed.query)
            return
        self.send_error(404)

    def handle_search(self, raw_query: str) -> None:
        params = parse_qs(raw_query)
        query = params.get("query", [""])[0].strip()
        if not query:
            self.respond_json({"error": "query is required"}, status=400)
            return
        try:
            hits = search(
                query=query,
                top_k=int(params.get("top_k", ["5"])[0] or 5),
                corpus=params.get("corpus", [""])[0] or None,
                region=params.get("region", [""])[0] or None,
                period=params.get("period", [""])[0] or None,
                chunk_type=params.get("chunk_type", [""])[0] or None,
            )
            self.respond_json({"results": hits})
        except Exception as exc:
            self.respond_json({"error": str(exc)}, status=500)

    def respond_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    load_env()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
