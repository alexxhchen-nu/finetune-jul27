"""Local web UI for the archaeology vector search.

Usage:
    uv run python scripts/search_ui.py
    open http://localhost:8000
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
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

RAG_PROMPT = """你是一位考古学研究助手。根据以下检索到的考古报告原文片段，回答用户的问题。

规则：
1. 只基于提供的原文回答，不要编造
2. 每个关键论述必须标注出处，格式为 [1] [2] 等（对应原文片段编号）
3. 如果原文中有具体数字（墓数、尺寸、年代），必须引用
4. 如果原文之间有矛盾，指出矛盾
5. 如果原文不足以回答问题，说明缺少什么信息
6. 用中文回答，简洁准确"""


def call_llm(query: str, hits: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("RAG_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)

    context_parts = []
    for i, hit in enumerate(hits):
        e = hit.get("entity", {})
        context_parts.append(f"[{i+1}] 标题: {e.get('title','')} | 章节: {e.get('heading','')} | 类型: {e.get('chunk_type','')}\n{e.get('text','')}")
    context = "\n\n---\n\n".join(context_parts)

    user_msg = f"问题：{query}\n\n原文片段：\n{context}"

    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RAG_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if any(code in err for code in ("429", "500", "502", "503", "504", "rate limit", "overloaded", "timeout", "connection")):
                time.sleep(2 ** attempt)
                continue
            raise
    return f"LLM 调用失败: {last_error}"


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
  <title>Arch RAG</title>
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
    .layout { display: flex; gap: 24px; }
    .main-col { flex: 1; min-width: 0; }
    .side-col { width: 380px; flex-shrink: 0; }
    .side-panel {
      position: sticky; top: 24px; max-height: calc(100vh - 48px); overflow-y: auto;
      border: 1px solid var(--line); border-radius: 20px; padding: 18px;
      background: rgba(255,255,255,.92); box-shadow: 0 12px 40px rgba(30,58,138,.08);
    }
    .side-panel.collapsed { display: none; }
    .side-toggle {
      position: fixed; right: 16px; top: 50%; transform: translateY(-50%);
      z-index: 100; width: 40px; height: 80px; border: 1px solid var(--line);
      border-radius: 12px 0 0 12px; background: rgba(255,255,255,.95);
      box-shadow: -4px 0 20px rgba(30,58,138,.08); cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: var(--violet); font-size: 18px; font-weight: 800;
    }
    .side-toggle:hover { background: rgba(118,88,255,.08); }
    .side-title { font-size: 14px; font-weight: 800; color: var(--violet); margin-bottom: 12px; }
    .side-item {
      padding: 10px; margin-bottom: 8px; border: 1px solid var(--line); border-radius: 12px;
      cursor: pointer; font-size: 13px; line-height: 1.6;
      background: rgba(255,255,255,.6);
    }
    .side-item:hover { background: rgba(35,103,255,.06); border-color: rgba(35,103,255,.2); }
    .side-item .si-title { font-weight: 700; color: var(--text); margin-bottom: 4px; }
    .side-item .si-meta { color: var(--muted); font-size: 12px; }
    .side-item .si-text { color: #52657d; margin-top: 6px; max-height: 4.5em; overflow: hidden; }
    .tooltip {
      position: fixed; z-index: 200; max-width: 420px; padding: 14px 16px;
      background: #fff; border: 1px solid var(--line); border-radius: 14px;
      box-shadow: 0 12px 40px rgba(30,58,138,.18); font-size: 13px; line-height: 1.7;
      color: #263b57; pointer-events: none; opacity: 0; transition: opacity .15s;
    }
    .tooltip.visible { opacity: 1; }
    .tooltip .tt-title { font-weight: 800; margin-bottom: 6px; color: var(--text); }
    .tooltip .tt-meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .tooltip .tt-text { max-height: 12em; overflow: hidden; }
    @media (max-width: 1100px) { .side-col { display: none; } .side-toggle { display: none; } }
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
    .num { color: var(--blue); font-weight: 700; background: rgba(35,103,255,.08); padding: 0 3px; border-radius: 4px; }
    .key-data { display:flex; gap: 8px; flex-wrap:wrap; margin: 10px 0; }
    .key-data .chip { color: var(--text); background: rgba(35,103,255,.06); border: 1px solid rgba(35,103,255,.14); padding: 4px 10px; border-radius: 8px; font-size: 13px; }
    .key-data .chip b { color: var(--blue); }
    .group { margin-bottom: 24px; }
    .group-head { font-size: 15px; font-weight: 800; color: var(--violet); margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid rgba(118,88,255,.18); }
    .answer {
      margin-top: 28px; border: 1px solid var(--line); border-radius: 24px; padding: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(239,246,255,.82));
      box-shadow: 0 18px 52px rgba(30,58,138,.10);
    }
    .answer-head { font-size: 15px; font-weight: 800; color: var(--violet); margin-bottom: 14px; }
    .answer-body { color: #263b57; line-height: 1.9; font-size: 15px; }
    .answer-body .cite { color: var(--blue); cursor: pointer; font-weight: 700; text-decoration: underline; text-decoration-color: rgba(35,103,255,.3); }
    .answer-body .cite:hover { background: rgba(35,103,255,.08); border-radius: 4px; padding: 0 2px; }
    .evidence-highlight { background: rgba(35,103,255,.10); border-left: 3px solid var(--blue); padding: 2px 8px; border-radius: 4px; }
    @media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-copy { padding: 28px; } .nav-links { display:none; } }
    @media (max-width: 560px) { .grid { grid-template-columns: 1fr; } .shell { width: min(100% - 20px, 1180px); } h1 { font-size: 43px; } }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="nav">
      <div class="brand"><div class="mark"></div><span>Arch RAG</span></div>
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
        <button id="rag-btn" type="button" style="margin-top:10px;background:linear-gradient(135deg,var(--violet),var(--blue) 55%,var(--cyan));color:#fff;">AI 综合回答</button>
      </form>
    </section>
    <div class="layout">
      <div class="main-col">
        <section class="answer" id="answer" style="display:none;"></section>
        <section class="results" id="results"><div class="empty">等待查询。索引重建完成后即可检索。</div></section>
      </div>
      <div class="side-col">
        <div class="side-panel" id="side-panel">
          <div class="side-title">证据列表</div>
          <div id="side-items"><div class="empty" style="font-size:13px;">检索后显示全部证据</div></div>
        </div>
      </div>
    </div>
    <div class="side-toggle" id="side-toggle" title="切换证据面板">◀</div>
    <div class="tooltip" id="tooltip"></div>
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
    const hlNum = s => esc(s).replace(/(\\d[\\d,.]*\\d|\\d)(座|件|个|米|厘米|cm|mm|层|年|岁|种|类|型|式|组|座|M|m)/g, '<span class="num">$1$2</span>');
    const extractKeyData = s => {
      const nums = [...s.matchAll(/(\\d[\\d,.]*\\d|\\d)(座|件|个|米|厘米|层|年|种|类|型|式|组)/g)].slice(0, 6);
      return nums.map(m => `<span class="chip"><b>${esc(m[1])}</b>${esc(m[2])}</span>`).join('');
    };
    const answerEl = document.querySelector('#answer');
    const ragBtn = document.querySelector('#rag-btn');
    const sidePanel = document.querySelector('#side-panel');
    const sideItems = document.querySelector('#side-items');
    const sideToggle = document.querySelector('#side-toggle');
    const tooltip = document.querySelector('#tooltip');

    function updateSidePanel(hits) {
      sideItems.innerHTML = hits.map((hit, i) => {
        const e = hit.entity || {};
        return `<div class="side-item" data-idx="${i}">
          <div class="si-title">[${i+1}] ${esc(e.title || '')}</div>
          <div class="si-meta">${esc(e.heading || '')} · ${esc(e.chunk_type || '')}</div>
          <div class="si-text">${esc(clean(e.text).slice(0, 120))}…</div>
        </div>`;
      }).join('');
    }

    function showTooltip(el, hit) {
      const e = hit.entity || {};
      tooltip.innerHTML = `<div class="tt-title">${esc(e.title || '')} · ${esc(e.heading || '')}</div><div class="tt-meta">${esc(e.chunk_type || '')} · ${esc(e.source_file || '')}</div><div class="tt-text">${esc(clean(e.text).slice(0, 500))}</div>`;
      const rect = el.getBoundingClientRect();
      tooltip.style.left = Math.min(rect.left, window.innerWidth - 440) + 'px';
      tooltip.style.top = (rect.bottom + 8) + 'px';
      tooltip.classList.add('visible');
    }
    function hideTooltip() { tooltip.classList.remove('visible'); }

    sideToggle.addEventListener('click', () => {
      sidePanel.classList.toggle('collapsed');
      sideToggle.textContent = sidePanel.classList.contains('collapsed') ? '▶' : '◀';
    });

    let lastHits = [];
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
          const keyData = extractKeyData(fullText);
          return `<article class="result">
            <div class="result-top"><span class="score">#${i + 1} · ${Number(hit.distance).toFixed(4)}</span></div>
            <div class="title">${esc(e.title || '未命名文档')}</div>
            <div class="meta"><strong>章节</strong>：${esc(e.heading || '无')}<br><strong>来源</strong>：${esc(e.source_file || '')}<br><strong>属性</strong>：${esc([e.region, e.period, e.chunk_topics].filter(Boolean).join(' · ') || '无')}</div>
            ${keyData ? `<div class="key-data">${keyData}</div>` : ''}
            <div class="text">${hlNum(preview(fullText))}</div>
            ${fullText.length > 360 ? `<details><summary>展开全文</summary><div class="text">${hlNum(fullText)}</div></details>` : ''}
          </article>`;
        }).join('');
        lastHits = data.results;
        updateSidePanel(data.results);
      } catch (err) {
        results.innerHTML = `<div class="error">${esc(err.message)}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = '检索证据';
      }
    });
    ragBtn.addEventListener('click', async () => {
      const params = new URLSearchParams(new FormData(form));
      ragBtn.disabled = true;
      ragBtn.textContent = 'AI 思考中...';
      answerEl.style.display = 'block';
      answerEl.innerHTML = '<div class="answer-head">综合回答</div><div class="answer-body">正在检索 + 生成回答...</div>';
      try {
        const res = await fetch('/api/rag?' + params.toString());
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'RAG 失败');
        let answerHtml = esc(data.answer || '').replace(/\\[(\\d+)\\]/g, '<span class="cite" data-ref="$1">[$1]</span>');
        answerEl.innerHTML = `<div class="answer-head">综合回答</div><div class="answer-body">${answerHtml}</div>`;
        results.innerHTML = data.results.map((hit, i) => {
          const e = hit.entity || {};
          const fullText = clean(e.text);
          const keyData = extractKeyData(fullText);
          return `<article class="result" id="ref-${i+1}">
            <div class="result-top"><span class="score">[${i + 1}] · ${Number(hit.distance).toFixed(4)}</span></div>
            <div class="title">${esc(e.title || '未命名文档')}</div>
            <div class="meta"><strong>章节</strong>：${esc(e.heading || '无')}<br><strong>来源</strong>：${esc(e.source_file || '')}<br><strong>属性</strong>：${esc([e.region, e.period, e.chunk_topics].filter(Boolean).join(' · ') || '无')}</div>
            ${keyData ? `<div class="key-data">${keyData}</div>` : ''}
            <div class="text">${hlNum(preview(fullText))}</div>
            ${fullText.length > 360 ? `<details><summary>展开全文</summary><div class="text">${hlNum(fullText)}</div></details>` : ''}
          </article>`;
        }).join('');
        lastHits = data.results;
        updateSidePanel(data.results);
        answerEl.querySelectorAll('.cite').forEach(el => {
          const refIdx = parseInt(el.dataset.ref) - 1;
          el.addEventListener('mouseenter', () => {
            if (lastHits[refIdx]) showTooltip(el, lastHits[refIdx]);
          });
          el.addEventListener('mouseleave', hideTooltip);
          el.addEventListener('click', () => {
            const ref = document.getElementById('ref-' + el.dataset.ref);
            if (ref) {
              ref.scrollIntoView({ behavior: 'smooth', block: 'center' });
              ref.classList.add('evidence-highlight');
              setTimeout(() => ref.classList.remove('evidence-highlight'), 3000);
            }
          });
        });
      } catch (err) {
        answerEl.innerHTML = `<div class="answer-head">综合回答</div><div class="error">${esc(err.message)}</div>`;
      } finally {
        ragBtn.disabled = false;
        ragBtn.textContent = 'AI 综合回答';
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
        if parsed.path == "/api/rag":
            self.handle_rag(parsed.query)
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

    def handle_rag(self, raw_query: str) -> None:
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
            answer = call_llm(query, hits)
            self.respond_json({"answer": answer, "results": hits})
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
