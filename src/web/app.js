const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const streamBox = document.getElementById('stream');
const thinkBox = document.getElementById('think');
const ragBox = document.getElementById('rag');
const retrievalMode = document.getElementById('retrievalMode');
const ragTopK = document.getElementById('ragTopK');
const ragStatus = document.getElementById('ragStatus');
const ragProcess = document.getElementById('ragProcess');
const sources = document.getElementById('sources');
const docsList = document.getElementById('docsList');
const uploadForm = document.getElementById('uploadForm');
const uploadFile = document.getElementById('uploadFile');
const uploadBtn = document.getElementById('uploadBtn');
const tokenStats = document.getElementById('tokenStats');
const memStats = document.getElementById('memStats');

const history = [];
let mcpReady = false;
let lastUsage = null;
let lastMem = null;

function setTokenStats(usage) {
  if (!tokenStats) return;
  if (!usage || typeof usage !== 'object') {
    tokenStats.textContent = 'Tokens：-';
    return;
  }
  const prompt = Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens : null;
  const completion = Number.isFinite(usage.completion_tokens) ? usage.completion_tokens : null;
  const total = Number.isFinite(usage.total_tokens) ? usage.total_tokens : (prompt !== null && completion !== null ? prompt + completion : null);

  if (total === null) {
    tokenStats.textContent = 'Tokens：-';
    return;
  }
  const parts = [];
  parts.push(`KV ${total}`);
  if (prompt !== null) parts.push(`prompt ${prompt}`);
  if (completion !== null) parts.push(`gen ${completion}`);
  tokenStats.textContent = `Tokens：${parts.join(' · ')}`;
  lastUsage = usage;
}

function toMiB(bytes) {
  const b = typeof bytes === 'number' ? bytes : Number(bytes);
  if (!Number.isFinite(b)) return null;
  return b / 1024 / 1024;
}

function fmtMiB(bytes, digits = 1) {
  const m = toMiB(bytes);
  if (m === null) return null;
  return `${m.toFixed(digits)} MiB`;
}

function setMemStats(mem) {
  if (!memStats) return;
  if (!mem || typeof mem !== 'object') {
    memStats.textContent = 'Mem：-';
    return;
  }
  const rss = fmtMiB(mem.rss_bytes);
  const hwm = fmtMiB(mem.hwm_bytes);
  const kv = fmtMiB(mem.kv_cache_bytes);
  const parts = [];
  if (rss) parts.push(`RSS ${rss}`);
  if (hwm) parts.push(`HWM ${hwm}`);
  if (kv) parts.push(`KV ${kv}`);
  memStats.textContent = parts.length ? `Mem：${parts.join(' · ')}` : 'Mem：-';
  lastMem = mem;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderAssistantStreaming(raw) {
  if (!raw) return '';
  let html = '';
  let pos = 0;
  while (pos < raw.length) {
    const start = raw.indexOf('<think>', pos);
    if (start === -1) {
      html += escapeHtml(raw.slice(pos));
      break;
    }
    html += escapeHtml(raw.slice(pos, start));
    const close = raw.indexOf('</think>', start + 7);
    if (close === -1) {
      const thinkText = raw.slice(start + 7);
      html += `<details class="details-think" open><summary>思考中</summary><pre>${escapeHtml(thinkText)}</pre></details>`;
      pos = raw.length;
      break;
    } else {
      const thinkText = raw.slice(start + 7, close);
      html += `<details class="details-think"><summary>思考过程</summary><pre>${escapeHtml(thinkText)}</pre></details>`;
      pos = close + 8;
    }
  }
  return html || '&nbsp;';
}

function addMessage(role, text, allowHtml = false) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role === 'user' ? 'user' : 'assistant'}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (allowHtml) bubble.innerHTML = text;
  else bubble.textContent = text;
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function updateLastAssistantHTML(html) {
  const bubbles = log.querySelectorAll('.msg.assistant .bubble');
  if (bubbles.length) {
    bubbles[bubbles.length - 1].innerHTML = html;
    log.scrollTop = log.scrollHeight;
  }
}

function resetProcess() {
  ragProcess.innerHTML = '';
}

function logProcess(text, highlight = false) {
  const div = document.createElement('div');
  div.className = 'process-item';
  if (highlight) {
    div.innerHTML = `<strong>${escapeHtml(text)}</strong>`;
  } else {
    div.textContent = text;
  }
  ragProcess.appendChild(div);
}

function setStatus(text) {
  ragStatus.textContent = text;
}

function renderSources(rag) {
  sources.innerHTML = '';
  if (!rag || !rag.enabled) {
    sources.innerHTML = '<div class="source"><strong>RAG 关闭</strong>启用后会展示检索片段。</div>';
    return;
  }
  if (!rag.chunks || rag.chunks.length === 0) {
    sources.innerHTML = '<div class="source"><strong>无命中</strong>没有找到相关片段。</div>';
    return;
  }
  rag.chunks.forEach((chunk, idx) => {
    const div = document.createElement('div');
    div.className = 'source';
    const title = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = `[${idx + 1}] ${chunk.source}`;
    title.appendChild(strong);
    const meta = document.createElement('div');
    meta.className = 'meta';
    const score = typeof chunk.score === 'number' ? chunk.score.toFixed(4) : '';
    meta.textContent = score ? `score ${score}` : '';
    if (meta.textContent) title.appendChild(meta);
    if (chunk.url) {
      const link = document.createElement('a');
      link.href = chunk.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.style.marginLeft = '8px';
      link.textContent = '打开原文';
      title.appendChild(link);
    }
    div.appendChild(title);
    sources.appendChild(div);
  });
}

function buildRagContext(chunks) {
  if (!chunks || chunks.length === 0) return '';
  let ctx = '';
  chunks.forEach((chunk, idx) => {
    ctx += `[${idx + 1}] Source: ${chunk.source}\n`;
    ctx += `${chunk.text}\n\n`;
  });
  return ctx.trim();
}

function buildSystemPrompt(context, ragEnabled) {
  let prompt =
    'You are a helpful assistant. Answer using the provided context. ' +
    'If the context does not contain the answer, say you do not know. ' +
    'Keep responses concise and cite sources by their bracketed ids.';
  if (ragEnabled) {
    if (context) prompt += `\n\nContext:\n${context}`;
    else prompt += '\n\nContext:\n(No relevant sources found.)';
  }
  return prompt;
}

async function summarizeKeywords(query) {
  const payload = {
    model: 'qwen3-0.6b',
    messages: [
      { role: 'system', content: 'Extract concise search keywords for retrieval. Output keywords only, separated by spaces.' },
      { role: 'user', content: query }
    ],
    stream: false,
    rag_mode: 'client',
    temperature: 0.0,
    top_p: 1.0,
    max_tokens: 64
  };
  const resp = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) {
    const msg = await resp.text();
    throw new Error(`keyword summarize failed: ${msg}`);
  }
  const j = await resp.json();
  const content = j.choices?.[0]?.message?.content || '';
  return content.trim();
}

function extractKeywordsHeuristic(query, maxTokens = 8) {
  const q = (query || '').trim();
  if (!q) return '';

  const cnStop = new Set([
    '的', '了', '和', '与', '及', '或', '是', '在', '有', '没有',
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
    '这', '那', '这些', '那些', '一个', '一种', '如何', '怎么', '为什么', '什么',
    '请', '帮我', '一下', '一下子', '能否', '可以', '以及'
  ]);
  const enStop = new Set([
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'and', 'or', 'is', 'are', 'was', 'were',
    'with', 'as', 'by', 'at', 'from', 'that', 'this', 'these', 'those', 'it', 'its', 'be'
  ]);

  const counts = new Map();
  const add = (t) => {
    const tok = (t || '').trim();
    if (!tok) return;
    const lower = tok.toLowerCase();
    if (lower.length <= 1) return;
    if (cnStop.has(tok) || enStop.has(lower)) return;
    counts.set(tok, (counts.get(tok) || 0) + 1);
  };

  const cnPhrases = q.match(/[\u4e00-\u9fff]{2,}/g) || [];
  cnPhrases.forEach(add);

  const enWords = q.match(/[A-Za-z0-9]+/g) || [];
  enWords.forEach(add);

  if (counts.size === 0) {
    const han = (q.match(/[\u4e00-\u9fff]/g) || []).join('');
    for (let i = 0; i + 1 < han.length; i++) add(han.slice(i, i + 2));
  }

  const scored = Array.from(counts.entries()).map(([tok, freq]) => {
    const len = tok.length;
    return { tok, score: freq * (len >= 6 ? 2.0 : len >= 4 ? 1.5 : 1.0) };
  });
  scored.sort((a, b) => b.score - a.score || b.tok.length - a.tok.length);
  return scored.slice(0, maxTokens).map((x) => x.tok).join(' ');
}

async function fetchRagInfo() {
  try {
    const resp = await fetch('/rag/info');
    if (!resp.ok) return;
    const info = await resp.json();
    const enabled = info.enabled ? '已启用' : '未启用';
    const tool = mcpReady ? 'MCP: rag_search' : 'MCP: 不可用';
    const dim = info.embed_dim ? `dim ${info.embed_dim}` : 'dim ?';
    let extra = '';
    if (info.ready === false && info.error) extra = ` · 错误：${info.error}`;
    setStatus(`索引状态：${enabled} · 文档 ${info.doc_count} · 片段 ${info.chunk_count} · ${dim} · ${tool}${extra}`);
  } catch (err) {
    setStatus('索引状态：无法读取');
  }
}

function renderDocsList(docs) {
  if (!docsList) return;
  docsList.innerHTML = '';
  if (!docs || docs.length === 0) {
    docsList.innerHTML = '<div class="source"><strong>暂无文档</strong>请先上传 txt/pdf。</div>';
    return;
  }
  docs.forEach((d) => {
    const div = document.createElement('div');
    div.className = 'source';
    const title = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = `${d.filename || 'unknown'} · ${d.chunk_count || 0} 片段`;
    title.appendChild(strong);
    if (d.url) {
      const link = document.createElement('a');
      link.href = d.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.style.marginLeft = '8px';
      link.textContent = '查看';
      title.appendChild(link);
    }
    if (d.id) {
      const del = document.createElement('button');
      del.type = 'button';
      del.textContent = '删除';
      del.style.marginLeft = '8px';
      del.addEventListener('click', async () => {
        const ok = confirm(`确定删除文档？\\n${d.filename || ''}\\n(id=${d.id})`);
        if (!ok) return;
        try {
          const resp = await fetch(`/rag/doc/${encodeURIComponent(d.id)}`, { method: 'DELETE' });
          if (!resp.ok) {
            const msg = await resp.text();
            throw new Error(msg);
          }
          await fetchRagInfo();
          await fetchDocsList();
        } catch (err) {
          alert(`删除失败：${err.message}`);
        }
      });
      title.appendChild(del);
    }
    div.appendChild(title);
    docsList.appendChild(div);
  });
}

async function fetchDocsList() {
  if (!docsList) return;
  try {
    const resp = await fetch('/rag/docs?limit=200');
    if (!resp.ok) {
      const msg = await resp.text();
      docsList.innerHTML = `<div class="source"><strong>读取失败</strong>${escapeHtml(msg)}</div>`;
      return;
    }
    const data = await resp.json();
    renderDocsList(data.docs || []);
  } catch (err) {
    docsList.innerHTML = `<div class="source"><strong>读取失败</strong>${escapeHtml(err.message)}</div>`;
  }
}

async function fetchMcpTools() {
  try {
    const resp = await fetch('/mcp/tools/list');
    if (!resp.ok) return;
    const tools = await resp.json();
    mcpReady = Array.isArray(tools) && tools.some((t) => t.name === 'rag_search');
  } catch (err) {
    mcpReady = false;
  }
}

async function callMcpTool(name, args) {
  const resp = await fetch('/mcp/tools/call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, arguments: args })
  });
  if (!resp.ok) {
    const msg = await resp.text();
    try {
      const j = JSON.parse(msg);
      const err = new Error(j?.error?.message ? `MCP ${resp.status}: ${j.error.message}` : `MCP ${resp.status}: ${msg}`);
      if (Array.isArray(j?.trace)) err.trace = j.trace;
      throw err;
    } catch (_) {
      throw new Error(`MCP ${resp.status}: ${msg}`);
    }
  }
  const data = await resp.json();
  return data.result;
}

async function sendMessage(content) {
  sendBtn.disabled = true;
  const stream = streamBox.checked;
  const disableThinking = thinkBox.checked;
  const ragEnabled = ragBox.checked;
  const topK = parseInt(ragTopK.value, 10) || 10;
  const mode = retrievalMode?.value || 'heuristic';

  history.push({ role: 'user', content });
  addMessage('user', content);
  addMessage('assistant', '生成中...');
  setTokenStats(null);
  setMemStats(null);

  resetProcess();
  let ragChunks = [];
  let ragContext = '';
  let ragActive = false;
  if (ragEnabled) {
    if (!mcpReady) {
      logProcess('MCP 工具不可用，跳过检索', true);
    } else {
      try {
        let keywordQuery = content;
        if (mode === 'raw') {
          logProcess('检索模式：原始问题', true);
          keywordQuery = content;
        } else if (mode === 'llm') {
          logProcess('检索模式：LLM 关键词', true);
          logProcess(`关键词提炼输入：${content}`);
          try {
            const summarized = await summarizeKeywords(content);
            if (summarized) {
              keywordQuery = summarized;
              logProcess(`关键词提炼输出：${keywordQuery}`, true);
            } else {
              logProcess('关键词为空，使用原始问题', true);
            }
          } catch (err) {
            logProcess(`关键词提炼失败：${err.message}（回退到原始问题）`, true);
            keywordQuery = content;
          }
        } else {
          logProcess('检索模式：传统关键词', true);
          keywordQuery = extractKeywordsHeuristic(content);
          if (keywordQuery) {
            logProcess(`关键词提炼输出：${keywordQuery}`, true);
          } else {
            keywordQuery = content;
            logProcess('关键词为空，使用原始问题', true);
          }
        }

        logProcess('调用 MCP 工具 rag_search...', true);
        const result = await callMcpTool('rag_search', { query: keywordQuery, top_k: topK });
        ragChunks = result.chunks || [];
        ragContext = result.context || buildRagContext(ragChunks);
        ragActive = true;
        if (Array.isArray(result.trace)) {
          result.trace.forEach((line) => logProcess(`检索步骤：${line}`));
        }
        if (result.elapsed_ms !== undefined) {
          logProcess(`检索耗时：${result.elapsed_ms} ms`);
        }
        logProcess(`检索完成：命中 ${ragChunks.length} 个片段`);
        if (ragContext) {
          logProcess(`构造上下文：${ragContext.length} 字符`);
        }
        renderSources({ enabled: true, chunks: ragChunks });
      } catch (err) {
        logProcess(`检索失败：${err.message}`, true);
        if (Array.isArray(err.trace)) {
          err.trace.forEach((line) => logProcess(`检索错误步骤：${line}`));
        }
      }
    }
  }
  if (!ragActive) {
    renderSources({ enabled: false });
  }

  const systemPrompt = buildSystemPrompt(ragContext, ragEnabled && ragActive);
  const payload = {
    model: 'qwen3-0.6b',
    messages: [{ role: 'system', content: systemPrompt }, ...history],
    stream,
    enable_thinking: !disableThinking,
    rag_mode: 'client',
    temperature: 0.7,
    top_p: 0.9
  };

  let assistantText = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${msg}`);
    }

    if (stream) {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let streamDone = false;
      while (!streamDone) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split('\n\n');
        buf = parts.pop();
        for (const block of parts) {
          const line = block.trim();
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') { streamDone = true; break; }
          try {
            const j = JSON.parse(data);
            if (j.usage) setTokenStats(j.usage);
            if (j.mem) setMemStats(j.mem);
            const delta = j.choices?.[0]?.delta?.content || '';
            assistantText += delta;
            updateLastAssistantHTML(renderAssistantStreaming(assistantText));
          } catch (err) {
            console.warn('parse chunk failed', err);
          }
        }
        if (streamDone) break;
      }
    } else {
      const j = await resp.json();
      assistantText = j.choices?.[0]?.message?.content || '';
      updateLastAssistantHTML(renderAssistantStreaming(assistantText));
      if (j.usage) setTokenStats(j.usage);
      if (j.mem) setMemStats(j.mem);
    }
    history.push({ role: 'assistant', content: assistantText });
  } catch (err) {
    updateLastAssistantHTML(escapeHtml(`出错了：${err.message}`));
  } finally {
    sendBtn.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendMessage(text);
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = uploadFile.files[0];
  if (!file) {
    resetProcess();
    logProcess('请选择要上传的文件', true);
    return;
  }
  resetProcess();
  logProcess(`开始上传：${file.name}`, true);
  uploadBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch('/rag/upload', { method: 'POST', body: formData });
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(msg);
    }
    const data = await resp.json();
    if (Array.isArray(data.trace)) {
      data.trace.forEach((line) => logProcess(`索引步骤：${line}`));
    }
    if (data.doc) {
      logProcess(`索引完成：${data.doc.filename} · ${data.doc.chunks} 片段`, true);
    }
    await fetchRagInfo();
    await fetchDocsList();
  } catch (err) {
    logProcess(`上传失败：${err.message}`, true);
  } finally {
    uploadBtn.disabled = false;
    uploadFile.value = '';
  }
});

fetchMcpTools().then(async () => {
  await fetchRagInfo();
  await fetchDocsList();
});
renderSources({ enabled: false });
