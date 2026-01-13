const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const streamBox = document.getElementById('stream');
const thinkBox = document.getElementById('think');
const ragBox = document.getElementById('rag');
const ragTopK = document.getElementById('ragTopK');
const ragStatus = document.getElementById('ragStatus');
const ragProcess = document.getElementById('ragProcess');
const sources = document.getElementById('sources');

const history = [];
let mcpReady = false;

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
    const title = document.createElement('strong');
    title.textContent = `[${idx + 1}] ${chunk.source}`;
    const body = document.createElement('div');
    body.textContent = chunk.text;
    div.appendChild(title);
    div.appendChild(body);
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

async function fetchRagInfo() {
  try {
    const resp = await fetch('/rag/info');
    if (!resp.ok) return;
    const info = await resp.json();
    const enabled = info.enabled ? '已启用' : '未启用';
    const tool = mcpReady ? 'MCP: rag_search' : 'MCP: 不可用';
    setStatus(`索引状态：${enabled} · 文档 ${info.doc_count} · 片段 ${info.chunk_count} · ${tool}`);
  } catch (err) {
    setStatus('索引状态：无法读取');
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
    throw new Error(`MCP ${resp.status}: ${msg}`);
  }
  const data = await resp.json();
  return data.result;
}

async function sendMessage(content) {
  sendBtn.disabled = true;
  const stream = streamBox.checked;
  const disableThinking = thinkBox.checked;
  const ragEnabled = ragBox.checked;
  const topK = parseInt(ragTopK.value, 10) || 4;

  history.push({ role: 'user', content });
  addMessage('user', content);
  addMessage('assistant', '生成中...');

  resetProcess();
  let ragChunks = [];
  let ragContext = '';
  let ragActive = false;
  if (ragEnabled) {
    if (!mcpReady) {
      logProcess('MCP 工具不可用，跳过检索', true);
    } else {
      try {
        logProcess('调用 MCP 工具 rag_search...', true);
        const result = await callMcpTool('rag_search', { query: content, top_k: topK });
        ragChunks = result.chunks || [];
        ragContext = result.context || buildRagContext(ragChunks);
        ragActive = true;
        logProcess(`检索完成：命中 ${ragChunks.length} 个片段`);
        if (ragContext) {
          logProcess(`构造上下文：${ragContext.length} 字符`);
        }
        renderSources({ enabled: true, chunks: ragChunks });
      } catch (err) {
        logProcess(`检索失败：${err.message}`, true);
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

fetchMcpTools().then(fetchRagInfo);
renderSources({ enabled: false });
