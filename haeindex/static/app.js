'use strict';
const $ = id => document.getElementById(id);
const state = {documents: [], selected: new Set(), messages: [], busy: false, ready: false, active: null, searchAvailable: false};
const sessionKey = 'haeindex-buddy-session-v1';
function remember() {
  try { sessionStorage.setItem(sessionKey, JSON.stringify({messages: state.messages, selected: [...state.selected], active: state.active})); } catch { /* Storage may be disabled. Chat remains usable. */ }
}
function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $('toast').hidden = true; }, 4500);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { throw new Error('서버 응답을 읽지 못했어요. 연결을 확인해 주세요.'); }
  if (!response.ok) {
    const error = new Error(body.error || '요청을 처리하지 못했어요.');
    error.status = response.status;
    throw error;
  }
  return body;
}
function jsonPost(path, body) {
  return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
}
function setBusy(busy) {
  state.busy = busy;
  $('send').disabled = busy || !state.ready;
  $('send').firstChild.textContent = busy ? '찾는 중…' : '보내기';
  for (const id of ['add-pdf', 'attach', 'file-menu', 'new-chat']) $(id).disabled = busy;
  for (const button of document.querySelectorAll('.reindex-doc')) button.disabled = busy;
  const pending = state.documents.filter(doc => doc.indexed !== null && doc.index_complete !== true);
  $('reindex-pending').disabled = busy || !pending.length;
  $('enhance-docs').disabled = busy || !state.selected.size;
  $('activity').classList.toggle('working', busy);
}
function updateScope() {
  const selected = state.documents.filter(d => state.selected.has(d.id));
  $('room-title').textContent = selected.length === 1 ? selected[0].name : selected.length ? `선택한 문서 ${selected.length}개` : '내 문서들';
  $('scope').textContent = selected.length ? `선택한 문서 ${selected.length}개에서 찾기` : '전체 문서에서 찾기';
  $('all-docs').classList.toggle('selected', !selected.length);
  $('all-docs').setAttribute('aria-pressed', String(!selected.length));
  $('enhance-docs').disabled = state.busy || !selected.length;
  remember();
}
function renderDocs() {
  const list = $('document-list');
  list.replaceChildren();
  const term = $('doc-filter').value.toLocaleLowerCase();
  const docs = state.documents.filter(d => d.name.toLocaleLowerCase().includes(term));
  for (const doc of docs) {
    const row = document.createElement('div');
    row.className = 'doc-item' + (state.selected.has(doc.id) ? ' chosen' : '');
    row.title = doc.name;
    const choice = document.createElement('label');
    choice.className = 'doc-choice';
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = state.selected.has(doc.id);
    check.disabled = doc.indexed !== true || doc.index_complete !== true;
    check.addEventListener('change', () => {
      if (check.checked) state.selected.add(doc.id); else state.selected.delete(doc.id);
      row.classList.toggle('chosen', check.checked);
      updateScope();
    });
    const info = document.createElement('span');
    info.className = 'doc-info';
    info.append(document.createTextNode(doc.name.replace(/\.pdf$/i, '')));
    const meta = document.createElement('small');
    const pageLabel = doc.pages ? `${doc.pages}쪽 · ` : '';
    meta.textContent = doc.indexed === true && doc.index_complete === true
      ? `${pageLabel}${doc.chunks}개 문단 색인 완료`
      : doc.indexed === true
        ? `${doc.indexed_through_page || 0}/${doc.pages || '?'}쪽까지만 색인 · 재색인 필요`
      : doc.indexed === false ? `${pageLabel}색인 필요` : `${pageLabel}검색 서버 연결 필요`;
    info.append(meta); choice.append(check, info); row.append(choice);
    if (doc.indexed !== null && doc.index_complete !== true && doc.has_pdf) {
      const retry = document.createElement('button');
      retry.type = 'button'; retry.className = 'reindex-doc'; retry.textContent = '색인';
      retry.title = '이 PDF의 전체 페이지를 다시 색인합니다';
      retry.addEventListener('click', () => reindexDocument(doc));
      row.append(retry);
    }
    list.append(row);
  }
  if (!docs.length) {
    const empty = document.createElement('p'); empty.className = 'list-state';
    empty.textContent = state.documents.length ? '일치하는 문서가 없어요.' : '아직 문서가 없어요. 아래에서 PDF를 추가해 주세요.';
    list.append(empty);
  }
  $('doc-count').textContent = state.documents.length;
  const pending = state.documents.filter(doc => doc.indexed !== null && doc.index_complete !== true);
  $('reindex-pending').disabled = state.busy || !pending.length;
  $('reindex-pending').textContent = pending.length ? `색인 필요한 문서 ${pending.length}개 처리` : '모든 문서 색인 완료';
  updateScope();
}
async function loadDocs() {
  $('refresh-docs').disabled = true;
  try {
    const data = await api('/api/documents');
    state.documents = data.documents;
    state.searchAvailable = data.search_available;
    state.selected = new Set([...state.selected].filter(id => state.documents.some(d => d.id === id && d.index_complete === true)));
    renderDocs();
  } catch (error) {
    $('document-list').replaceChildren();
    const p = document.createElement('p'); p.className = 'list-state';
    p.textContent = '문서를 불러오지 못했어요. OpenSearch 연결을 확인하고 ↻를 눌러주세요.';
    $('document-list').append(p);
    toast(error.message);
  } finally { $('refresh-docs').disabled = false; }
}
async function health() {
  try {
    const data = await api('/api/health');
    state.ready = data.search && data.models;
    $('connection').className = 'connection ' + (state.ready ? 'online' : 'offline');
    $('connection-text').textContent = state.ready ? 'Searchdoc 준비 완료' : !data.search ? '검색 연결 필요' : 'Searchdoc 모델 설정 필요';
    $('connection').title = data.missing_models.length ? `필요한 설정: ${data.missing_models.join(', ')}` : `Searchdoc 연결 상태\n답변: ${data.answer_model}\n분석: ${data.analysis_model}\n임베딩: ${data.embed_model}`;
    if (!state.busy && !state.ready) $('activity-text').textContent = '로컬 검색 서버와 모델의 연결을 확인해 주세요.';
    if (!state.busy && state.ready) $('activity-text').textContent = '질문을 기다리고 있어요.';
  } catch {
    state.ready = false;
    $('connection').className = 'connection offline';
    $('connection-text').textContent = '서버 연결 끊김';
  }
  $('send').disabled = state.busy || !state.ready;
}
function pdfUrl(source) {
  return `/api/pdf?doc=${encodeURIComponent(source.doc_id)}#page=${Math.max(1, Number(source.page) || 1)}`;
}
function linkFor(source, label) {
  const link = document.createElement('a');
  link.href = pdfUrl(source); link.target = '_blank'; link.rel = 'noopener'; link.textContent = label;
  return link;
}
function renderText(target, text, citations) {
  // Never interpret model output or PDF text as HTML.
  const parts = text.split(/(\[cite:\s*[\d,\s]+\]|\*\*[^*\n]+\*\*)/g);
  for (const part of parts) {
    if (part.startsWith('[cite:')) {
      for (const n of part.match(/\d+/g) || []) {
        const source = citations.find(c => c.n === Number(n));
        if (!source) continue;
        const link = linkFor(source, n); link.className = 'citation-link';
        link.title = `${source.doc_id} · ${source.page}쪽 원문 열기`;
        link.setAttribute('aria-label', `출처 ${n}, ${source.page}쪽 PDF 열기`);
        target.append(link);
      }
    } else if (part.startsWith('**') && part.endsWith('**')) {
      const strong = document.createElement('strong'); strong.textContent = part.slice(2, -2); target.append(strong);
    } else target.append(document.createTextNode(part));
  }
}
function appendTimingProfile(content, trace, total) {
  const spans = trace.spans || [];
  if (!spans.length) return;
  const definitions = [
    ['bedrock.chat', '생성 모델'],
    ['bedrock.embed', 'Titan 임베딩'],
    ['opensearch', 'OpenSearch'],
    ['pdf', 'PDF 처리'],
  ];
  const totals = new Map(definitions.map(([key]) => [key, 0]));
  for (const item of spans) totals.set(item.category, (totals.get(item.category) || 0) + (Number(item.seconds) || 0));
  const measured = [...totals.values()].reduce((sum, seconds) => sum + seconds, 0);
  const groups = document.createElement('div'); groups.className = 'timing-groups';
  const title = document.createElement('strong'); title.textContent = '시간 사용처'; groups.append(title);
  for (const [key, label] of [...definitions, ['local', '로컬 처리·기타']]) {
    const seconds = key === 'local' ? Math.max(0, total - measured) : totals.get(key) || 0;
    const row = document.createElement('div'); row.className = 'timing-row';
    const name = document.createElement('span'); name.textContent = label;
    const track = document.createElement('i');
    const bar = document.createElement('b'); bar.style.width = `${total ? Math.min(100, seconds / total * 100) : 0}%`; track.append(bar);
    const value = document.createElement('em'); value.textContent = `${seconds.toFixed(2)}초 · ${total ? Math.round(seconds / total * 100) : 0}%`;
    row.append(name, track, value); groups.append(row);
  }
  content.append(groups);

  const calls = document.createElement('details'); calls.className = 'timing-calls';
  const summary = document.createElement('summary');
  const inputTokens = spans.reduce((sum, item) => sum + (Number(item.input_tokens) || 0), 0);
  const outputTokens = spans.reduce((sum, item) => sum + (Number(item.output_tokens) || 0), 0);
  summary.textContent = `외부 호출 ${spans.length}건 상세 · 토큰 ${inputTokens.toLocaleString()} → ${outputTokens.toLocaleString()}`;
  const list = document.createElement('ol');
  for (const item of spans) {
    const row = document.createElement('li');
    const task = item.task || item.name;
    const extras = [];
    if (item.input_tokens || item.output_tokens) extras.push(`토큰 ${item.input_tokens || 0}→${item.output_tokens || 0}`);
    if (item.server_latency_ms) extras.push(`서버 ${(item.server_latency_ms / 1000).toFixed(2)}초`);
    if (item.leg) extras.push(item.leg);
    if (item.page) extras.push(`${item.page}쪽`);
    row.textContent = `[${item.stage}] ${task} · ${(Number(item.seconds) || 0).toFixed(3)}초${extras.length ? ` · ${extras.join(' · ')}` : ''}`;
    list.append(row);
  }
  calls.append(summary, list); content.append(calls);
  const cached = (trace.events || []).filter(item => item.cache_hit).map(item => item.task);
  if (cached.length) {
    const note = document.createElement('p');
    note.textContent = `캐시 적중: ${cached.join(', ')}`;
    content.append(note);
  }
}
function renderMessage(message) {
  $('welcome').hidden = true;
  const log = $('chat-log');
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 100;
  const article = document.createElement('article'); article.className = `message ${message.role}`;
  const header = document.createElement('div'); header.className = 'message-header';
  if (message.role === 'assistant') { const img = document.createElement('img'); img.src = petSource(); img.className = 'cat-avatar'; img.alt = ''; header.append(img); }
  const name = document.createElement('strong'); name.textContent = message.role === 'user' ? '나' : 'HAEINDEX';
  const time = document.createElement('time'); time.textContent = message.time;
  header.append(name, time);
  const body = document.createElement('div'); body.className = 'message-body';
  renderText(body, message.text, message.citations || []);
  article.append(header, body);
  if (message.citations?.length) {
    const sources = document.createElement('div'); sources.className = 'sources';
    for (const source of message.citations) {
      const card = document.createElement('div'); card.className = 'source-card';
      const doc = state.documents.find(d => d.id === source.doc_id);
      card.append(linkFor(source, `[${source.n}] ${doc?.name || source.doc_id} · ${source.page}쪽 ↗`));
      const details = document.createElement('details');
      const summary = document.createElement('summary'); summary.textContent = source.origin === 'vision_transcription' ? '이미지에서 읽은 내용 보기 · PDF 원본 확인 권장' : '인용 원문 보기';
      const pre = document.createElement('pre'); pre.textContent = source.text;
      details.append(summary, pre); card.append(details); sources.append(card);
    }
    article.append(sources);
  }
  if (message.meta) { const meta = document.createElement('div'); meta.className = 'message-meta'; meta.textContent = message.meta; article.append(meta); }
  if (message.trace?.stages?.length) {
    const details = document.createElement('details'); details.className = 'process-trace';
    const summary = document.createElement('summary');
    const total = message.trace.stages.reduce((sum, stage) => sum + (Number(stage.seconds) || 0), 0);
    summary.textContent = `Agent 처리 과정 보기 · ${message.trace.stages.length}단계 · ${total.toFixed(1)}초`;
    const content = document.createElement('div');
    const plan = message.trace.plan || {};
    if (plan.requirements?.length) {
      const p = document.createElement('p'); p.textContent = `질문 해석: ${plan.requirements.join(' · ')}`; content.append(p);
    }
    if (message.trace.documents?.length) {
      const p = document.createElement('p'); p.textContent = `검색 문서: ${message.trace.documents.join(', ')}`; content.append(p);
    }
    const list = document.createElement('ol');
    for (const stage of message.trace.stages) {
      const item = document.createElement('li');
      const seconds = Number(stage.seconds) || 0;
      const label = document.createElement('span'); label.textContent = stage.detail;
      const track = document.createElement('i');
      const bar = document.createElement('b'); bar.style.width = `${total ? Math.min(100, seconds / total * 100) : 0}%`; track.append(bar);
      const value = document.createElement('em'); value.textContent = seconds < .05 ? '<0.1초' : `${seconds.toFixed(1)}초`;
      item.append(label, track, value);
      list.append(item);
    }
    content.append(list);
    if (message.trace.assessments?.length) {
      const checks = document.createElement('details');
      const summary = document.createElement('summary'); summary.textContent = '근거 검사 결과';
      const results = document.createElement('ol');
      const reasons = {not_found: '해당 내용 미발견', missing_context: '이어지는 근거 필요', table_layout: '표의 행·열 확인 필요', diagram: '도식 확인 필요', unreadable_text: '문자 손상 확인 필요'};
      for (const check of message.trace.assessments) {
        const row = document.createElement('li');
        const supported = (check.supports || []).map(item => item.requirement);
        const missing = (check.missing || []).map(requirement => {
          const gap = (check.gaps || []).find(item => item.requirement === requirement);
          return `${requirement}${gap ? ` (${reasons[gap.reason] || gap.reason})` : ''}`;
        });
        row.textContent = [supported.length ? `확인: ${supported.join(' · ')}` : '', missing.length ? `미확인: ${missing.join(' · ')}` : '', check.clarification || ''].filter(Boolean).join(' / ');
        results.append(row);
      }
      checks.append(summary, results); content.append(checks);
    }
    if (message.trace.events?.length) {
      const calls = message.trace.events.filter(event => !event.cache_hit);
      const modelTime = calls.reduce((sum, event) => sum + (Number(event.seconds) || 0), 0);
      const p = document.createElement('p');
      p.textContent = `모델 호출 ${calls.length}회 ${modelTime.toFixed(1)}초 · 캐시 ${message.trace.cache_hits || 0}회`;
      content.append(p);
    }
    appendTimingProfile(content, message.trace, total);
    details.append(summary, content); article.append(details);
  }
  $('messages').append(article);
  if (nearBottom || message.role === 'user') log.scrollTop = log.scrollHeight;
}
function addMessage(role, text, citations = [], meta = '', trace = null) {
  const message = {role, text, citations, meta, trace, time: new Date().toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit'})};
  state.messages.push(message); renderMessage(message); remember();
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function pollJob(id, kind) {
  state.active = {id, kind}; remember(); setBusy(true);
  let failures = 0;
  while (state.active?.id === id) {
    try {
      const job = await api(`/api/jobs/${id}`);
      failures = 0;
      const latest = job.events.at(-1);
      $('activity-text').textContent = latest?.detail || 'HAEINDEX가 준비하고 있어요.';
      $('elapsed').textContent = latest ? `이 단계 ${(latest.seconds || 0).toFixed(1)}초 · 전체 ${Math.round(job.seconds)}초` : `${Math.round(job.seconds)}초`;
      if (job.status === 'complete' || job.status === 'error') {
        if (job.status === 'error') {
          addMessage('assistant', '처리 중 문제가 생겼어요. 연결 상태를 확인하고 다시 시도해 주세요.', [], job.error);
          $('activity-text').textContent = '작업을 완료하지 못했어요. 다시 시도할 수 있어요.';
          if (kind !== 'ask') await loadDocs();
        } else if (kind === 'ask') {
          const result = job.result;
          const trace = {plan: result.plan, documents: result.documents, stages: result.stages, assessments: result.assessments, events: result.events, spans: result.spans, cache_hits: result.cache_hits};
          addMessage('assistant', result.message, result.citations,
            result.error ? `처리 오류: ${result.error}` : `${Math.round(job.seconds)}초 · ${result.clarification ? '문서 확인 필요' : result.answer.refusal === 'not_found' ? '해당 내용 미발견' : result.answer.refusal ? '근거 확인 필요' : '원문 대조 완료'}`,
            trace);
          $('activity-text').textContent = result.answer.refusal || result.clarification ? '질문이나 문서를 더 구체적으로 지정해 보세요.' : '답변을 받았어요. 출처도 함께 확인해 보세요.';
        } else {
          addMessage('assistant', job.result.message);
          if (job.result.doc_id) state.selected = new Set([job.result.doc_id]);
          await loadDocs();
          $('activity-text').textContent = kind === 'enhance' ? '검색 보강이 끝났어요.' : '전체 원문 색인이 끝났어요. 이제 질문해 보세요.';
        }
        state.active = null; remember(); setBusy(false); $('question').focus(); return;
      }
    } catch (error) {
      if (error.status === 404) {
        state.active = null; remember(); setBusy(false);
        $('activity-text').textContent = '이전 작업을 찾을 수 없어요. 질문을 다시 보내 주세요.';
        toast('서버가 다시 시작되었을 수 있어요. 대화 내용은 유지됩니다.');
        return;
      }
      failures += 1;
      $('activity-text').textContent = '연결을 다시 확인하고 있어요…';
      if (failures >= 5) {
        toast('연결이 끊겼어요. 화면을 새로고침하면 진행 중인 작업을 다시 확인합니다.');
        setBusy(false); return;
      }
    }
    await pause(failures ? 3000 : 1000);
  }
}
$('chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || state.busy || !state.ready) return;
  setBusy(true);
  try {
    const job = await jsonPost('/api/ask', {question, documents: [...state.selected]});
    addMessage('user', question, [], state.selected.size ? `선택한 문서 ${state.selected.size}개` : '전체 문서');
    $('question').value = ''; updateCount();
    await pollJob(job.id, 'ask');
  } catch (error) { toast(error.message); setBusy(false); }
});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault(); $('chat-form').requestSubmit();
  }
});
function updateCount() { $('char-count').textContent = `${$('question').value.length.toLocaleString()} / 2,000`; }
$('question').addEventListener('input', updateCount);
for (const button of document.querySelectorAll('[data-question]')) button.addEventListener('click', () => {
  $('question').value = button.dataset.question; updateCount(); $('question').focus();
});
$('insert-face').addEventListener('click', () => {
  const input = $('question'); if (input.value.length > 1980) return;
  input.setRangeText(' ฅ^•ﻌ•^ฅ ', input.selectionStart, input.selectionEnd, 'end'); updateCount(); input.focus();
});
$('doc-filter').addEventListener('input', renderDocs);
$('refresh-docs').addEventListener('click', () => { loadDocs(); health(); });
$('all-docs').addEventListener('click', () => { state.selected.clear(); renderDocs(); });
function toggleDocs(open = $('sidebar').hidden) {
  $('sidebar').hidden = !open;
  document.querySelector('.workspace').classList.toggle('no-sidebar', !open);
  $('toggle-docs').setAttribute('aria-expanded', String(open));
}
$('toggle-docs').addEventListener('click', () => toggleDocs());
if (matchMedia('(max-width: 760px)').matches) toggleDocs(false);
$('new-chat').addEventListener('click', () => {
  if (state.busy) return;
  state.messages = []; $('messages').replaceChildren(); $('welcome').hidden = false;
  $('activity-text').textContent = '새 대화를 시작했어요.'; $('elapsed').textContent = ''; remember(); $('question').focus();
});
$('save-chat').addEventListener('click', () => {
  if (!state.messages.length) { toast('저장할 대화가 아직 없어요.'); return; }
  const text = state.messages.map(m => `${m.role === 'user' ? '나' : 'HAEINDEX'} (${m.time})\n${m.text}\n${m.citations.map(c => `[${c.n}] ${c.doc_id} · p.${c.page}\n${c.text}`).join('\n')}`).join('\n\n');
  const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `HAEINDEX-${new Date().toISOString().slice(0, 10)}.txt`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function reindexDocument(doc) {
  if (state.busy || doc.index_complete === true || doc.indexed === null) return;
  setBusy(true); $('activity-text').textContent = `${doc.name} 재색인을 준비하고 있어요.`;
  try {
    const job = await jsonPost('/api/reindex', {documents: [doc.id]});
    addMessage('user', `문서 재색인: ${doc.name}`);
    await pollJob(job.id, 'reindex');
  } catch (error) { toast(error.message); setBusy(false); }
}
$('reindex-pending').addEventListener('click', async () => {
  const documents = state.documents.filter(doc => doc.indexed !== null && doc.index_complete !== true).map(doc => doc.id);
  if (state.busy || !documents.length) return;
  setBusy(true); $('activity-text').textContent = `문서 ${documents.length}개 전체 재색인을 준비하고 있어요.`;
  try {
    const job = await jsonPost('/api/reindex', {documents});
    addMessage('user', `색인 필요한 문서 ${documents.length}개 전체 처리`);
    await pollJob(job.id, 'reindex');
  } catch (error) { toast(error.message); setBusy(false); }
});
for (const id of ['add-pdf', 'attach', 'file-menu']) $(id).addEventListener('click', () => $('pdf-input').click());
$('pdf-input').addEventListener('change', async () => {
  const files = [...$('pdf-input').files]; $('pdf-input').value = '';
  if (!files.length || state.busy) return;
  const valid = files.filter(file => file.name.toLowerCase().endsWith('.pdf') && file.size <= 32 * 1024 * 1024);
  if (valid.length !== files.length) toast(`32MB 이하 PDF만 처리합니다. 제외 ${files.length - valid.length}개`);
  for (const [index, file] of valid.entries()) {
    setBusy(true);
    $('activity-text').textContent = `PDF ${index + 1}/${valid.length} · ${file.name} 전달 중`;
    try {
      const job = await api('/api/upload', {method: 'POST', headers: {'Content-Type': 'application/pdf', 'X-Filename': encodeURIComponent(file.name)}, body: file});
      addMessage('user', `문서 추가 (${index + 1}/${valid.length}): ${file.name}`);
      await pollJob(job.id, 'upload');
    } catch (error) {
      addMessage('assistant', `${file.name}을 추가하지 못했어요.`, [], error.message);
      setBusy(false);
    }
  }
});
$('enhance-docs').addEventListener('click', async () => {
  if (state.busy || !state.selected.size) return;
  setBusy(true);
  try {
    const job = await jsonPost('/api/enhance', {documents: [...state.selected]});
    addMessage('user', `선택한 문서 ${state.selected.size}개 검색 보강`); await pollJob(job.id, 'enhance');
  } catch (error) { toast(error.message); setBusy(false); }
});
$('help-menu').addEventListener('click', () => $('help-dialog').showModal());
for (const id of ['close-help', 'help-done']) $(id).addEventListener('click', () => $('help-dialog').close());
for (const id of ['minimize', 'close-window']) $(id).addEventListener('click', () => {
  $('window').hidden = true; $('restore').focus(); toast('아래 HAEINDEX 버튼을 누르면 다시 열려요.');
});
$('restore').addEventListener('click', () => { $('window').hidden = false; $('question').focus(); });
$('maximize').addEventListener('click', () => $('window').classList.toggle('maximized'));
function clock() { $('clock').textContent = new Date().toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit'}); }
clock(); setInterval(clock, 10000);
// A small, non-interactive companion. Animation stops when it reaches the pointer.
const cat = $('cursor-cat');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = matchMedia('(pointer: fine)');
let catEnabled = true;
try { catEnabled = localStorage.getItem('haeindex-cat') !== 'off'; } catch { /* optional preference */ }
const catKinds = new Set(['black', 'cheese', 'tabby', 'munchkin', 'searchdog']);
let catKind = 'cheese';
try { const saved = localStorage.getItem('haeindex-cat-kind'); if (catKinds.has(saved)) catKind = saved; } catch { /* optional preference */ }
function petSource() { return catKind === 'searchdog' ? '/searchdog.svg' : `/cat-${catKind}.svg`; }
function petName() { return catKind === 'searchdog' ? '서치독' : '고양이'; }
let x = -100, y = -100, tx = 0, ty = 0, frame = 0, last = 0;
function catAllowed() { return catEnabled && !reducedMotion.matches && finePointer.matches; }
function hideCat() { cancelAnimationFrame(frame); frame = 0; cat.style.visibility = 'hidden'; cat.classList.remove('walking'); }
function catFrame(now) {
  const dt = Math.min(40, now - last || 16); last = now;
  const dx = tx - x, dy = ty - y, distance = Math.hypot(dx, dy);
  const ease = 1 - Math.exp(-dt / 110);
  x += dx * ease; y += dy * ease;
  cat.style.transform = `translate3d(${x}px,${y}px,0)`;
  cat.classList.toggle('walking', distance > 8);
  if (Math.abs(dx) > 2) cat.querySelector('img').style.transform = dx < 0 ? 'scaleX(-1)' : '';
  if (distance > .7 && catAllowed()) frame = requestAnimationFrame(catFrame); else frame = 0;
}
document.addEventListener('pointermove', event => {
  if (event.target.closest('.cat-care, .club-title')) { hideCat(); return; }
  if (!catAllowed() || event.pointerType === 'touch') return;
  tx = Math.max(0, Math.min(innerWidth - 49, event.clientX + 18));
  ty = Math.max(0, Math.min(innerHeight - 49, event.clientY + 19));
  if (cat.style.visibility !== 'visible') { x = tx + 25; y = ty + 15; }
  cat.style.visibility = 'visible';
  if (!frame) { last = performance.now(); frame = requestAnimationFrame(catFrame); }
}, {passive: true});
document.addEventListener('pointerdown', event => {
  if (event.target.closest('.cat-care, .club-title')) { hideCat(); return; }
  if (!catAllowed() || event.pointerType === 'touch') return;
  cat.classList.add('meow'); clearTimeout(cat.meowTimer);
  cat.meowTimer = setTimeout(() => cat.classList.remove('meow'), 650);
}, {passive: true});
document.documentElement.addEventListener('pointerleave', hideCat);
window.addEventListener('blur', hideCat);
document.addEventListener('visibilitychange', () => { if (document.hidden) hideCat(); });
reducedMotion.addEventListener('change', hideCat); finePointer.addEventListener('change', hideCat);
const careStage = $('care-cat');
const careReaction = $('care-reaction');
const careMessages = {
  fish: ['생선 맛있다 🐟', '하나 더 먹고싶다', '나는 참치가 좋아', '배고프다', '더 줘!', '바다 맛 좋아'],
  kibble: ['사료를 오독오독 먹었어요.', '사료보다는 생선이 더 좋아', '사료는 맛없어', '사료는 조금만 줘', '사료는 싫어', '사료는 별로야'],
  pet: ['기분 좋다~', '골골골…', '한 번만 더 쓰다듬어줘!','좋아 좋아~', '쓰다듬는 손길이 좋아', '쓰다듬는 게 최고야', '나랑 평생 함께하자', '나랑 같이 놀거지?', '나랑만 놀자~'],
};
const dogCareMessages = {
  fish: ['냠냠! 맛있다 🐟', '나는 고기가 더 좋아', '간식 먹고 산책 가자!'],
  kibble: ['오독오독, 사료 맛있다!', '멍멍! 한 그릇 뚝딱!', '밥 먹고 힘이 났어 🐾'],
  pet: ['멍! 기분 좋아!', '한 번만 더 쓰다듬어줘!', '더 일해야지','RAG 잘 알아?', '인덱싱 좋아해?'],
};
const careSymbols = {fish: '🐟', kibble: '🍚', pet: '🫳🏻'};
const catRoom = $('cat-room');
const roomBowl = $('room-bowl');
const roomDestination = $('room-destination');
const roomPause = $('room-pause');
const catHomeCaption = $('room-caption').textContent;
const homeTitle = document.querySelector('.club-title').firstChild;
const catHomeTitle = homeTitle.textContent;
let roomX = .43, roomY = .83, roomTarget = null, roomFrameId = 0, roomLast = 0;
let roomVisible = false, roomPaused = false, wanderTimer = 0, roomAction = '';
let careTimer = 0;

function roomCanMove() { return roomVisible && !document.hidden && !reducedMotion.matches; }
function roomPosition(x, y) {
  const edge = Math.min(.3, 33 / (catRoom.clientWidth || 230));
  return {x: Math.max(edge, Math.min(1 - edge, x)), y: Math.max(.67, Math.min(.95, y))};
}
function placeRoomCat(x, y) {
  ({x: roomX, y: roomY} = roomPosition(x, y));
  careStage.style.setProperty('--room-x', `${roomX * 100}%`);
  careStage.style.setProperty('--room-y', `${roomY * 100}%`);
}
function stopRoomWalk(clearTarget = true) {
  cancelAnimationFrame(roomFrameId); roomFrameId = 0;
  clearTimeout(wanderTimer);
  careStage.classList.remove('walking');
  if (clearTarget) { roomTarget = null; roomDestination.classList.remove('visible'); }
}
function planRoomWander() {
  clearTimeout(wanderTimer);
  if (!roomCanMove() || roomPaused || roomTarget || roomAction || catRoom.matches(':focus-visible') || catRoom.querySelector(':focus-visible')) return;
  wanderTimer = setTimeout(() => walkRoomCat(.16 + Math.random() * .68, .69 + Math.random() * .23), 2400 + Math.random() * 2500);
}
function roomArrived() {
  const onArrival = roomTarget?.onArrival;
  stopRoomWalk();
  if (onArrival) onArrival(); else planRoomWander();
}
function animateRoomCat(now) {
  roomFrameId = 0;
  if (!roomTarget || !roomCanMove()) { careStage.classList.remove('walking'); return; }
  const dt = Math.min(40, now - roomLast || 16); roomLast = now;
  const dx = (roomTarget.x - roomX) * catRoom.clientWidth;
  const dy = (roomTarget.y - roomY) * catRoom.clientHeight;
  const distance = Math.hypot(dx, dy), step = 60 * dt / 1000;
  if (distance <= Math.max(1, step)) {
    placeRoomCat(roomTarget.x, roomTarget.y); roomArrived(); return;
  }
  careStage.classList.add('walking');
  if (Math.abs(dx) > 1) careStage.style.setProperty('--cat-facing', dx < 0 ? '-1' : '1');
  placeRoomCat(roomX + (roomTarget.x - roomX) * step / distance, roomY + (roomTarget.y - roomY) * step / distance);
  roomFrameId = requestAnimationFrame(animateRoomCat);
}
function walkRoomCat(x, y, onArrival = null) {
  stopRoomWalk();
  roomTarget = {...roomPosition(x, y), onArrival};
  if (reducedMotion.matches) {
    placeRoomCat(roomTarget.x, roomTarget.y); roomArrived();
  } else if (roomCanMove()) {
    roomLast = performance.now(); roomFrameId = requestAnimationFrame(animateRoomCat);
  }
}
function clearCareReaction() {
  clearTimeout(careTimer);
  careStage.classList.remove('eating', 'petted');
  roomBowl.classList.remove('filled');
  for (const button of document.querySelectorAll('.cat-care-actions button')) button.classList.remove('active');
  roomAction = '';
}
function showCareReaction(action) {
  const messages = (catKind === 'searchdog' ? dogCareMessages : careMessages)[action];
  $('cat-mood').textContent = messages[Math.floor(Math.random() * messages.length)];
  careReaction.textContent = action === 'pet' ? '♥' : '♪';
  careStage.classList.add(action === 'pet' ? 'petted' : 'eating');
  careTimer = setTimeout(() => { clearCareReaction(); planRoomWander(); }, 2200);
}
function careForCat(action) {
  if (!careMessages[action]) return;
  stopRoomWalk(); clearCareReaction(); roomAction = action;
  document.querySelector(`.cat-care-actions [data-care="${action}"]`).classList.add('active');
  if (action === 'pet') { showCareReaction(action); return; }
  $('room-food').textContent = action === 'fish' ? careSymbols.fish : '● ∴ ●';
  roomBowl.classList.add('filled');
  $('cat-mood').textContent = action === 'fish' ? '생선 냄새다! 지금 갈게 🐾' : '밥 먹으러 가는 중… 🐾';
  walkRoomCat(.64, .89, () => {
    careStage.style.setProperty('--cat-facing', '1'); showCareReaction(action);
  });
}
careStage.addEventListener('click', () => careForCat('pet'));
for (const button of document.querySelectorAll('[data-care]')) {
  button.addEventListener('click', () => careForCat(button.dataset.care));
}
catRoom.addEventListener('click', event => {
  if (event.target.closest('button')) return;
  clearCareReaction();
  const rect = catRoom.getBoundingClientRect();
  const point = roomPosition((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height);
  $('cat-mood').textContent = '불렀어? 그쪽으로 갈게';
  walkRoomCat(point.x, point.y);
  if (roomTarget) {
    roomDestination.style.left = `${point.x * 100}%`; roomDestination.style.top = `${point.y * 100}%`;
    roomDestination.classList.add('visible');
  }
});
catRoom.addEventListener('keydown', event => {
  if (event.target !== catRoom) return;
  const directions = {ArrowLeft: [-.12, 0], ArrowRight: [.12, 0], ArrowUp: [0, -.08], ArrowDown: [0, .08]};
  if (!directions[event.key]) return;
  event.preventDefault(); clearCareReaction();
  const [dx, dy] = directions[event.key]; walkRoomCat(roomX + dx, roomY + dy);
});
catRoom.addEventListener('focusin', event => { if (!roomAction && event.target.matches(':focus-visible')) stopRoomWalk(); });
catRoom.addEventListener('focusout', () => setTimeout(planRoomWander, 0));
roomPause.addEventListener('click', () => {
  roomPaused = !roomPaused;
  roomPause.setAttribute('aria-pressed', String(roomPaused));
  roomPause.textContent = roomPaused ? '▶ 산책 시작' : 'Ⅱ 산책 멈춤';
  roomPause.setAttribute('aria-label', `${petName()} 자동 산책 ${roomPaused ? '시작하기' : '멈추기'}`);
  if (roomPaused && !roomAction) stopRoomWalk(); else planRoomWander();
});
function syncRoomMotion() {
  if (!roomVisible || document.hidden) { stopRoomWalk(false); return; }
  roomPause.disabled = reducedMotion.matches;
  if (reducedMotion.matches) {
    stopRoomWalk(false);
    if (roomTarget) { placeRoomCat(roomTarget.x, roomTarget.y); roomArrived(); }
  } else if (roomTarget && !roomFrameId) {
    roomLast = performance.now(); roomFrameId = requestAnimationFrame(animateRoomCat);
  } else planRoomWander();
}
new IntersectionObserver(entries => {
  roomVisible = entries[0].isIntersecting && entries[0].intersectionRatio > .25;
  syncRoomMotion();
}, {threshold: [0, .25]}).observe(catRoom);
document.addEventListener('visibilitychange', syncRoomMotion);
reducedMotion.addEventListener('change', syncRoomMotion);
new ResizeObserver(() => placeRoomCat(roomX, roomY)).observe(catRoom);
placeRoomCat(roomX, roomY);
function catPreference() {
  const source = petSource(), name = petName(), isDog = catKind === 'searchdog';
  for (const image of document.querySelectorAll('img[src="/cat.svg"], img.cat-avatar')) {
    image.classList.add('cat-avatar'); image.src = source;
  }
  $('cat-menu').textContent = `친구(C) ${catEnabled ? '✓' : '·'} ▾`;
  $('cat-state').textContent = catEnabled ? (isDog ? '🐾' : 'ฅ') : '·';
  $('cat-state').title = `포인터를 따라오는 ${name}`;
  cat.querySelector('span').textContent = isDog ? '멍멍!' : '야옹!';
  homeTitle.textContent = isDog ? "SEARCHDOG'S HOME " : catHomeTitle;
  $('room-caption').textContent = isDog ? 'Searchdog 집 · 서치독' : catHomeCaption;
  careStage.setAttribute('aria-label', `${name} 쓰다듬기`);
  catRoom.setAttribute('aria-label', `${name} 방. 바닥을 누르거나 방향키로 ${name}을 불러 보세요.`);
  roomPause.setAttribute('aria-label', `${name} 자동 산책 ${roomPaused ? '시작하기' : '멈추기'}`);
  document.querySelector('.cat-care-actions').setAttribute('aria-label', `${name} 돌보기 메뉴`);
  document.querySelector('[data-care="pet"] span').textContent = isDog ? '🐶' : '😻';
  $('cat-enabled').checked = catEnabled;
  for (const button of document.querySelectorAll('[data-cat]')) {
    const selected = button.dataset.cat === catKind;
    button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected));
  }
  if (!catEnabled) hideCat();
}
$('cat-menu').addEventListener('click', () => {
  $('cat-dialog').showModal();
});
for (const button of document.querySelectorAll('[data-cat]')) button.addEventListener('click', () => {
  stopRoomWalk(); clearCareReaction();
  catKind = button.dataset.cat; catEnabled = true; catPreference();
  $('cat-mood').textContent = catKind === 'searchdog' ? '멍! 나는 서치독이야. 같이 놀자 🐾' : '같이 놀자! 무얼 해줄 거야?';
  planRoomWander();
  try { localStorage.setItem('haeindex-cat-kind', catKind); localStorage.setItem('haeindex-cat', 'on'); } catch { /* optional preference */ }
});
$('cat-enabled').addEventListener('change', event => {
  catEnabled = event.target.checked; catPreference();
  try { localStorage.setItem('haeindex-cat', catEnabled ? 'on' : 'off'); } catch { /* optional preference */ }
});
for (const id of ['close-cat', 'cat-done']) $(id).addEventListener('click', () => $('cat-dialog').close());
catPreference();
if (catKind === 'searchdog') $('cat-mood').textContent = '멍! 나는 서치독이야. 같이 놀자 🐾';
async function init() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(sessionKey) || 'null');
    if (saved && Array.isArray(saved.messages) && Array.isArray(saved.selected)) {
      state.messages = saved.messages; state.selected = new Set(saved.selected); state.active = saved.active;
    }
  } catch { /* Start a fresh conversation if stored data is invalid. */ }
  await Promise.all([loadDocs(), health()]);
  for (const message of state.messages) renderMessage(message);
  if (state.active) pollJob(state.active.id, state.active.kind);
  setInterval(health, 30000);
}
init();
