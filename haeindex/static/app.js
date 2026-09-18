'use strict';
const $ = id => document.getElementById(id);
const state = {documents: [], selected: new Set(), messages: [], busy: false, ready: false, active: null};
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
    const row = document.createElement('label');
    row.className = 'doc-item' + (state.selected.has(doc.id) ? ' chosen' : '');
    row.title = doc.name;
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = state.selected.has(doc.id);
    check.addEventListener('change', () => {
      if (check.checked) state.selected.add(doc.id); else state.selected.delete(doc.id);
      row.classList.toggle('chosen', check.checked);
      updateScope();
    });
    const info = document.createElement('span');
    info.className = 'doc-info';
    info.append(document.createTextNode(doc.name.replace(/\.pdf$/i, '')));
    const meta = document.createElement('small');
    meta.textContent = `${doc.pages ? `${doc.pages}쪽 · ` : ''}${doc.chunks}개 문단`;
    info.append(meta); row.append(check, info); list.append(row);
  }
  if (!docs.length) {
    const empty = document.createElement('p'); empty.className = 'list-state';
    empty.textContent = state.documents.length ? '일치하는 문서가 없어요.' : '아직 문서가 없어요. 아래에서 PDF를 추가해 주세요.';
    list.append(empty);
  }
  $('doc-count').textContent = state.documents.length;
  updateScope();
}
async function loadDocs() {
  $('refresh-docs').disabled = true;
  try {
    const data = await api('/api/documents');
    state.documents = data.documents;
    state.selected = new Set([...state.selected].filter(id => state.documents.some(d => d.id === id)));
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
    $('connection-text').textContent = state.ready ? 'Bedrock 설정 완료' : !data.search ? 'OpenSearch 연결 필요' : 'Bedrock 모델 설정 필요';
    $('connection').title = data.missing_models.length ? `필요한 설정: ${data.missing_models.join(', ')}` : `공급자: AWS Bedrock\n답변: ${data.answer_model}\n분석: ${data.analysis_model}\n임베딩: ${data.embed_model}`;
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
function renderMessage(message) {
  $('welcome').hidden = true;
  const log = $('chat-log');
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 100;
  const article = document.createElement('article'); article.className = `message ${message.role}`;
  const header = document.createElement('div'); header.className = 'message-header';
  if (message.role === 'assistant') { const img = document.createElement('img'); img.src = '/cat.svg'; img.alt = ''; header.append(img); }
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
      item.textContent = `${stage.detail} — ${seconds < .05 ? '<0.1' : seconds.toFixed(1)}초`;
      list.append(item);
    }
    content.append(list);
    if (message.trace.events?.length) {
      const calls = message.trace.events.filter(event => !event.cache_hit);
      const modelTime = calls.reduce((sum, event) => sum + (Number(event.seconds) || 0), 0);
      const p = document.createElement('p');
      p.textContent = `모델 호출 ${calls.length}회 ${modelTime.toFixed(1)}초 · 캐시 ${message.trace.cache_hits || 0}회`;
      content.append(p);
    }
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
        } else if (kind === 'ask') {
          const result = job.result;
          const trace = {plan: result.plan, documents: result.documents, stages: result.stages, events: result.events, cache_hits: result.cache_hits};
          addMessage('assistant', result.message, result.citations,
            result.error ? `처리 오류: ${result.error}` : `${Math.round(job.seconds)}초 · ${result.clarification ? '문서 확인 필요' : result.answer.refusal ? '근거 확인 필요' : '원문 대조 완료'}`,
            trace);
          $('activity-text').textContent = result.answer.refusal || result.clarification ? '질문이나 문서를 더 구체적으로 지정해 보세요.' : '답변을 받았어요. 출처도 함께 확인해 보세요.';
        } else {
          addMessage('assistant', job.result.message);
          if (job.result.doc_id) state.selected = new Set([job.result.doc_id]);
          await loadDocs();
          $('activity-text').textContent = '문서 준비가 끝났어요. 이제 질문해 보세요.';
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
for (const id of ['add-pdf', 'attach', 'file-menu']) $(id).addEventListener('click', () => $('pdf-input').click());
$('pdf-input').addEventListener('change', async () => {
  const file = $('pdf-input').files[0]; $('pdf-input').value = '';
  if (!file || state.busy) return;
  if (!file.name.toLowerCase().endsWith('.pdf') || file.size > 32 * 1024 * 1024) { toast('32MB 이하 PDF 파일을 선택해 주세요.'); return; }
  setBusy(true); $('activity-text').textContent = 'PDF를 전달하고 있어요.';
  try {
    const job = await api('/api/upload', {method: 'POST', headers: {'Content-Type': 'application/pdf', 'X-Filename': encodeURIComponent(file.name)}, body: file});
    addMessage('user', `문서 추가: ${file.name}`); await pollJob(job.id, 'upload');
  } catch (error) { toast(error.message); setBusy(false); }
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
const catKinds = new Set(['black', 'cheese', 'tabby', 'munchkin']);
let catKind = 'cheese';
try { const saved = localStorage.getItem('haeindex-cat-kind'); if (catKinds.has(saved)) catKind = saved; } catch { /* optional preference */ }
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
  if (!catAllowed() || event.pointerType === 'touch') return;
  tx = Math.max(0, Math.min(innerWidth - 49, event.clientX + 18));
  ty = Math.max(0, Math.min(innerHeight - 49, event.clientY + 19));
  if (cat.style.visibility !== 'visible') { x = tx + 25; y = ty + 15; }
  cat.style.visibility = 'visible';
  if (!frame) { last = performance.now(); frame = requestAnimationFrame(catFrame); }
}, {passive: true});
document.addEventListener('pointerdown', event => {
  if (!catAllowed() || event.pointerType === 'touch') return;
  cat.classList.add('meow'); clearTimeout(cat.meowTimer);
  cat.meowTimer = setTimeout(() => cat.classList.remove('meow'), 650);
}, {passive: true});
document.documentElement.addEventListener('pointerleave', hideCat);
window.addEventListener('blur', hideCat);
document.addEventListener('visibilitychange', () => { if (document.hidden) hideCat(); });
reducedMotion.addEventListener('change', hideCat); finePointer.addEventListener('change', hideCat);
function catPreference() {
  const source = `/cat-${catKind}.svg`;
  for (const image of document.querySelectorAll('img[src="/cat.svg"], img.cat-avatar')) {
    image.classList.add('cat-avatar'); image.src = source;
  }
  $('cat-menu').textContent = `고양이(C) ${catEnabled ? '✓' : '·'} ▾`;
  $('cat-state').textContent = catEnabled ? 'ฅ' : '·';
  $('cat-enabled').checked = catEnabled;
  for (const button of document.querySelectorAll('[data-cat]')) button.classList.toggle('selected', button.dataset.cat === catKind);
  if (!catEnabled) hideCat();
}
$('cat-menu').addEventListener('click', () => {
  $('cat-dialog').showModal();
});
for (const button of document.querySelectorAll('[data-cat]')) button.addEventListener('click', () => {
  catKind = button.dataset.cat; catEnabled = true; catPreference();
  try { localStorage.setItem('haeindex-cat-kind', catKind); localStorage.setItem('haeindex-cat', 'on'); } catch { /* optional preference */ }
});
$('cat-enabled').addEventListener('change', event => {
  catEnabled = event.target.checked; catPreference();
  try { localStorage.setItem('haeindex-cat', catEnabled ? 'on' : 'off'); } catch { /* optional preference */ }
});
for (const id of ['close-cat', 'cat-done']) $(id).addEventListener('click', () => $('cat-dialog').close());
catPreference();
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
