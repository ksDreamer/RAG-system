const $ = (id) => document.getElementById(id);
let busy = false;
let library = [];
let selectedDocuments = null;
let readerState = null;
function element(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
function notice(message, error = false) { $('notice').textContent = message; $('notice').className = error ? 'error' : ''; }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed. Check your input and try again.');
  return data;
}
function updateControls() {
  const count = selectedDocuments === null ? library.length : selectedDocuments.size;
  $('scope-label').textContent = selectedDocuments === null ? `All ${count} documents` : `${count} of ${library.length} documents selected`;
  $('ask-button').disabled = busy || count === 0;
  $('all-documents').disabled = busy || selectedDocuments === null;
  $('reader-previous').disabled = busy || !readerState || readerState.offset === 0;
  $('reader-next').disabled = busy || !readerState || readerState.offset + readerState.limit >= readerState.total;
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('button,input[type=file],input[type=checkbox]').forEach(b => b.disabled = value);
  $('reader-close').disabled = false;
  $('ask-button').firstChild.textContent = value ? 'Working… ' : 'Ask workspace ';
  updateControls();
}
function clearAnswer() { $('answer-section').hidden = true; $('welcome').hidden = false; }
async function refresh() {
  const [docs, status] = await Promise.all([api('/api/documents'), api('/api/status')]);
  library = docs;
  if (selectedDocuments !== null) selectedDocuments = new Set(docs.filter(d => selectedDocuments.has(d.id)).map(d => d.id));
  $('document-count').textContent = docs.length;

  $('mode-label').textContent = status.provider === 'extractive-v1' ? 'Local · source excerpts' : `${status.provider} · generated answers`;
  $('privacy-label').textContent = status.provider === 'compatible' ? 'Remote provider configured' : 'Local workspace';
  $('index-meta').textContent = `${status.chunks} passages · ${status.retrieval.toUpperCase()}`;
  $('documents').replaceChildren();
  if (!docs.length) $('documents').append(element('p', 'empty-library', 'Your library is ready for its first document.'));
  docs.forEach(doc => {
    const row = element('div', 'document');
    const include = element('input', 'document-select'); include.type = 'checkbox';
    include.checked = selectedDocuments === null || selectedDocuments.has(doc.id);
    include.setAttribute('aria-label', `Include ${doc.name}`);
    include.addEventListener('change', () => {
      if (selectedDocuments === null) selectedDocuments = new Set(library.map(d => d.id));
      if (include.checked) selectedDocuments.add(doc.id); else selectedDocuments.delete(doc.id);
      clearAnswer(); updateControls();
      notice(selectedDocuments.size ? 'Questions will use only the selected documents.' : 'Select a document before asking a question.');
    });
    row.append(include);
    const label = element('div'); const name = element('button','document-name document-open',doc.name); name.type='button'; name.title = doc.name; name.setAttribute('aria-label', `Read ${doc.name}`); name.addEventListener('click', () => task(() => openReader(doc.id)));
    label.append(name, element('small','',`${doc.chunks} passages · indexed`)); row.append(label);
    const remove = element('button','delete-doc','×'); remove.type='button'; remove.setAttribute('aria-label',`Delete ${doc.name}`);
    remove.addEventListener('click', () => task(async () => { await api(`/api/documents/${doc.id}`,{method:'DELETE'}); clearAnswer(); await refresh(); notice(`Removed ${doc.name} and invalidated cached answers.`); }));
    row.append(remove); $('documents').append(row);
  });
  updateControls();
}
async function task(action) { if (busy) return; setBusy(true); try { await action(); } catch(error) { notice(error.message,true); } finally { setBusy(false); } }
function showAnswer(answer) {
  $('welcome').hidden = true; $('answer-section').hidden = false;
  $('answer-title').textContent = answer.status === 'abstained' ? 'More evidence needed' : answer.mode === 'extractive-v1' ? 'From your documents' : 'Evidence-linked answer';
  $('answer-meta').textContent = `${answer.cached ? 'Cached · ' : ''}${answer.elapsed_ms} ms · ${answer.retrieval.toUpperCase()}`;
  $('answer-card').replaceChildren(); $('sources').replaceChildren();
  const sourceNumbers = new Map(answer.sources.map((s,i) => [s.id,i+1]));
  if (answer.reason) $('answer-card').append(element('p','',answer.reason));
  answer.claims.forEach(claim => {
    const p = element('p','',claim.text);
    claim.evidence.forEach(e => { const a = element('a','citation-link',`[${sourceNumbers.get(e.source_id)}]`); a.href=`#source-${e.source_id}`; a.title=e.quote; p.append(a); });
    $('answer-card').append(p);
  });
  $('answer-card').append(element('p','answer-note',answer.mode === 'extractive-v1' ? 'Verbatim excerpts, selected by retrieval. No generated interpretation.' : 'Source identifiers and exact quotes checked. This does not verify the meaning of a generated claim; review the evidence.'));
  $('source-count').textContent = `${answer.sources.length} retrieved passages`;
  answer.sources.forEach((s,i) => {
    const card=element('article','source-card'); card.id=`source-${s.id}`;
    card.append(element('div','source-title',`[${i+1}] ${s.name}`),element('small','',`Page ${s.page}${s.section ? ' · '+s.section : ''} · characters ${s.start}–${s.end}`));
    const details=element('details'); details.open=true; details.append(element('summary','','View source passage'),element('pre','',s.text)); card.append(details); const read=element('button','read-document','Read in document'); read.type='button'; read.addEventListener('click',()=>task(()=>openReader(s.document_id,s.id))); card.append(read); $('sources').append(card);
  });
}
$('question-form').addEventListener('submit', e => { e.preventDefault(); task(async () => {
  const question=$('question').value.trim(); if(!question) return;
  notice('Searching your sources…');
  const answer=await api('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question, document_ids: selectedDocuments === null ? null : [...selectedDocuments]})});
  showAnswer(answer); notice('');
}); });
$('question').addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('question-form').requestSubmit();} });
$('file-input').addEventListener('change', () => task(async () => {
  let completed=0; const failures=[];
  for(const file of $('file-input').files){
    if(file.size>20*1024*1024){failures.push(`${file.name}: exceeds 20 MiB`);continue;}
    const form=new FormData();form.append('file',file);notice(`Indexing ${file.name}…`);
    try {await api('/api/documents',{method:'POST',body:form});completed++;}catch(error){failures.push(`${file.name}: ${error.message}`);}
  }
  $('file-input').value='';clearAnswer();await refresh();notice(`${completed} document(s) indexed.${failures.length?' '+failures.join(' '):' Your library is ready.'}`,failures.length>0);
}));
$('demo-button').addEventListener('click',()=>task(async()=>{notice('Indexing the example collection…');await api('/api/demo',{method:'POST'});clearAnswer();await refresh();notice('Example collection ready. Pick a question below.');}));
for(const button of document.querySelectorAll('[data-question]')) button.addEventListener('click',()=>{$('question').value=button.dataset.question;$('question').focus();});
async function openReader(documentId, focus = null, offset = 0) {
  const dialog = $('document-reader');
  if (!dialog.open) dialog.showModal();
  $('reader-passages').replaceChildren(element('p', '', 'Loading source text…'));
  $('reader-message').textContent = '';
  const params = new URLSearchParams({offset: String(offset), limit: '12'});
  if (focus) params.set('focus', focus);
  try {
    const data = await api(`/api/documents/${encodeURIComponent(documentId)}/passages?${params}`);
    readerState = {...data, documentId};
    $('reader-title').textContent = data.document.name;
    $('reader-page').textContent = `${data.offset + 1}–${Math.min(data.offset + data.limit, data.total)} of ${data.total} passages`;
    $('reader-passages').replaceChildren();
    for (const p of data.passages) {
      const section = element('section', p.id === focus ? 'reader-passage highlighted' : 'reader-passage');
      section.id = `reader-${p.id}`;
      section.append(element('small', '', `Page ${p.page}${p.section ? ' · '+p.section : ''} · characters ${p.start}–${p.end}`), element('pre', '', p.text));
      $('reader-passages').append(section);
    }
    const target = focus && document.getElementById(`reader-${focus}`);
    if (target) target.scrollIntoView({block: 'center'}); else $('reader-passages').scrollTop = 0;
  } catch (error) {
    readerState = null;
    $('reader-passages').replaceChildren();
    $('reader-message').textContent = error.message;
  }
  updateControls();
}
$('all-documents').addEventListener('click', () => task(async () => { selectedDocuments = null; clearAnswer(); await refresh(); notice('Questions will use the whole library.'); }));
$('reader-close').addEventListener('click', () => $('document-reader').close());
$('reader-previous').addEventListener('click', () => task(() => openReader(readerState.documentId, null, Math.max(0, readerState.offset - readerState.limit))));
$('reader-next').addEventListener('click', () => task(() => openReader(readerState.documentId, null, readerState.offset + readerState.limit)));
refresh().catch(error=>notice(error.message,true));
