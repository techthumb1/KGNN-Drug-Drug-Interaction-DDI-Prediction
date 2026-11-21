(() => {
  const query = document.getElementById('drug-query');
  const suggestions = document.getElementById('drug-suggestions');
  const selected = document.getElementById('selected-drugs');
  const searchStatus = document.getElementById('search-status');
  const status = document.getElementById('list-status');
  const button = document.getElementById('check-list');
  const results = document.getElementById('list-results');
  let drugs = [], matches = [], active = -1, timer, request, sequence = 0, checking = false;

  function element(tag, text, className) {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    if (className) item.className = className;
    return item;
  }
  function identifiers(drug) {
    return [drug.drugbank_id, drug.pubchem_cid ? `PubChem CID ${drug.pubchem_cid}` : null,
      drug.biokg_index !== null ? `BioKG index ${drug.biokg_index}` : null].filter(Boolean).join(' · ');
  }
  function closeSuggestions() {
    suggestions.hidden = true;
    query.setAttribute('aria-expanded', 'false');
    query.removeAttribute('aria-activedescendant');
    active = -1;
  }
  function invalidateResults() {
    results.hidden = true;
    results.replaceChildren();
    status.textContent = '';
  }
  function renderSelected() {
    selected.replaceChildren();
    for (const drug of drugs) {
      const row = element('li');
      row.append(element('strong', drug.name), element('small', identifiers(drug)));
      const remove = element('button', 'Remove');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${drug.name}`);
      remove.disabled = checking;
      remove.addEventListener('click', () => {
        drugs = drugs.filter(item => item.key !== drug.key);
        invalidateResults(); renderSelected();
      });
      row.append(remove); selected.append(row);
    }
    button.disabled = checking || drugs.length < 2;
    query.disabled = checking || drugs.length >= 10;
    document.getElementById('list-relation').disabled = checking;
    button.textContent = checking ? 'Checking pairs…' : `Check ${drugs.length * (drugs.length - 1) / 2} selected pairs`;
  }
  function add(drug) {
    if (checking) return;
    if (drugs.some(item => item.key === drug.key || (item.pubchem_cid && item.pubchem_cid === drug.pubchem_cid))) {
      searchStatus.textContent = 'That drug is already selected.';
      return;
    }
    if (drugs.length >= 10) return;
    drugs.push(drug); query.value = ''; closeSuggestions(); invalidateResults(); renderSelected();
    searchStatus.textContent = `${drug.name} added. ${drugs.length} drugs selected.`;
    query.focus();
  }
  function highlight() {
    [...suggestions.querySelectorAll('button')].forEach((item, index) => {
      item.classList.toggle('active', index === active);
      item.setAttribute('aria-selected', String(index === active));
    });
    if (active >= 0) query.setAttribute('aria-activedescendant', `drug-option-${active}`);
  }
  query.addEventListener('input', () => {
    clearTimeout(timer); request?.abort(); const current = ++sequence;
    if (query.value.trim().length < 2) { closeSuggestions(); return; }
    timer = setTimeout(async () => {
      request = new AbortController();
      searchStatus.textContent = 'Searching supplied drug records…';
      try {
        const response = await fetch('/drugs/search?' + new URLSearchParams({q: query.value.trim()}), {signal: request.signal});
        if (!response.ok) throw new Error('Drug search is unavailable.');
        const body = await response.json();
        if (current !== sequence) return;
        matches = body.drugs; suggestions.replaceChildren(); active = -1;
        matches.forEach((drug, index) => {
          const row = element('li'); row.setAttribute('role', 'presentation');
          const choice = element('button'); choice.type = 'button'; choice.id = `drug-option-${index}`;
          choice.setAttribute('role', 'option'); choice.setAttribute('aria-selected', 'false');
          choice.append(element('strong', drug.name), element('small', identifiers(drug)));
          choice.addEventListener('click', () => add(drug)); row.append(choice); suggestions.append(row);
        });
        suggestions.hidden = !matches.length;
        query.setAttribute('aria-expanded', String(Boolean(matches.length)));
        searchStatus.textContent = matches.length ? `${matches.length} matching entries. Select one to add it.` :
          'No matching name or identifier in the loaded records. Try a canonical name, DrugBank ID or CID.';
      } catch (error) {
        if (error.name !== 'AbortError') { closeSuggestions(); searchStatus.textContent = error.message; }
      }
    }, 220);
  });
  query.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeSuggestions();
    if (!suggestions.hidden && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault(); active = (active + (event.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length; highlight();
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      if (!suggestions.hidden && matches.length) add(matches[active >= 0 ? active : 0]);
    }
  });
  document.getElementById('list-relation').addEventListener('change', invalidateResults);
  button.addEventListener('click', async () => {
    if (drugs.length < 2 || checking) return;
    closeSuggestions(); request?.abort(); ++sequence;
    checking = true; renderSelected(); status.textContent = 'Checking recorded entries and requested model scores…';
    results.hidden = true;
    try {
      const response = await fetch('/check', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({drugs: drugs.map(drug => drug.key), relation_type: document.getElementById('list-relation').value})});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || 'Pair check failed.');
      results.replaceChildren(element('h2', `${body.pairs.length} pair results`), element('p', body.note, 'source-note'));
      for (const pair of body.pairs) {
        const card = element('article', undefined, 'card');
        card.append(element('h3', `${pair.drug1.name} + ${pair.drug2.name}`));
        card.append(element('p', `${identifiers(pair.drug1)} / ${identifiers(pair.drug2)}`, 'muted'));
        card.append(element('p', pair.recorded_status === 'recorded' ? 'Recorded DrugBank interaction' :
          pair.recorded_status === 'not_found' ? 'No unambiguous entry found in the supplied DrugBank export' :
          'Recorded DrugBank lookup unavailable for this pair', 'pair-source'));
        for (const description of pair.recorded_interactions) card.append(element('p', description, 'description'));
        if (pair.relation_type) card.append(element('p', pair.model_status === 'ready' ?
          `Model relation ${pair.relation_type} · uncalibrated score: ${pair.model_score.toFixed(4)}` : `Model score unavailable: ${pair.model_status}`));
        if (pair.graph_url) { const link = element('a', 'View recorded relationships'); link.href = pair.graph_url; card.append(link); }
        results.append(card);
      }
      results.append(element('p', body.source, 'muted')); results.hidden = false;
      status.textContent = `Checked ${body.pairs.length} unique pairs from ${body.drugs.length} drugs.`;
    } catch (error) { status.textContent = error.message; }
    finally { checking = false; renderSelected(); }
  });
  renderSelected();
})();
