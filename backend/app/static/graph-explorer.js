(() => {
  const params = new URLSearchParams(location.search), first = params.get('drug1_id'), second = params.get('drug2_id');
  const loading = document.getElementById('loading'), filter = document.getElementById('nodeFilter');
  const colors = {drug:'#1f77b4', protein:'#2ca02c', disease:'#d62728', sideeffect:'#9467bd', function:'#c46b00'};
  let nodes = [], links = [], selectedId, simulation, graphData, sequence = 0, annotationRequest;
  let canvas, zoom, circles, labels, drawnLinks;
  const anchors = new Set([`drug:${first}`, `drug:${second}`]);
  const el = (tag, text, className) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; };
  const nodeById = id => nodes.find(n => n.id === id);
  const name = id => nodeById(id)?.label || id;
  const endpoint = value => typeof value === 'object' ? value.id : value;

  function updateLabels(hovered) {
    if (!labels) return;
    labels.style('display', n => document.getElementById('show-labels').checked || anchors.has(n.id) || n.id === selectedId || n.id === hovered ? null : 'none');
  }
  function describe(id) {
    const node = nodeById(id); if (!node) return;
    selectedId = id;
    const panel = document.getElementById('tooltip'); panel.replaceChildren();
    panel.append(el('h3', node.label), el('span', node.type === 'protein' ? 'Protein / gene' : node.type, 'badge'));
    if (node.full_name) panel.append(el('p', node.full_name));
    const details = el('dl');
    for (const [label, value] of [['Source identifier', node.identifier], ['BioKG entity', node.id], ['DrugBank ID', node.drugbank_id], ['Concept type', node.semantic_type], ['Organism', node.organism], ['Name source', node.annotation_source]]) {
      if (value) details.append(el('dt', label), el('dd', value));
    }
    panel.append(details);
    if (node.description) panel.append(el('p', node.description, 'description'));
    else panel.append(el('p', node.annotation_status === 'unavailable' ? 'The annotation service is unavailable. The original identifier is retained; try reloading later.' : node.annotation_status === 'not_found' ? 'No exact annotation was returned by the source. The original identifier is retained.' : 'No source definition is available yet.', 'muted'));
    if (node.obsolete) panel.append(el('p', 'This ontology term is marked obsolete by the source.', 'error'));
    if (node.source_url) { const a = el('a', 'Open source record'); a.href = node.source_url; a.target = '_blank'; a.rel = 'noopener noreferrer'; panel.append(a); }
    if (Array.isArray(node.targets) && node.targets.length) {
      const group = el('details'), list = el('ul'); group.append(el('summary', `${node.targets.length} recorded DrugBank targets`));
      for (const target of node.targets) list.append(el('li', typeof target === 'string' ? target : [target.name, target.organism].filter(Boolean).join(' — ') || JSON.stringify(target)));
      group.append(list); panel.append(group);
    }
    const neighbors = document.getElementById('neighbors'); neighbors.replaceChildren();
    const adjacent = links.filter(l => l.source === id || l.target === id);
    document.getElementById('neighbor-heading').hidden = false;
    document.getElementById('neighbor-heading').textContent = `Recorded connections in this view (${adjacent.length})`;
    for (const edge of adjacent) {
      const other = edge.source === id ? edge.target : edge.source, row = el('li'), button = el('button', name(other)); button.type = 'button'; button.addEventListener('click', () => describe(other));
      row.append(button, el('small', `${edge.source === id ? 'Outgoing →' : 'Incoming ←'} ${edge.relation_name || edge.relation} · ID ${edge.relation}`)); neighbors.append(row);
    }
    if (!adjacent.length) neighbors.append(el('li', 'No connections to this entity in the returned view.'));
    if (circles) circles.classed('selected', n => n.id === id);
    if (drawnLinks) drawnLinks.classed('active', l => endpoint(l.source) === id || endpoint(l.target) === id);
    document.getElementById('entity-list').value = id; updateLabels();
  }
  function renderEntityList() {
    const list = document.getElementById('entity-list'), term = document.getElementById('entity-search').value.trim().toLocaleLowerCase(); list.replaceChildren();
    for (const node of [...nodes].sort((a,b) => a.label.localeCompare(b.label))) {
      if (!`${node.label} ${node.identifier} ${node.full_name || ''} ${node.id}`.toLocaleLowerCase().includes(term)) continue;
      const option = el('option', `${node.label} · ${node.type} · ${node.identifier}`); option.value = node.id; list.append(option);
    }
    if (selectedId) list.value = selectedId;
  }
  function renderTable() {
    const body = document.getElementById('relationship-rows'); body.replaceChildren();
    for (const edge of links) {
      const row = el('tr');
      for (const id of [edge.source, edge.target]) { const td = el('td'), button = el('button', name(id)); button.type = 'button'; button.addEventListener('click', () => describe(id)); td.append(button); row.append(td); }
      const relation = el('td', `${edge.relation_name || edge.relation} (ID ${edge.relation})`); row.insertBefore(relation, row.lastChild); body.append(row);
    }
  }
  function renderGraph() {
    simulation?.stop();
    if (typeof d3 === 'undefined') { loading.textContent = 'Network drawing unavailable: the graph library did not load. Use the entity list and relationship table below.'; return; }
    const svg = d3.select('svg'); svg.selectAll('*').remove();
    svg.append('defs').append('marker').attr('id','arrow').attr('viewBox','0 -5 10 10').attr('refX',17).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#93a5b2');
    canvas = svg.append('g'); zoom = d3.zoom().extent([[0,0],[1000,650]]).scaleExtent([.25, 5]).on('zoom', event => canvas.attr('transform', event.transform)); svg.call(zoom);
    const displayed = nodes.map(n => ({...n})), edges = links.map(l => ({...l}));
    simulation = d3.forceSimulation(displayed).force('link', d3.forceLink(edges).id(n => n.id).distance(125)).force('charge', d3.forceManyBody().strength(-280)).force('collision',d3.forceCollide(22)).force('center',d3.forceCenter(500,325));
    drawnLinks = canvas.append('g').selectAll('line').data(edges).join('line').attr('class','link').attr('marker-end','url(#arrow)')
      .on('click', (event, edge) => { describe(endpoint(edge.source)); document.getElementById('tooltip').prepend(el('p', `${name(endpoint(edge.source))} → ${name(endpoint(edge.target))}: ${edge.relation_name || edge.relation} (ID ${edge.relation})`, 'pair-source')); });
    drawnLinks.append('title').text(e => `${name(endpoint(e.source))} → ${name(endpoint(e.target))}\n${e.relation_name || e.relation} (ID ${e.relation})`);
    circles = canvas.append('g').selectAll('circle').data(displayed).join('circle').attr('class','node').attr('r', n => anchors.has(n.id) ? 11 : 7).attr('fill',n => colors[n.type] || '#71818d').attr('tabindex',0).attr('role','button').attr('aria-label',n => `${n.label}, ${n.type}, ${n.identifier}`)
      .on('click',(event,n) => describe(n.id)).on('mouseover',(event,n) => updateLabels(n.id)).on('mouseout',() => updateLabels())
      .on('keydown',(event,n) => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); describe(n.id);}})
      .call(d3.drag().on('start',(event,n) => {if(!event.active) simulation.alphaTarget(.2).restart(); n.fx=n.x;n.fy=n.y;}).on('drag',(event,n) => {n.fx=event.x;n.fy=event.y;}).on('end',(event,n) => {if(!event.active) simulation.alphaTarget(0);n.fx=null;n.fy=null;}));
    circles.append('title').text(n => `${n.label}\n${n.identifier} · ${n.type}`);
    labels = canvas.append('g').selectAll('text').data(displayed).join('text').attr('class','nodelabel').attr('text-anchor','middle');
    labels.each(function(n) {
      const words = String(n.label).match(/.{1,26}(?:\s|$)|.{1,26}/g) || [n.label];
      d3.select(this).selectAll('tspan').data(words).join('tspan').attr('x',0).attr('dy',(line,index) => index ? 14 : 0).text(line => line.trim()); n.labelHeight=words.length*14;
    }); updateLabels();
    simulation.on('tick',() => {
      drawnLinks.attr('x1',e=>e.source.x).attr('y1',e=>e.source.y).attr('x2',e=>e.target.x).attr('y2',e=>e.target.y);
      circles.attr('cx',n=>n.x).attr('cy',n=>n.y); labels.attr('transform',n=>`translate(${n.x},${n.y-12-n.labelHeight})`);
    });
    if (selectedId) describe(selectedId);
  }
  function refreshAnnotations() {
    document.getElementById('named-count').textContent = nodes.filter(n => n.annotation_status === 'resolved').length;
    renderEntityList(); renderTable(); renderGraph();
  }
  function explanationLink() {
    document.getElementById('explain-link').href = '/explain_view?' + new URLSearchParams({drug1_id:first, drug2_id:second, relation_type:document.getElementById('model-relation').value});
  }
  async function loadGraph() {
    const current = ++sequence; annotationRequest?.abort(); filter.disabled = true; loading.textContent = 'Loading recorded relationships…'; loading.className = '';
    document.getElementById('annotation-status').textContent='';
    try {
      if (!/^\d+$/.test(first || '') || !/^\d+$/.test(second || '') || Number(first) === Number(second)) throw new Error('Choose two distinct valid BioKG drug indices.');
      const response = await fetch('/graph_data?' + new URLSearchParams({drug1_id:first, drug2_id:second, entity_type:filter.value}));
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Graph unavailable.'); if (current !== sequence) return;
      graphData = data; nodes = data.nodes; links = data.links; selectedId = nodes.some(n=>n.id===selectedId) ? selectedId : `drug:${first}`;
      document.getElementById('pair-label').textContent = (data.selected_drugs || []).map(d=>`${d.name} (BioKG ${d.biokg_index})`).join(' + ');
      document.getElementById('entity-count').textContent = nodes.length; document.getElementById('edge-count').textContent = links.length;
      loading.textContent = `${links.length} recorded edges in this ${filter.value === 'all' ? 'all-type' : filter.options[filter.selectedIndex].text} view.` + (data.truncated ? ' Truncated at 100 edges; filter by type to explore other relationships.' : '') + (!links.length ? ' No incident relationships of this type were returned.' : '');
      const relation = document.getElementById('model-relation'), previous = relation.value || params.get('relation_type'); relation.replaceChildren();
      for (const item of data.model_relations || []) { const option=el('option', `${item.name} (ID ${item.id})`);option.value=item.id;relation.append(option); }
      if ([...relation.options].some(o=>o.value===previous)) relation.value=previous; explanationLink();
      document.getElementById('download-graph').disabled=false; refreshAnnotations(); filter.disabled=false;
      const unresolved = nodes.filter(n=>n.annotation_status !== 'resolved');
      if (!unresolved.length) {document.getElementById('annotation-status').textContent='All displayed names are resolved from available metadata.';return;}
      document.getElementById('annotation-status').textContent='Resolving exact identifiers using public annotation sources…';
      annotationRequest = new AbortController();
      // Keep each annotation request bounded to one service/type, with progressive updates.
      for (const kind of [...new Set(unresolved.map(n=>n.type))]) {
        if (current !== sequence) return;
        const group=unresolved.filter(n=>n.type===kind);
        document.getElementById('annotation-status').textContent=`Resolving ${kind} names from exact source identifiers…`;
        const annotations=await fetch('/graph_entities',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:group.map(n=>n.id)}),signal:annotationRequest.signal});
        const details=await annotations.json(); if(current!==sequence)return;if(!annotations.ok)throw new Error(details.error || 'Annotations unavailable.');
        const lookup=new Map(details.nodes.map(n=>[n.id,n]));nodes=nodes.map(n=>lookup.get(n.id)||n);graphData.nodes=nodes;refreshAnnotations();
      }
      const missing=nodes.filter(n=>n.annotation_status!=='resolved').length;
      document.getElementById('annotation-status').textContent=missing ? `${missing} entities retain their source identifier because an exact name was unavailable. Select an entity for its lookup status and source link.` : 'Names resolved and cached. Select an entity for its source definition and connections.';
    } catch(error) {
      if(error.name==='AbortError'||current!==sequence)return;
      if(graphData)document.getElementById('annotation-status').textContent='Additional names unavailable. Available graph data and identifiers remain visible. '+error.message;
      else {loading.textContent=error.message;loading.className='error';}
    } finally { if(current===sequence)filter.disabled=false; }
  }
  filter.addEventListener('change', loadGraph);
  document.getElementById('entity-search').addEventListener('input',renderEntityList);
  document.getElementById('entity-list').addEventListener('change',event=>describe(event.target.value));
  document.getElementById('show-labels').addEventListener('change',()=>updateLabels());
  document.getElementById('model-relation').addEventListener('change',explanationLink);
  document.getElementById('reset-view').addEventListener('click',()=>{if(zoom)d3.select('svg').transition().duration(250).call(zoom.transform,d3.zoomIdentity);});
  document.getElementById('download-graph').addEventListener('click',()=>{if(!graphData)return;const url=URL.createObjectURL(new Blob([JSON.stringify(graphData,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='biokg-recorded-relationships.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  window.addEventListener('pagehide',()=>{simulation?.stop();annotationRequest?.abort();});
  loadGraph();
})();
