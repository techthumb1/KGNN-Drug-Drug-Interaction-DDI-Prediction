const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');

(async () => {
  const html = fs.readFileSync(
    'backend/app/templates/graph.html',
    'utf8'
  );

  // Test fixture only; never served as application data.
  const data = {
    nodes: [
      {id: 'drug:0', label: 'Drug 0', type: 'drug'},
      {id: 'drug:1', label: 'Drug 1', type: 'drug'},
      {id: 'protein:0', label: 'Protein 0', type: 'protein'}
    ],
    links: [
      {source: 'drug:0', target: 'drug:1', relation: '2'},
      {source: 'drug:0', target: 'protein:0', relation: '0'}
    ],
    truncated: true
  };

  const dom = new JSDOM(html, {
    url: 'http://localhost/graph?drug1_id=0&drug2_id=1',
    runScripts: 'outside-only',
    pretendToBeVisual: true
  });

  try {
    const w = dom.window;
    const d3path = path.join(
      path.dirname(require.resolve('d3')),
      '../dist/d3.min.js'
    );

    w.eval(fs.readFileSync(d3path, 'utf8'));
    w.AbortController = AbortController;
    w.fetch = async url => {
      const result = structuredClone(data);
      const type = new URL(url, 'http://localhost').searchParams.get('entity_type');
      if (type && type !== 'all') {
        result.nodes = result.nodes.filter(n => n.type === type || n.type === 'drug');
        const ids = new Set(result.nodes.map(n => n.id));
        result.links = result.links.filter(l => ids.has(l.source) && ids.has(l.target) &&
          (type === 'drug' || result.nodes.some(n => n.type === type && (n.id === l.source || n.id === l.target))));
      }
      result.nodes = result.nodes.map(n => ({...n, identifier: n.id, annotation_status: 'resolved'}));
      result.model_relations = [{id:'2', name:'Recorded fixture relation'}];
      return {ok:true, json:async()=>result};
    };

    w.eval(fs.readFileSync('backend/app/static/graph-explorer.js', 'utf8'));

    await new Promise(resolve => setTimeout(resolve, 30));

    const counts = () => [
      w.document.querySelectorAll('circle.node').length,
      w.document.querySelectorAll('line.link').length
    ];

    assert.deepEqual(counts(), [3, 2]);
    assert.match(
      w.document.getElementById('loading').textContent,
      /Truncated/
    );

    const select = w.document.getElementById('nodeFilter');

    for (const [type, expected] of [
      ['drug', [2, 1]],
      ['all', [3, 2]],
      ['protein', [3, 1]],
      ['all', [3, 2]]
    ]) {
      select.value = type;
      select.dispatchEvent(new w.Event('change'));
      await new Promise(resolve => setTimeout(resolve, 30));
      assert.deepEqual(counts(), expected);
    }

    w.document.querySelector('circle.node').dispatchEvent(
      new w.MouseEvent('click', {bubbles: true})
    );

    assert.match(
      w.document.getElementById('tooltip').textContent,
      /BioKG entitydrug:0/
    );

    console.log(
      'PASS: graph rendering, truncation, filter restore and tooltip'
    );
  } finally {
    dom.window.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});