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
    w.d3.json = async () => structuredClone(data);

    for (const script of w.document.querySelectorAll(
      'script:not([src])'
    )) {
      w.eval(script.textContent);
    }

    await new Promise(resolve => setTimeout(resolve, 30));

    const counts = () => [
      w.document.querySelectorAll('circle.node').length,
      w.document.querySelectorAll('line.link').length
    ];

    assert.deepEqual(counts(), [3, 2]);
    assert.match(
      w.document.getElementById('loading').textContent,
      /truncated/
    );

    const select = w.document.getElementById('nodeFilter');

    for (const [type, expected] of [
      ['drug', [2, 1]],
      ['all', [3, 2]],
      ['protein', [1, 0]],
      ['all', [3, 2]]
    ]) {
      select.value = type;
      select.dispatchEvent(new w.Event('change'));
      assert.deepEqual(counts(), expected);
    }

    w.document.querySelector('circle.node').dispatchEvent(
      new w.MouseEvent('click', {bubbles: true})
    );

    assert.match(
      w.document.getElementById('tooltip').textContent,
      /ID: drug:0/
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