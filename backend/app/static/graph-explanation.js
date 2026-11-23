/* Trained graph attribution, shown separately from exact decoder arithmetic. */
(() => {
  const button = document.getElementById('explain-graph');
  const status = document.getElementById('graph-status');
  let result;
  button.addEventListener('click', async () => {
    if (!explanation) return;
    const query = new URLSearchParams({drug1_id: explanation.drug1_id,
      drug2_id: explanation.drug2_id, relation_type: explanation.relation_type});
    const requested = explanation;
    button.disabled = true;
    status.className = '';
    status.textContent = 'Calculating graph attribution and edge ablations…';
    document.getElementById('graph-attribution').hidden = true;
    try {
      const response = await fetch('/explain_graph?' + query);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Graph attribution unavailable.');
      if (explanation !== requested) return;
      result = data;
      const names = new Map(data.nodes.map(n => [n.id, n.label + ' (' + n.id + ')']));
      const body = document.getElementById('graph-edges');
      body.replaceChildren();
      for (const edge of data.edges) {
        const row = document.createElement('tr');
        for (const value of [names.get(edge.source) || edge.source, edge.relation_name,
          names.get(edge.target) || edge.target, edge.mask_weight.toFixed(4),
          edge.logit_change_when_gated.toFixed(6)]) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        body.append(row);
      }
      for (const [id, value] of Object.entries({'graph-original': data.uncalibrated_model_score,
        'graph-kept': data.top_edges_kept_score, 'graph-gated': data.top_edges_gated_score})) {
        document.getElementById(id).textContent = value.toFixed(6);
      }
      document.getElementById('graph-scope').textContent = data.scope + '. Ranking ' +
        data.computational_edges + ' candidate edges; displaying ' + data.displayed_edges + '.';
      const report = data.evaluation;
      document.getElementById('graph-evaluation').textContent = 'Evaluated on ' + report.queries +
        ' held-out explanation queries. At a 20% edge budget, mean retained-logit error: ' +
        report.mean_retained_20pct_logit_error.toFixed(4) + '; equal-budget random selection: ' +
        report.mean_random_20pct_logit_error.toFixed(4) + '. Full results are included in the JSON download.';
      document.getElementById('graph-limits').textContent = data.limitations;
      document.getElementById('graph-attribution').hidden = false;
      status.textContent = data.interpretation;
    } catch (error) {
      if (explanation === requested) { status.textContent = error.message; status.className = 'error'; }
    } finally { button.disabled = false; }
  });
  document.getElementById('download-graph').addEventListener('click', () => {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'biokg-pgexplainer.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();
