// Browser-console helpers for driving QuantConnect's REST API (/api/v2) from a logged-in QuantConnect tab.
// They upload qc/vol_study.py, compile, launch the backtest, and save runtime statistics / chart series
// to a local helper server (localhost:8765, POST /save/<name> -> results/). No credentials are stored here.
window.qapi = async (path, body) => fetch('/api/v2/' + path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})}).then(r => r.json());
window.lf = async (f) => fetch('http://localhost:8765/' + f + '?t=' + Date.now()).then(r => r.text());
window.sleep = (ms) => new Promise(r => setTimeout(r, ms));
window.runqc = async (pid, file, name, repl) => {
  let code = await lf(file);
  for (const [a, b] of Object.entries(repl || {})) { code = code.replace(new RegExp('^' + a + ' = .*$', 'm'), a + ' = ' + b); }
  await qapi('files/update', {projectId: pid, name: 'main.py', content: code});
  let cc = await qapi('compile/create', {projectId: pid});
  for (let i = 0; i < 40 && cc.state !== 'BuildSuccess' && cc.state !== 'BuildError'; i++) { await sleep(2000); cc = await qapi('compile/read', {projectId: pid, compileId: cc.compileId}); }
  if (cc.state !== 'BuildSuccess') return {build: cc.state, logs: (cc.logs || []).filter(l => /Error/.test(l)).slice(0, 5)};
  const b = await qapi('backtests/create', {projectId: pid, compileId: cc.compileId, backtestName: name});
  return {bt: b.backtest && b.backtest.backtestId, err: b.errors};
};
window.waitqc = async (pid, bt, maxs) => { let r; for (let i = 0; i < (maxs || 55); i++) { r = await qapi('backtests/read', {projectId: pid, backtestId: bt}); if (r.backtest && r.backtest.completed) break; await sleep(2000); } return r.backtest; };
window.newproj = async (name) => { const pc = await qapi('projects/create', {name, language: 'Py'}); return pc.projects[0].projectId; };
window.fmtstats = (b) => { const rs = b.runtimeStatistics || {}; return Object.entries(rs).filter(([k]) => k.includes('|') || k.startsWith('_')).map(([k, v]) => k.padEnd(26) + ' ' + v).join('\n'); };
window.savelocal = async (name, text) => fetch('http://localhost:8765/save/' + name, {method: 'POST', body: text}).then(r => r.text());
window.saveBt = async (pid, bt, name) => { const r = await qapi('backtests/read', {projectId: pid, backtestId: bt}); const b = r.backtest; const keep = {name: b.name, status: b.status, projectId: pid, backtestId: bt, runtimeStatistics: b.runtimeStatistics, statistics: b.statistics, error: b.error, stacktrace: b.stacktrace}; return savelocal(name, JSON.stringify(keep, null, 1)); };
// Background queue: runs jobs one at a time, saves each finished backtest to results/<tag>.json
window.qstatus = window.qstatus || [];
window.runQueue = async (pid, file, jobs) => {
  for (const j of jobs) {
    const rec = {tag: j.tag, state: 'starting'}; qstatus.push(rec);
    try {
      let r;
      for (let k = 0; k < 400; k++) { r = await runqc(j.pid || pid, j.file || file, j.tag, j.repl); if (r.bt) break; rec.state = 'retry ' + JSON.stringify(r).slice(0, 120); await sleep(30000); }
      rec.bt = r.bt; rec.state = 'running';
      let b; for (let i = 0; i < 2400; i++) { const x = await qapi('backtests/read', {projectId: j.pid || pid, backtestId: r.bt}); b = x.backtest; if (b && b.completed) break; rec.prog = b && b.progress; await sleep(5000); }
      await saveBt(j.pid || pid, r.bt, j.tag + '.json'); rec.state = 'done ' + b.status;
    } catch (e) { rec.state = 'error ' + e; }
  }
  qstatus.push({tag: 'QUEUE_FINISHED'});
};
window.waitAndSave = async (pid, bt, tag) => { const rec = {tag, bt, state: 'running'}; qstatus.push(rec); let b; for (let i = 0; i < 2400; i++) { const x = await qapi('backtests/read', {projectId: pid, backtestId: bt}); b = x.backtest; if (b && b.completed) break; rec.prog = b && b.progress; await sleep(5000); } await saveBt(pid, bt, tag + '.json'); rec.state = 'done ' + b.status; };
window.savePxChart = async (pid, bt, fname) => {
  const c = await qapi('backtests/chart/read', {projectId: pid, backtestId: bt, name: 'PX', count: 5000, start: 1262304000, end: 1800000000});
  const ser = c.chart.series; const key = v => new Date(v[0] * 1000).toISOString().slice(0, 10).replace(/-/g, '');
  const dates = ser.SPY.values.map(key); const px = {};
  for (const t of Object.keys(ser)) { const m = new Map(ser[t].values.map(v => [key(v), v[1]])); px[t] = dates.map(d => m.has(d) ? m.get(d) : null); }
  return savelocal(fname, JSON.stringify({dates, px}));
};
'helpers loaded';
window.saveCharts = async (pid, bt, names, fname) => {
  const out = {};
  for (const n of names) {
    let c;
    for (let i = 0; i < 30; i++) { c = await qapi('backtests/chart/read', {projectId: pid, backtestId: bt, name: n, count: 5000, start: 1262304000, end: 1800000000}); if (c.chart) break; await sleep(3000); }
    if (!c.chart) { out[n] = {error: JSON.stringify(c).slice(0, 200)}; continue; }
    out[n] = {};
    for (const [s, v] of Object.entries(c.chart.series)) out[n][s] = v.values.map(p => [new Date(p[0] * 1000).toISOString().slice(0, 10), p[1]]);
  }
  return savelocal(fname, JSON.stringify(out));
};
'helpers loaded';
