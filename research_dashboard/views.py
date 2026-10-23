"""Authoritative read-only evidence views; absent facts remain explicitly absent."""


def evidence_payload(report):
    config = report.experiment_config
    return {'experiment_id': config.experiment_id, 'architecture': config.architecture.value,
            'research_only': True, 'transaction_cost_assumptions': config.transaction_costs.model_dump(mode='json'),
            'runs': [{'run_id': run.run_id, 'outcome_class': run.outcome_class, 'status': run.status,
                'strategy': run.result.get('selected_strategy'),
                'strategy_id': run.strategy_id,
                'execution_config': run.result.get('execution_config'),
                'evidence': {scope: run.result.get(scope+'_metrics') or None
                             for scope in ('train', 'validation', 'test', 'final_holdout')},
                'paper': {'status': 'NOT_OBSERVED', 'metrics': None},
                'certification': run.result.get('certification') or {},
                'risk': run.result.get('risk'), 'provenance': run.result.get('provenance'),
                'lineage': run.result.get('lineage'), 'invocation_audit': run.result.get('invocation_audit')}
                for run in report.runs]}


EVIDENCE_HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy research evidence</title><style>
body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#e7edf8;background:#0b1020}
a{color:#82d2ff}nav{display:flex;gap:1rem;flex-wrap:wrap}section{padding:1rem 0;border-bottom:1px solid #27324a}
pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:1rem;background:#121b30}p{line-height:1.5}
</style></head><body><a href="/">Research dashboard</a><h1>Strategy research evidence</h1>
<p>Research results and simulations do not establish paper performance. Missing evidence is shown as unavailable.</p>
<nav><a href="#strategy">Strategy</a><a href="#evidence">Evidence</a><a href="#risk">Risk</a><a href="#provenance">Provenance</a></nav>
<main id="content"></main><script>
const id = decodeURIComponent(location.pathname.slice('/evidence/'.length));
const main = document.getElementById('content');
const show = (parent, title, value) => {
  const h = document.createElement('h3'); h.textContent = title;
  const p = document.createElement('pre'); p.textContent = value == null ? 'UNAVAILABLE' : JSON.stringify(value,null,2);
  parent.append(h,p);
};
fetch('/api/evidence/'+encodeURIComponent(id)).then(r=>{if(!r.ok)throw Error('unavailable');return r.json()}).then(report=>{
  const title=document.createElement('p');title.textContent=report.experiment_id+' · '+report.architecture;main.append(title);
  for (const [key,label] of [['strategy','Strategy definition and execution'],['evidence','Evidence by permitted scope'],['risk','Risk policy and decision'],['provenance','Data, model, prompt and source identity']]) {
    const section=document.createElement('section');section.id=key;
    const heading=document.createElement('h2');heading.textContent=label;section.append(heading);
    report.runs.forEach(run=>{
      show(section,run.run_id+' · '+run.outcome_class, key==='strategy' ? {strategy:run.strategy,execution_config:run.execution_config} : key==='evidence' ? {status:run.status,scopes:run.evidence,paper:run.paper,certification:run.certification} : key==='risk' ? run.risk : {provenance:run.provenance,lineage:run.lineage,invocations:run.invocation_audit});
    });main.append(section);
  }
}).catch(()=>{main.textContent='Evidence is unavailable.'});
</script></body></html>'''
