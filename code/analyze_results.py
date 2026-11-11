from pathlib import Path
import sys,json,collections
import csv
import dreci as d
P=d.ROOT;ps=d.rows(P/'results/test/predictions.jsonl');xs={x['item_id']:x for x in d.get_split('test')}
notes={
19984:('发现阶段漏检','标题将枪击与审判联系起来，原标注为正。三种问法都回答No，后续模块没有机会补救，说明发现阈值不能弥补所有语境不足。'),
19784:('否定与标注范围差异','模型依据denies拒绝把纵火解释为已发生的火灾原因，而数据保留正标签。此例应同时考虑否定语义与事件标注口径，不能直接断言模型完全不懂因果。'),
19690:('同一短语拆分导致误检','catches fire整体描述起火。模型把catches和fire当成两个可独立建立因果的事件，用常识规则3判正，原标签为负。'),
19685:('显式线索过度匹配','调查what caused the fire并未确定具体原因。模型看到caused就按规则1判正，忽视了所问事件是否对应独立原因。'),
7478:('共同关联实体被当作共同原因','模型将California prosecutors作为Fork的Z，但检察官的发言并不证明continued与a rehab facility存在共同原因关系。共现实体可误导间接分支。')}
cases=[]
for ident,(typ,note) in notes.items():
    x=next(x for x in ps if x['id']==ident);cases.append({**x,'input':d.context(xs[ident]),'error_type':typ,'analysis':note})
d.save(P/'results/error_cases.json',cases)
examples=[]
for ident in [20464,7489]:
    x=next(x for x in ps if x['id']==ident);examples.append({**x,'input':d.context(xs[ident])})
d.save(P/'results/inference_examples.json',examples)
st={'by_topic':{},'positive_gate_no':sum(x['gold']==1 and x['route']=='gate_no' for x in ps),'positive_reasoner_no':sum(x['gold']==1 and x['E1']==0 and x['route']!='gate_no' for x in ps),'positive_failure':sum(x['gold']==1 and x['E1'] is None for x in ps),'E0_to_E1':dict(collections.Counter(f"gold={x['gold']},E0={x['E0']},E1={x['E1']}" for x in ps)), 'rules':dict(collections.Counter(str(r) for x in ps for r in x['calls'].get('reasoner',{}).get('parsed',{}).get('matched_rules',[])))}
for t in sorted({x['topic'] for x in ps}):
    z=[x for x in ps if x['topic']==t];st['by_topic'][t]={e:d.metrics(d.ablate(z,3),e) for e in ['E0','E1','E2']}
reverse={};inconsistent=0;total=0
for x in ps:
    inp=xs[x['id']];k=d.digest([inp['words'],inp['events'][::-1]])
    if k in reverse:total+=1;inconsistent+=int(reverse[k]!=x['E1'])
    reverse[d.sample_key(inp)]=x['E1']
st['reverse_pair_comparisons']=total;st['reverse_prediction_disagreements']=inconsistent
d.save(P/'results/diagnostics.json',st)
q=d.read(P/'configs/frozen.json')['q'];table=[]
for split in ['validation','test']:
    zs=d.rows(P/f'results/{split}/predictions.jsonl')
    for e in ['E0','E1','E2']:
        m=d.read(P/f'results/{e}/{split}_metrics.json');cs={};ncall=0
        for z in zs:
            keys=['explorer1'] if e=='E0' else ['explorer1','explorer2','explorer3']
            if e!='E0' and z['votes']>=(1 if e=='E1' else q):keys += [k for k in ['mediator','reasoner'] if k in z['calls']]
            for k in keys:
                c=z['calls'][k];cs[c['request_hash']]=c;ncall+=1
        m.update(unique_cost_cny=sum(c['cost_cny'] for c in cs.values()),unique_serial_seconds=sum(c['seconds'] for c in cs.values()),logical_call_count=ncall)
        d.save(P/f'results/{e}/{split}_metrics.json',m);table.append(m)
with (P/'results/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(json.dumps(st,ensure_ascii=False))
