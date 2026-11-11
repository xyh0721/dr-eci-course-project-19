"""Offline verification, including raw-response provenance; no API key needed."""
import json
import dreci as d

frozen=d.read(d.ROOT/'configs/frozen.json')
assert frozen['code_sha256']==d.digest((d.ROOT/'code/dreci.py').read_bytes()),'Frozen inference implementation changed'
assert frozen['protocol_sha256']==d.digest((d.ROOT/'configs/protocol.json').read_bytes())
assert frozen['validation_predictions_sha256']==d.digest((d.ROOT/'results/validation/predictions.jsonl').read_bytes())
for split in ['validation','test']:
    ps=d.ablate(d.rows(d.ROOT/f'results/{split}/predictions.jsonl'),frozen['q'])
    wanted={m['id'] for m in d.rows(d.ROOT/'data_info/split_manifest.jsonl') if m['split']==split}
    assert len(wanted)==len(ps) and wanted=={x['id'] for x in ps}
    for e in ['E0','E1','E2']:
        computed=d.metrics(ps,e);saved=d.read(d.ROOT/f'results/{e}/{split}_metrics.json')
        assert all(saved[k]==v for k,v in computed.items()),(split,e)
    for x in ps:
        for call in x['calls'].values():
            raw=d.read(d.ROOT/'results/raw'/(call['request_hash']+'.json'))
            assert d.digest(raw['request'])==call['request_hash']
            assert all('Authorization' not in raw['request'] for _ in [0])
            assert 'bi_causal_label' not in json.dumps(raw['request'])
print('PASS: frozen code/config, exact sample coverage, E0/E1/E2 metrics, request hashes, no gold labels in prompts')
