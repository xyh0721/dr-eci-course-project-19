"""E3: all eight reasoning rules; reuse frozen E1 discovery/mediators.

Only run/restart this experiment with: python code/e3.py run
Offline provenance/metric verification: python code/e3.py verify
Original E0/E1/E2 artifacts are read-only inputs.
"""
import argparse
import collections
import concurrent.futures as futures
import copy
import csv
import json
import time
from pathlib import Path

import dreci as d

OUT = d.ROOT / 'results/E3'
CONFIG = d.ROOT / 'configs/E3.json'
SPLITS = ('validation', 'test')


def reasoner(x, z):
    # Change only the available rule set. Keep every other prompt word,
    # output example, generation setting and input representation identical.
    old = d.INDIRECT if z else d.DIRECT
    return d.reasoner(x, z).replace(old, d.DIRECT + '\n' + d.INDIRECT, 1)


def validate_call(call):
    if call['status'] == 'ok':
        p = call['parsed']
        if not set(p['matched_rules']) <= set(range(1, 9)) or (
            (p['Answer'] == 'Yes') != bool(p['matched_rules'])
        ):
            call.update(status='failure', error='rule/answer mismatch')
    return call


def frozen_inputs():
    paths = ['code/dreci.py', 'code/e3.py', 'configs/protocol.json',
             'configs/frozen.json', 'data_info/split_manifest.jsonl',
             'local_data/intra_sent_causality.json']
    for split in SPLITS:
        paths.append(f'results/{split}/predictions.jsonl')
    return {p: d.digest((d.ROOT / p).read_bytes()) for p in paths}


def prepare():
    original = d.read(d.ROOT / 'configs/frozen.json')
    assert original['code_sha256'] == d.digest((d.ROOT / 'code/dreci.py').read_bytes())
    assert original['protocol_sha256'] == d.digest((d.ROOT / 'configs/protocol.json').read_bytes())
    hashes = frozen_inputs()
    if CONFIG.exists():
        cfg = d.read(CONFIG)
        assert cfg['input_sha256'] == hashes, 'E3 code or frozen inputs changed'
        return cfg
    cfg = {
        'experiment': 'E3', 'parent': 'E1', 'q': 1,
        'change': 'Reasoner receives rules 1-8 for every gate-passed sample; output validation accepts 1-8.',
        'unchanged': ['explorer responses', 'mediator responses and filtering',
                      'q=1', 'data and labels', 'all non-rule prompt wording',
                      'model alias and generation parameters', 'failure accounting'],
        'reuse': 'Frozen E1 explorer/mediator responses; new reasoner requests only.',
        'evaluation_status': 'Exploratory: existing test errors informed the E3 design; not a new independent test.',
        'temporal_limitation': 'API calls occur later than E1; the same model alias does not guarantee identical backend weights.',
        'selection': 'One prespecified variant; no tuning or prompt changes after validation.',
        'input_sha256': hashes,
        'generation_protocol': d.read(d.ROOT / 'configs/protocol.json'),
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    d.save(CONFIG, cfg)
    return cfg


def inputs(split):
    xs = {x['item_id']: x for x in d.get_split(split)}
    old = d.rows(d.ROOT / f'results/{split}/predictions.jsonl')
    assert len(xs) == len(old) and set(xs) == {p['id'] for p in old}
    for p in old:
        assert p['gold'] == xs[p['id']]['bi_causal_label']
    return xs, old


def predict(x, old, client):
    out = copy.deepcopy(old)
    out['original_route'] = old['route']
    out['rule_scope'] = 'all_8'
    out['baseline_reasoner_hash'] = old['calls'].get('reasoner', {}).get('request_hash')
    out['calls'].pop('reasoner', None)
    if old['route'] in ('gate_no', 'explorer_failure', 'mediator_failure'):
        out['E3'] = old['E1']
        return out
    assert old['votes'] >= 1
    call = validate_call(client.call(reasoner(x, old['mediators']), 'reasoner'))
    if client.stop:
        raise RuntimeError('API authorization/balance failure; stopped without completing the experiment.')
    out['calls']['reasoner'] = call
    out['E3'] = int(call['parsed']['Answer'] == 'Yes') if call['status'] == 'ok' else None
    return out


def run_split(split, client, workers):
    xs, old = inputs(split)
    folder = OUT / split
    checkpoints = folder / 'samples'
    checkpoints.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem): d.read(p) for p in checkpoints.glob('*.json')}
    assert set(done) <= set(xs)
    pending = [p for p in old if p['id'] not in done]
    start = time.perf_counter()
    cost_before = client.spent
    raw_before = set(client.cache.glob('*.json'))
    print(json.dumps({'split': split, 'total': len(old), 'pending': len(pending)}), flush=True)
    # Check one real request before dispatching the remainder. It is retained
    # as a normal sample and not used for selecting prompts or configurations.
    pilot = next((p for p in pending if 'reasoner' in p['calls']), None)
    if pilot:
        p = predict(xs[pilot['id']], pilot, client)
        d.save(checkpoints / f"{p['id']}.json", p)
        done[p['id']] = p
        pending = [v for v in pending if v['id'] != p['id']]
    last = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs = {pool.submit(predict, xs[p['id']], p, client): p['id'] for p in pending}
        for f in futures.as_completed(fs):
            p = f.result()
            d.save(checkpoints / f"{p['id']}.json", p)
            done[p['id']] = p
            now = time.perf_counter()
            if now - last >= 20 or len(done) == len(old):
                print(json.dumps({'split': split, 'done': len(done), 'total': len(old),
                                  'elapsed_s': round(now-start, 1),
                                  'incremental_estimated_cny': round(client.spent-cost_before, 4)}), flush=True)
                last = now
    ps = [done[p['id']] for p in old]
    d.jsonl(folder / 'predictions.jsonl', ps)
    d.jsonl(OUT / f'{split}_predictions.jsonl', [
        {'id': p['id'], 'gold': p['gold'], 'prediction': p['E3'],
         'votes': p['votes'], 'original_route': p['original_route']} for p in ps])
    d.save(OUT / f'{split}_metrics.json', {**d.metrics(ps, 'E3'), 'split': split, 'experiment': 'E3'})
    log = folder / 'execution_sessions.json'
    sessions = d.read(log) if log.exists() else []
    sessions.append({'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     'new_wall_seconds': time.perf_counter()-start,
                     'new_unique_requests': len(set(client.cache.glob('*.json'))-raw_before),
                     'incremental_cost_cny_estimate': client.spent-cost_before})
    d.save(log, sessions)
    print(json.dumps({'complete': split, 'E1': d.metrics(ps, 'E1'), 'E3': d.metrics(ps, 'E3')}), flush=True)
    return ps


def summarize():
    table = []
    diagnostics = {}
    for split in SPLITS:
        ps = d.rows(OUT / split / 'predictions.jsonl')
        for e in ('E0', 'E1', 'E3'):
            table.append({'split': split, 'experiment': e, **d.metrics(ps, e)})
        diagnostics[split] = {
            'transitions': dict(collections.Counter(f"gold={p['gold']},E1={p['E1']},E3={p['E3']}" for p in ps)),
            'by_original_route': {r: {'n': len(z), 'E1': d.metrics(z, 'E1'), 'E3': d.metrics(z, 'E3')}
                                  for r in sorted({p['original_route'] for p in ps})
                                  if (z := [p for p in ps if p['original_route'] == r])},
            'by_topic': {t: {'E1': d.metrics(z, 'E1'), 'E3': d.metrics(z, 'E3')}
                         for t in sorted({p['topic'] for p in ps})
                         if (z := [p for p in ps if p['topic'] == t])},
        }
    with (OUT / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    d.save(OUT / 'diagnostics.json', diagnostics)
    raw = [d.read(p) for p in (OUT / 'raw').glob('*.json')]
    attempts = [a for r in raw for a in r['attempts']]
    d.save(OUT / 'cost_ledger.json', {
        'scope': 'Only newly requested E3 reasoner calls; upstream responses were reused.',
        'unique_requests': len(raw), 'attempts': len(attempts),
        'http_network_failed_attempts': sum('error' in a for a in attempts),
        'incremental_cost_cny_estimated_peak': sum(a.get('cost_cny', 0) for a in attempts),
        'total_serial_seconds': sum(a['seconds'] for a in attempts),
        'actual_models': dict(collections.Counter(a['response'].get('model', '') for a in attempts if 'response' in a)),
        'pricing_cny_per_million': {'cache_hit': .04, 'cache_miss': 2, 'completion': 8},
        'usage': dict(sum((collections.Counter({k:v for k,v in a.get('response', {}).get('usage', {}).items()
                                               if isinstance(v, (int, float))}) for a in attempts), collections.Counter())),
        'cost_note': 'Estimate using the original protocol peak prices; not an account bill.',
    })


def verify():
    cfg = d.read(CONFIG)
    assert cfg['input_sha256'] == frozen_inputs()
    for split in SPLITS:
        xs, old = inputs(split)
        ps = d.rows(OUT / split / 'predictions.jsonl')
        assert [p['id'] for p in ps] == [p['id'] for p in old]
        for p, b in zip(ps, old):
            assert p['gold'] == b['gold'] and p['E1'] == b['E1'] and p['votes'] == b['votes']
            for key in ('explorer1', 'explorer2', 'explorer3', 'mediator'):
                assert p['calls'].get(key) == b['calls'].get(key)
            if 'reasoner' not in p['calls']:
                assert p['E3'] == b['E1']
                continue
            call = p['calls']['reasoner']
            raw = d.read(OUT / 'raw' / (call['request_hash']+'.json'))
            assert d.digest(raw['request']) == call['request_hash']
            request = raw['request']
            assert request['messages'] == [{'role': 'system', 'content': d.SYSTEM},
                                            {'role': 'user', 'content': reasoner(xs[p['id']], p['mediators'])}]
            assert 'bi_causal_label' not in json.dumps(request)
            baseline = d.read(d.ROOT/'results/raw'/(b['calls']['reasoner']['request_hash']+'.json'))['request']
            assert {k:v for k,v in request.items() if k != 'messages'} == {k:v for k,v in baseline.items() if k != 'messages'}
            response = raw['attempts'][-1].get('response')
            if response:
                choice = response['choices'][0]
                try:
                    if choice['finish_reason'] != 'stop':
                        raise ValueError('incomplete')
                    checked = validate_call({'status': 'ok', 'parsed': d.parse(choice['message']['content'], 'reasoner')})
                    expected = int(checked['parsed']['Answer'] == 'Yes') if checked['status'] == 'ok' else None
                except (ValueError, KeyError, TypeError):
                    expected = None
                assert p['E3'] == expected
            else:
                assert p['E3'] is None
        m = d.read(OUT / f'{split}_metrics.json')
        assert all(m[k] == v for k,v in d.metrics(ps, 'E3').items())
    d.save(OUT/'verification.json', {'status':'PASS', 'checks':[
        'frozen original code/config/data/predictions unchanged', 'exact validation and test coverage',
        'upstream responses identical', 'request hashes and prompt reconstruction',
        'generation parameters match E1', 'raw-response predictions and metrics recomputed'],
        'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('PASS: E3 provenance, unchanged upstream, full coverage, raw predictions and metrics', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'verify'])
    args = parser.parse_args()
    if args.command == 'verify':
        verify(); return
    cfg = prepare()
    client = d.Client()
    client.cache = OUT / 'raw'
    client.cache.mkdir(parents=True, exist_ok=True)
    client.spent += sum(a.get('cost_cny', 0) for p in client.cache.glob('*.json')
                        for a in d.read(p)['attempts'])
    workers = cfg['generation_protocol']['workers']
    run_split('validation', client, workers)
    freeze = OUT / 'validation_completed.json'
    if not freeze.exists():
        d.save(freeze, {'config_sha256': d.digest(CONFIG.read_bytes()),
                        'validation_predictions_sha256': d.digest((OUT/'validation/predictions.jsonl').read_bytes()),
                        'decision': 'Evaluate the one prespecified E3 variant without changing its configuration.',
                        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    assert d.read(freeze)['config_sha256'] == d.digest(CONFIG.read_bytes())
    run_split('test', client, workers)
    summarize()
    verify()


if __name__ == '__main__':
    main()
