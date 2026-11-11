"""Run a NEW, unlabelled example live; never modifies frozen test predictions."""
import argparse, json, time
import dreci as d

p=argparse.ArgumentParser()
p.add_argument('--text',default='Heavy rain caused flooding and the flooding blocked the road .')
p.add_argument('--source',default='rain')
p.add_argument('--target',default='blocked')
a=p.parse_args();words=a.text.split()
def locate(t):
    ts=t.split();hits=[list(range(i,i+len(ts))) for i in range(len(words)-len(ts)+1) if words[i:i+len(ts)]==ts]
    if len(hits)!=1:raise ValueError('Trigger must occur exactly once as a whitespace-delimited span.')
    return hits[0]
x={'item_id':-1,'words':words,'events':[locate(a.source),locate(a.target)],'bi_causal_label':None,'doc_name':'unlabelled_live_demo','topic':'demo'}
out=d.predict(x,d.Client())
d.save(d.ROOT/'results/demo/live_trace.json',{'input':d.context(x),'trace':out,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'notice':'New unlabelled demonstration. Excluded from evaluation. Repeated identical requests replay cache.'})
print(json.dumps({'input':d.context(x),'trace':out},ensure_ascii=False,indent=2))
