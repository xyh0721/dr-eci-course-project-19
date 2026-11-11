import unittest, json, tempfile, os
from pathlib import Path
from unittest.mock import patch
import dreci as d

X={'item_id':7,'words':['The','blast','caused','damage','.'],'events':[[1],[3]],'bi_causal_label':1,'doc_name':'d','topic':'t'}
class PipelineTest(unittest.TestCase):
    def test_strict_parse(self):
        for s in ['maybe','{"Answer":"maybe"}','["Yes"]','{}']:
            with self.assertRaises((ValueError,TypeError)):d.parse(s,'explorer')
        self.assertEqual(d.parse('```json\n{"Answer":"No"}\n```','explorer')['Answer'],'No')
    def test_prompt_has_no_gold(self):
        a=d.explorer(X,0);b=d.explorer({**X,'bi_causal_label':0,'tri_causal_label':2},0)
        self.assertEqual(a,b);self.assertNotIn('bi_causal_label',a)
    def test_mediator_filter(self):
        s=json.dumps({'source_related':[],'target_related':[],'mediators':['blast','invented','caused']})
        a=d.parse(s,'mediator',X);self.assertEqual(a['mediators'],['caused']);self.assertEqual(len(a['rejected_mediators']),2)
    def test_branch_and_failures(self):
        class Fake:
            z=[]
            def call(self,p,kind,x=None):
                if kind=='explorer':return {'status':'ok','parsed':{'Answer':'Yes'}}
                if kind=='mediator':return {'status':'ok','parsed':{'mediators':self.z}}
                return {'status':'ok','parsed':{'Answer':'Yes','matched_rules':[6] if self.z else [1],'evidence':'blast caused damage'}}
        f=Fake();a=d.predict(X,f);self.assertEqual((a['route'],a['E1']),('direct',1))
        f.z=['caused'];b=d.predict(X,f);self.assertEqual((b['route'],b['E1']),('indirect',1))
    def test_gate_and_ablation(self):
        class No:
            def call(self,p,kind,x=None):
                self_kind=kind
                if self_kind!='explorer':raise AssertionError('gate should stop')
                return {'status':'ok','parsed':{'Answer':'No'}}
        a=d.predict(X,No());self.assertEqual(a['E1'],0)
        z=d.ablate([{'votes':1,'E1':1,'route':'direct'}],2);self.assertEqual(z[0]['E2'],0)
    def test_failed_positive_is_not_dropped(self):
        ps=[{'gold':1,'p':1},{'gold':1,'p':None},{'gold':0,'p':0},{'gold':0,'p':None}]
        a=d.metrics(ps,'p');self.assertEqual(a['n'],4);self.assertEqual(a['FN'],1);self.assertEqual(a['accuracy'],.5);self.assertEqual(a['coverage'],.5)
    def test_split_integrity(self):
        a=d.rows(d.ROOT/'data_info/split_manifest.jsonl')
        for field in ['doc','input_hash']:
            seen={}
            for x in a:
                self.assertEqual(seen.setdefault(x[field],x['split']),x['split'])
    def test_persistent_cache_has_no_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(d,'ROOT',Path(tmp)):
                d.save(Path(tmp)/'configs/protocol.json',{'budget_cny':1,'base_url':'https://api.deepseek.com','model':d.MODEL})
                c=d.Client();body={'model':d.MODEL,'messages':[{'role':'system','content':d.SYSTEM},{'role':'user','content':'q'}],'temperature':0,'thinking':{'type':'disabled'},'response_format':{'type':'json_object'},'max_tokens':24}
                h=d.digest(body);d.save(c.cache/(h+'.json'),{'attempts':[{'seconds':1,'cost_cny':0,'response':{'choices':[{'finish_reason':'stop','message':{'content':'{"Answer":"Yes"}'}}]}}]})
                self.assertEqual(c.call('q','explorer')['status'],'ok')
                self.assertNotIn('Authorization',(c.cache/(h+'.json')).read_text())

if __name__=='__main__':unittest.main(verbosity=2)
