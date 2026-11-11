import copy
import unittest
import dreci as d
import e3

X = {'item_id': 7, 'words': ['The', 'fire', 'destroyed', 'the', 'shop', '.'],
     'events': [[1], [2]], 'bi_causal_label': 1}


class E3Test(unittest.TestCase):
    def test_prompt_changes_only_rules_and_never_uses_gold(self):
        for z in ([], ['shop']):
            old_rules = d.INDIRECT if z else d.DIRECT
            prompt = e3.reasoner(X, z)
            self.assertEqual(prompt.replace(d.DIRECT+'\n'+d.INDIRECT, old_rules, 1), d.reasoner(X, z))
            self.assertEqual(prompt, e3.reasoner({**X, 'bi_causal_label': 0}, z))

    def test_direct_rule_allowed_with_mediator_and_upstream_not_called(self):
        class Fake:
            stop = False
            def call(self, prompt, kind):
                assert kind == 'reasoner'
                return {'status':'ok', 'parsed':{'Answer':'Yes', 'matched_rules':[1], 'evidence':'fire destroyed shop'}}
        old = {'id':7, 'gold':1, 'E0':1, 'E1':None, 'votes':3, 'route':'indirect',
               'mediators':['shop'], 'calls':{'mediator':{'parsed':{'mediators':['shop']}},
                                             'reasoner':{'request_hash':'old'}}}
        saved = copy.deepcopy(old)
        out = e3.predict(X, old, Fake())
        self.assertEqual(out['E3'], 1)
        self.assertEqual(old, saved)
        self.assertEqual(out['calls']['mediator'], old['calls']['mediator'])

    def test_gate_and_failure_accounting(self):
        class Never:
            def call(self, *args):
                raise AssertionError('Gate-negative sample must not call API')
        old = {'E1':0, 'votes':0, 'route':'gate_no', 'calls':{}}
        self.assertEqual(e3.predict(X, old, Never())['E3'], 0)
        a = e3.validate_call({'status':'ok', 'parsed':{'Answer':'Yes', 'matched_rules':[]}})
        self.assertEqual(a['status'], 'failure')
        self.assertEqual(d.metrics([{'gold':1, 'E3':None}], 'E3')['FN'], 1)


if __name__ == '__main__':
    unittest.main()
