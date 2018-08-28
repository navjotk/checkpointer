""" Compute non uniform checkpointing schedules
"""

import argparse
import json
from util import Schedule, Action, InfeasibleSchedule
        

def constant_memory(chain):
    #print("Processing chain: %s" % str(chain))
    s = Schedule()
    processed_bw = 0
    first_node = {'index': chain.first['index'] - 1, 'output_size': chain.first['input_size']}
    s.add_action_cpsave(first_node)
    while processed_bw < chain.length():
        s.add_action_cpload(first_node)
        for n in chain.nodes[:chain.length()-processed_bw]:
            s.add_action_fw(n)
        processed_bw += 1
        #print("%d/%d steps processed" % (processed_bw, chain.length()))
        s.add_action_bw(chain.nodes[-processed_bw])
    #print("Done")
    return s


def cost(chain, memory, toplevel = True):
    if chain.t_memcost() < memory:
        s = Schedule()
        first_node = {'index': chain.first['index'] - 1, 'output_size': chain.first['input_size']}
        s.add_action_cpsave(first_node)
        for node in chain.nodes:
            s.add_action_fw(node)
            s.add_action_cpsave(node)
        for node in reversed(chain.nodes):
            s.add_action_cpload(node)
            s.add_action_bw(node)
            s.add_action_cpdel(node)
        assert(s.peakmemory < memory)
        return s
    if memory < chain.first['input_size']:
        return InfeasibleSchedule()
    schedules = []
    schedules.append(constant_memory(chain))
    for i in range(1, chain.length()):
        left, right = chain.split(i)
        leftschedule = cost(left, memory, False)
        assert(leftschedule.peakmemory < memory)
        rightschedule = cost(right, (memory - chain.memcost([i]) - chain.first['input_size']), False)
        if rightschedule.is_Feasible:
            assert(rightschedule.peakmemory < (memory - chain.memcost([i]) - chain.first['input_size']))
        totalschedule = leftschedule.merge_with_checkpoint(rightschedule)
        if totalschedule.peakmemory > memory:
            totalschedule = InfeasibleSchedule()
        schedules.append(totalschedule)
    assert(all([x.peakmemory < memory for x in schedules]))
    
    if toplevel:
        pass
    return min(schedules)

class Chain(object):
    def __init__(self, nodes):
        for i, step in enumerate(nodes):
            if i < len(nodes) - 1:
                assert(step['output_size']==nodes[i+1]['input_size'])
                step['next'] = nodes[i+1]
            else:
                step['next'] = None
        self.first = nodes[0]
        self.nodes = nodes

    def length(self):
        return len(self.nodes)

    def t_memcost(self):
        return sum([x['output_size'] for x in self.nodes]) + self.first['input_size']

    def memcost(self, cps):
        return sum([self.nodes[x]['output_size'] for x in cps])

    def t_comp_cost(self):
        return 2*sum([x['compute_cost'] for x in self.nodes])

    def split(self, y):
        assert(y > 0)
        assert(y < self.length())
        return Chain(self.nodes[0:y]), Chain(self.nodes[y:])

    def __str__(self):
        return " ".join(["--%d-->(%d)--%d-->" % (node['input_size'], node['compute_cost'], node['output_size']) for node in self.nodes])

parser= argparse.ArgumentParser(prog='dynamic.py', usage='python %(prog)s data.txt')
parser.add_argument ('data', type=str, help="Data file")
args = parser.parse_args()

memory_budget = 6100
with open(args.data) as f:
    data = json.load(f)

for i, step in enumerate(data):
    step['index'] = i

chain = Chain(data)

print("Minimum computational cost: %d" % chain.t_comp_cost())
print("Memory required for minimum cost: %d" % chain.t_memcost())
print("Given memory budget: %d"%memory_budget)

s = cost(chain, memory_budget)
print("Suggested schedule:")
print(s)
print('Schedule cost: %d' % s.cost)
print("Peak memory: %d" % s.peakmemory)


    
