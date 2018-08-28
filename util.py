import operator

LARGENUMBER = 100000000000

class EmulatesInt(object):
    def __init__(self):
        pass

    def operator(self, other, operation):
        if hasattr(other, "cost"):
            return operation(self.cost, other.cost)
        else:
            return operation(self.cost, other)

    def __gt__(self, other):
        return self.operator(other, operator.gt)

    def __ge__(self, other):
        return self.operator(other, operator.ge)

    def __lt__(self, other):
        return self.operator(other, operator.lt)

    def __le__(self, other):
        return self.operator(other, operator.le)

    def __eq__(self, other):
        return self.operator(other, operator.eq)

    def  __add__(self, other):
        return self.operator(other, operator.add)
        
    def __mul__(self, other):
        return self.operator(other, operator.mul)
        
    def __sub__(self, other):
        return self.operator(other, operator.sub)
        
    def __mod__(self, other):
        return self.operator(other, operator.mod)


class Action(object):
    FW = 0
    BW = 1
    CPSAVE = 2
    CPLOAD = 3
    CPDEL = 4
    _types = [FW, BW, CPSAVE, CPLOAD, CPDEL]
    _type_names = {FW: 'Forward', BW: 'Backward', CPSAVE:'Save Checkpoint', CPLOAD: 'Load Checkpoint', CPDEL: 'Delete Checkpoint'}
    def __init__(self, action_type, node):
        assert(action_type in self._types)
        self.type = action_type
        self.node = node

    @property
    def cost(self):
        if self.type in (self.FW, self.BW):
            # TODO: Separate forward and backward costs
            return self.node['compute_cost']
        else:
            # TODO: Cost of saving/loading checkpoint
            return 0

    @property
    def mem(self):
        assert(self.type in (self.CPSAVE, self.CPLOAD, self.CPDEL))
        return self.node['output_size']

    def __str__(self):
        return "%s (%s)" % (self._type_names[self.type], self.index)

    def __eq__(self, other):
        return self.type == other.type and self.index == other.index
        
class Schedule(EmulatesInt):
    is_Feasible = True
    def __init__(self, actions = None):
        self.actions = actions or []

    def add_action(self, node, kind):
        action = Action(kind, node)
        
        action.index = node['index']
        if len(self.actions) > 0 and action == self.actions[-1]:
            # Ignore repeated actions
            return
        if kind is Action.FW:
            # First step OR
            # Last action corresponded to the step before this one: either a forward or a cp load
            assert(((len(self.actions) == 0) or (self.actions[-1].index == (action.index - 1))))
        elif kind is Action.BW:
            # First action can't be backward
            assert(len(self.actions) > 0)
            # EITHER
            # 1. Last execute was a forward
            last_executed1 = self.actions[-1].type == Action.FW
            # 2. Of the same index
            last_executed2 = self.actions[-1].index == action.index
            # OR
            # 1. A checkpoint for this index was previously saved
            previously_saved = len([x for x in self.actions if x.type==Action.CPSAVE and x.index==action.index]) == 1
            # 2. And not deleted
            previously_saved2 = len([x for x in self.actions if x.type==Action.CPDEL and x.index==action.index]) == 0
            # 3. Unless the deletion was the last thing that happened
            assert((last_executed1 and last_executed2) or (previously_saved and previously_saved2))
        elif kind is Action.CPSAVE:
            # If the last thing we did was to load this checkpoint, and we're being asked to save it, ignore
            if len(self.actions) > 0 and self.actions[-1].type == Action.CPLOAD and self.actions[-1].index == action.index:
                return
            assert((len(self.actions) == 0) or self.actions[-1].type == Action.FW)
        elif kind is Action.CPLOAD:
            previously_saved = len([x for x in self.actions if x.type==Action.CPSAVE and x.index==action.index]) == 1
            previously_saved2 = len([x for x in self.actions if x.type==Action.CPDEL and x.index==action.index]) == 0
            assert(previously_saved and previously_saved2)
        elif kind is Action.CPDEL:
            assert(len([x for x in self.actions if x.type==Action.CPSAVE and x.index==action.index]) == 1)
        else:
            assert(False)
        self.actions.append(action)

    def add_action_fw(self, node):
        self.add_action(node, Action.FW)

    def add_action_bw(self, node):
        self.add_action(node, Action.BW)

    def add_action_cpsave(self, node):
        self.add_action(node, Action.CPSAVE)

    def add_action_cpload(self, node):
        self.add_action(node, Action.CPLOAD)

    def add_action_cpload_final(self, node):
        self.add_action(node, Action.CPLOAD)

    def add_action_cpdel(self, node):
        self.add_action(node, Action.CPDEL)

    def merge_with_checkpoint(self, other):
        if not other.is_Feasible:
            return InfeasibleSchedule()
        forward_part, backward_part = self.split_forward_backward()
        s = Schedule()
        for a in forward_part:
            s.add_action(a.node, a.type)
        if forward_part[-1].type == Action.FW:
            s.add_action_cpsave(forward_part[-1].node)
        for a in other.actions:
            s.add_action(a.node, a.type)
        s.add_action_cpload(backward_part[0].node)
        for a in backward_part:
            s.add_action(a.node, a.type)
        return s

    @property
    def cost(self):
        return sum(x.cost for x in self.actions)

    @property
    def peakmemory(self):
        peakmem = 0
        mem = 0
        for a in self.actions:
            if a.type == Action.CPSAVE:
                mem += a.mem
                if mem > peakmem:
                    peakmem = mem
            elif a.type == Action.CPDEL:
                mem -= a.mem
        return peakmem

    def __str__(self):
        return "-->".join(str(x) for x in self.actions)

    def split_forward_backward(self):
        first_backward_index = [a.type for a in self.actions].index(Action.BW)
        return self.actions[:first_backward_index], self.actions[first_backward_index:]
            

class InfeasibleSchedule(Schedule):
    is_Feasible = False
    @property
    def cost(self):
        return LARGENUMBER

    def __str__(self):
        return "(Infeasible Schedule)"
