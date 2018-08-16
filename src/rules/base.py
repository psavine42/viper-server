import networkx as nx


class Heuristic(object):
    def __init__(self):
        self._stats = {}
        self._heurstics = {}
        self._heurdict = {}

    def get_rules(self, tup):
        kys = self._heurdict.get(tup, [])
        for k in kys:
            if k in self._heurstics:
                yield (k, self._heurstics.get(k))

    def record(self, *args, **kwargs):
        pass

    def _has_subtree_of(self):
        return

    def apply(self, system):
        G = system.G
        source = system.root
        seen = set()
        nx.set_node_attributes(G, {source: {'label': 'source'}})
        q = source if isinstance(source, list) else [source]
        while q:
            el = q.pop(0)
            if el not in seen:
                seen.add(el)
                pred = list(set(G.predecessors(el)).difference([el]))
                sucs = list(set(G.successors(el)).difference([el]))

                lsize = (len(pred), len(sucs))

                for k, fn in self.get_rules(lsize):
                    if fn is None:
                        continue
                    res = fn(G, el, pred, sucs, seen)
                    if res is True:
                        nx.set_node_attributes(G, {el: {'label': k}})

                q.extend(set(pred + sucs).difference(seen))
        system.G = G
        return system

    def __call__(self, system):
        return self.apply(system)


def edge_attr(G, edge, attr, d=None):
    return G.edges[edge].get(attr, d)



"""

    +----------KB
    |
conditions -> rule firing -> True, False
    |
    +----- 

inputs:
    ISSYMBOL(node)  

Graph Structure:
    
    TotalSuccs(edge)        -> Number of reachable nodes
    ISSPLIT(node)           -> Nsucs(node) > 1, Npreds(node) = 1 
    
    ISMAIN(edge)            -> not ISBRANCH
    ISBRANCH(node, edge)    -> totalsucs(edge) < Totalsucs( ) and ISSPLIT( )
    
    NPreds(node or edge)    ->
    NSuccs(node or edge)    -> 
    ISEND(node or edge)     -> NSuccs == 0
    
    HBranch(node)       -> ISSPLIT and ISSYMBOL
    VBranch(node)       -> ISSPLIT 
    
    DropHEAD    ->     
    VertHEAD    -> 
    HorzHEAD    ->     

Geometric Structure
    DirectionSame(edge1, edge2)
    
    
    
Logic:
    
    Riser   := ISSYMBOL(node)
    Tee     := 


    
                            
"""



