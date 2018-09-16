from src.structs.node import Node
from src.geomType import GeomType
import json


ROOT = (2, 1, 0)
# test points data
SEGMENTS = [
    ([10, 10, 0], [10, 5, 0]),
    ([10, 10, 0], [2, 10, 0]),
    ([4, 9.5, 0],  [4, 4, 0]),
    ([4, 4, 0], [4, 7, 0]),
    ([4, 4.5, 0], [4, 6, 0]),
    ([2, 1, 0], [2, 5, 0]),
    ([2, 5, 0], [4, 5, 0]),
    ([6, 10, 0], [6, 12, 0]),
    ([8, 10, 0], [8, 8, 0]),
    ([6, 8, 0], [9, 8, 0]),
    ([9, 8, 0], [9, 5, 0]),
    ([6, 8, 0], [6, 4, 0]),
]

SEGMENTS2 = [
    [[10, 10, 0], [10, 5, 0]],
    [[10, 10, 0], [2, 10, 0]],
    [[4, 9.5, 0],  [4, 4, 0]],
    [[2, 1, 0], [2, 5, 0]],
    [[2, 5, 0], [4, 5, 0]],
    [[6, 10, 0], [6, 12, 0]],
    [[8, 10, 0], [8, 8, 0]],
]

SEGMENTS_COL = [
    [[10, 10, 0], [10, 7, 0]],
    [[10, 10, 0], [10, 0, 0]],
    [[10, 10, 0], [5, 10, 0]],
    [[10, 10, 0], [10, 15, 0]],
    [[2., 1., 0.], [5, 10, 0]],
    [[10, 7, 0], [10, 15, 0]],
]
ROOT_COL = [5, 10, 0]

SYMBOLS = [
    [6, 12, 0]
]

# test request args
ARGS1 = {

}


# test components for graph isomorphisms
COMPONENTS = [
    {'name': 'vbranch',
     '$input_node':
         {'connects':
             {'$node1':
                  {'symbol': '&any',
                   'connects':
                    {'$output_node1': {'angle': 90},
                     '$output_node2': {}}
              }}}

     }

]


def load_segs(fl='data1'):

    with open('./data/{}.json'.format(fl), 'r') as f:
        xs = json.load(f)
        # print(xs)
        if len(xs) == 1:
            segs = json.loads(xs['data'])
            xs = segs[0]['children']
        f.close()
    return xs


def vertical_branch():
    # nin1 = Node((0, 0, 0))
    nin = Node((0, 0, 0))
    nb = Node((0, 6, 0), type=GeomType(5), symbol_id=1)
    no1 = Node((0, 10, 0))
    no2 = Node((8, 6, 0), type=GeomType(5), symbol_id=2)
    # nin1.connect_to(nin)
    nin.connect_to(nb)
    nb.connect_to(no1)
    nb.connect_to(no2)
    return nin


from src import System, KB, RuleEngine, RenderNodeSystem, SystemFactory
from src.rules import heursitics


def get_rendered():
    data = load_segs(fl='1535158393.0-revit-signal')
    system = SystemFactory.from_serialized_geom(
        data, sys=System, root=(-246, 45, 0))
    system = system.bake()
    rules = heursitics.EngineHeurFP()
    Eng = RuleEngine(term_rule=rules.root, mx=2500, debug=False, nlog=20)
    Kb = KB(rules.root)
    root = Eng.alg2(system.root, Kb)

    renderer = RenderNodeSystem()
    root = renderer.render(root)
    return root

