from multiprocessing import current_process, cpu_count
from multiprocessing import Manager, Process, Condition, Lock, Pool
from multiprocessing.managers import BaseManager
from datetime import datetime
import sys

from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import minimal_spanning_tree,\
shortest_path, heuristic_search, shortest_path_bellman_ford, maximum_flow, cut_tree

class Thing(object):
    mylist = []
    
    def __init__(self):
        self.mylist = [None,None,None,None]


class ScriptManager(BaseManager):
    pass

ScriptManager.register('Thing', Thing, exposed=['mylist'])

def process(shared_object,i):

    shared_object.mylist[i] = 1
    print shared_object.mylist[i]

if __name__=='__main__':

    gr = digraph()
    gr.add_nodes([0,1,2,3,4,5])
    gr.add_edge((0,1), wt=9)
    gr.add_edge((0,2), wt=9)
    gr.add_edge((0,3), wt=1)
    gr.add_edge((0,4), wt=1)

    gr.add_edge((1,5), wt=1)
    gr.add_edge((1,2), wt=9)
    
    gr.add_edge((2,5), wt=1)
    gr.add_edge((2,3), wt=1)
    
    gr.add_edge((3,5), wt=9)    
    gr.add_edge((3,4), wt=9)
    
    gr.add_edge((4,5), wt=9)
    flows, cuts = maximum_flow(gr, 0, 5)

    print cuts
    

    


