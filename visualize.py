from pyvis.network import Network
import json
import pandas as pd
from io import StringIO

def visualize_jsonl(triplets, fname):
    net = Network()

    nodes = {}
    edges = []
    for (h, r, t) in triplets:
        nodes[h] = nodes.get(h, 0) + 1
        nodes[t] = nodes.get(t, 0) + 1

    for node, value in nodes.items():
        net.add_node(node, label=node)
    for (h, r, t) in triplets:
        net.add_edge(h, t, title=r)

    net.show_buttons(filter_=['physics'])
    net.show(f"{fname}.html", notebook=False)

def process(fname):
    if ".txt" in fname:
        nodes = {}
        with open(fname) as file:
            at_edge_table = False
            file.readline()
            while not at_edge_table:
                line = file.readline().strip()
                at_edge_table = len(line) == 0
                if not at_edge_table:
                    id, name = line.split(",")
                    id = int(id)
                    nodes[id] = name

            data = pd.read_csv(file)

        triplets = data.apply(lambda s: [nodes[s["src"]], s["edge_attr"], nodes[s["dst"]]], axis=1).tolist()
    else:
        with open(fname) as file:
            triplets = [json.loads(line) for line in file.readlines()]
            
    # return triplets
    return triplets
if __name__ == "__main__":
    fname = input("File name: ")
    # fname = "916.txt"
    triplets = process(fname)
    visualize_jsonl(triplets, fname)