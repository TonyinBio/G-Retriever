import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

import time

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def step_one():
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split():
    
    dataset = load_dataset("rmanluo/RoG-webqsp")
    last_index = 1748
    
    all_indices = np.arange(last_index + 1)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(all_indices, [0.6, 0.06, 0.34])

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two():
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(len(dataset))):

        # nodes
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')

        nodes.fillna({"node_attr": ""}, inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))

if __name__ == '__main__':
    generate_split()
