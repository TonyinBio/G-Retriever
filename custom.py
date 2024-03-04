import os
import wandb
import gc
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _reload_best_model

import pandas as pd

from torch_geometric.data.data import Data

from src.utils.lm_modeling import load_model as load_model_lm
from src.utils.lm_modeling import load_text2embedding
from src.utils.collate import collate_fn

def main(args):
    print(args)

    # build model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    graph_type = 'Explanation Graph'
    prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
    model = load_model[args.model_name](graph_type=graph_type, args=args, init_prompt=prompt)

    # load checkpoint
    args.dataset = "expla_graphs"
    model = _reload_best_model(model, args)

    # run
    model.eval()
    with torch.no_grad():
        print("Using the following prompt:")
        print('Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
)
        # input 1/3
        text = {
            "arg1": "Cannabis should be legal.",
            "arg2": "It's not a bad thing to make marijuana more available.",
            "label": "support"
        }
        
        arg1 = input("Input argument 1: ")
        arg2 = input("Input argument 2: ")
        text["arg1"] = arg1
        text["arg2"] = arg2

        print("Using cannabis knowledge graph")
        # input 2/3
        nodes = pd.DataFrame({
            "node_id": [0, 1, 2, 3, 4],
            "node_attr": ["cannabis", "marijuana", "legal", "more available", "good thing"]
        })

        # input 3/3
        edges = pd.DataFrame({
            "src": [0, 2, 1, 4],
            "edge_attr": ["synonym of", "causes", "capable of", "desires"],
            "dst": [1, 3, 4, 2]
        })

        batch = assemble_batch(text, nodes, edges, prompt=prompt)
        output = model.inference(batch)
        print()
        print("Output:")
        print(output["pred"])
        # print("Ground Truth:")
        # print(output["label"])

def assemble_batch(text, nodes, edges, prompt=None):
    model_name = "sbert"
    prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'

    model, tokenizer, device = load_model_lm[model_name]()
    text2embedding = load_text2embedding[model_name]
    x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
    e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
    edge_index = torch.LongTensor([edges.src, edges.dst])
    graph = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))


    question = f'Argument 1: {text["arg1"]}\nArgument 2: {text["arg2"]}\n{prompt}'

    # to add to hard prompt
    desc = nodes.to_csv(index=False) + '\n' + edges.to_csv(index=False)

    return collate_fn([{
        'id': 0,
        # 'label': text['label'],
        'label': "No Label",
        'desc': desc,  # hard prompt p1
        'graph': graph,  # soft prompt
        'question': question,  # hard prompt p2
    }])

if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
