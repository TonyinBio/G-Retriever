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
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.collate import collate_fn

import json

PATH = "dataset/webqsp"
PATH_NODES = f'{PATH}/nodes'
PATH_EDGES = f'{PATH}/edges'
PATH_GRAPHS = f'{PATH}/graphs'
LM_MODEL_NAME = "sbert"

def main(args):
    args.dataset = "webqsp"  # parameter
    print(args)

    # build model
    dataset = load_dataset[args.dataset]()  
    
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

    # load checkpoint
    model = _reload_best_model(model, args)

    # run
    model.eval()
    with torch.no_grad():
        user_question = ""
        while user_question != "exit":
            graph_id = int(input("Choose a graph: "))
            graph_file = f"graph{graph_id}.jsonl"
            open(graph_file, "w").write("\n".join([json.dumps(triplet) for triplet in dataset.dataset[graph_id]["graph"]]))
            print(f"Printed graph to {graph_file}")
            print("Example question:", dataset.dataset[graph_id]["question"])
            print("Example answer:", dataset.dataset[graph_id]["answer"])
            user_question = input("Ask a question: ")

            batch = assemble_batch(user_question, graph_id)

            text_subg = batch["desc"][0]
            sgraph_file = f"sgraph{graph_id}_{user_question}.txt"
            open(sgraph_file, "w").write(text_subg)
            print(f"Printed subgraph to {sgraph_file}")
            output = model.inference(batch)
            print()
            print("Output:")
            print(output["pred"])
            # print("Ground Truth:")
            # print(output["label"])

def assemble_batch(user_question, graph_id):
    question = f'Question: {user_question}\nAnswer: '

    graph = torch.load(f'{PATH_GRAPHS}/{graph_id}.pt')
    nodes = pd.read_csv(f'{PATH_NODES}/{graph_id}.csv')
    edges = pd.read_csv(f'{PATH_EDGES}/{graph_id}.csv')

    model, tokenizer, device = load_model_lm[LM_MODEL_NAME]()
    text2embedding = load_text2embedding[LM_MODEL_NAME]
    q_emb = text2embedding(model, tokenizer, device, user_question)

    subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)

    return collate_fn([{
        'id': 0,
        'question': question,
        'label': "No label",
        'graph': subg,
        'desc': desc,
    }])

if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
