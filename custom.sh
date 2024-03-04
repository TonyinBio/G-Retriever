python -m src.dataset.preprocess.expla_graphs
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat

python train.py --dataset expla_graphs --model_name graph_llm