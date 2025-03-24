import argparse, pickle, os, json, sys
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str, default='agentica-org/DeepScaleR-Preview-Dataset')
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="/home/myang4/TinyZero/data/DeepScaleR-Preview-Dataset")
    parser.add_argument("-K", type=int, default=16)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p parameter for sampling from the LLM.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)

    dataset = load_dataset(args.input_dataset_name, split=args.dataset_split)
    
    if args.dataset_start > len(dataset):
        exit()
    args.dataset_end = min(args.dataset_end, len(dataset))
    print(args.dataset_start, args.dataset_end)

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=20000, disable_custom_all_reduce=True)
    tokenizer = llm.get_tokenizer()

    output = {}

    for n in tqdm(range(args.dataset_start, args.dataset_end, args.batch_size)):

        batch_start = n
        batch_end = min(n + args.batch_size, args.dataset_end)
        
        problems = [[{"role": "user", "content": dataset[n]['problem']}] for n in range(batch_start, batch_end)]
        
        rollouts = llm.chat(problems,
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            ),
        )

        for i in range(batch_start, batch_end):
            output[i] = rollouts[i - n]

    with open(os.path.join(args.output_path, f'pass_at_k_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
        pickle.dump(output, f)

    print(output)