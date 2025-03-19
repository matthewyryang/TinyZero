import argparse, pickle, os, sys
import pandas as pd
from vllm import LLM, SamplingParams
from collections import defaultdict

sys.path.append(os.path.abspath("/home/myang4/TinyZero/verl/utils/reward_score"))
from countdown import compute_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--checkpoint', type=int)
    args = parser.parse_args()

    dataset = pd.read_parquet("/home/myang4/TinyZero/data/countdown-3-4-5-6/test.parquet")

    if args.checkpoint == 0:
        model_checkpoint = "/data/group_data/rl/myang4/countdown/cognitive-behaviors-Qwen2.5-3B/global_step_75"
        # model_checkpoint = "/data/user_data/myang4/countdown/countdown-equal-512/actor/global_step_150"
    else:
        model_checkpoint = f"/data/user_data/myang4/countdown/countdown-3-4-5-6-{args.context_length}/actor/global_step_{args.checkpoint}"
        # model_checkpoint = f"/data/user_data/myang4/countdown/countdown-{args.context_length}-length-curriculum/actor/global_step_{args.checkpoint}"
    
    llm = LLM(model_checkpoint)
    tokenizer = llm.get_tokenizer()

    rewards_by_difficulty = defaultdict(list)
    correct_lengths_by_difficulty = defaultdict(list)
    incorrect_lengths_by_difficulty = defaultdict(list)

    difficulties = range(3, 7)

    full_results = []
    for i, difficulty in enumerate(difficulties):
        subset = dataset[i * 1000 : (i + 1) * 1000]
        prompts = list(subset['prompt'].map(lambda x: list(x)))

        completions = llm.chat(prompts, sampling_params=SamplingParams(temperature=0, max_tokens=args.context_length))

        rewards = []
        correct_lengths = []
        incorrect_lengths = []
        
        for j in range(len(completions)):
            completion = completions[j].prompt + completions[j].outputs[0].text
            reward = compute_score(completion, subset['reward_model'][i * 1000 + j]['ground_truth'], do_print=False, format_score=0)
            
            if reward:
                correct_lengths.append(len(tokenizer.encode(completions[j].outputs[0].text)))
            else:
                incorrect_lengths.append(len(tokenizer.encode(completions[j].outputs[0].text)))

            rewards.append(reward)

            full_results.append({
                'completion': completion,
                'reward': reward,
                'length': len(tokenizer.encode(completions[j].outputs[0].text)),
                'target': subset['reward_model'][i * 1000 + j]['ground_truth']['target'],
                'nums': subset['reward_model'][i * 1000 + j]['ground_truth']['numbers'],
                'difficulty': len(subset['reward_model'][i * 1000 + j]['ground_truth']['numbers'])
            })
        
        rewards_by_difficulty[difficulty] = rewards
        correct_lengths_by_difficulty[difficulty] = correct_lengths
        incorrect_lengths_by_difficulty[difficulty] = incorrect_lengths


    mean_rewards = []
    mean_correct_lengths = []
    mean_incorrect_lengths = []
    mean_lengths = []

    for difficulty in difficulties:
        mean_rewards.append(sum(rewards_by_difficulty[difficulty]) / 1000)
        mean_lengths.append((sum(correct_lengths_by_difficulty[difficulty]) + sum(incorrect_lengths_by_difficulty[difficulty])) / 1000)
        
        if len(correct_lengths_by_difficulty[difficulty]) > 0:
            mean_correct_lengths.append(sum(correct_lengths_by_difficulty[difficulty]) / len(correct_lengths_by_difficulty[difficulty]))
        else:
            mean_correct_lengths.append(-1)
        
        if len(incorrect_lengths_by_difficulty[difficulty]) > 0:
            mean_incorrect_lengths.append(sum(incorrect_lengths_by_difficulty[difficulty]) / len(incorrect_lengths_by_difficulty[difficulty]))
        else:
            mean_incorrect_lengths.append(-1)
    
    # save results
    results = {
        'difficulty': list(difficulties),
        'mean_reward': mean_rewards,
        'mean_correct_length': mean_correct_lengths,
        'mean_incorrect_length': mean_incorrect_lengths,
        'mean_length': mean_lengths
    }

    output_dir = f"/home/myang4/TinyZero/evaluate/outputs/context-length-{args.context_length}-{args.checkpoint}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(output_dir, "full_results.pkl"), 'wb') as f:
        pickle.dump(full_results, f)
