import torch
import os
import json
import argparse
import gc
from dotenv import load_dotenv
from itertools import product
from utils import utils, load_model_pipeline, opensource_inference, closedsource_inference


def parse_args():
    parser = argparse.ArgumentParser(description="I2T ICL Inference")

    parser.add_argument(
        "--data_dir", default="./VL-ICL", type=str, help="Data directory."
    )
    parser.add_argument(
        "--caption_dir", default="./caption", type=str, help="Caption directory."
    )
    parser.add_argument(
        "--output_dir", default="./results", type=str, help="Output directory."
    )
    parser.add_argument(
        "--dataset",
        default="MMIU",
        type=str,
        choices=[
            "MMIU",
            "MUIR",
            "ReMI",
            "ScienceQA",
            "MathVista",
            "MMMU",
            "MMBench",
            "MMVet",
        ],
    )
    parser.add_argument(
        "--domain",
        default="All",
        type=str,
        choices=[
            "All",
            "2D-spatial",
            "3D-spatial",
            "Continuous-temporal",
            "Discrete-temporal",
            "High-level-obj-semantic",
            "High-level-sub-semantic",
            "Low-level-semantic",
            "relation_reasoning",
            "logic_reasoning",
            "attribute_reasoning",
            "coarse_perception",
            "finegrained_perception_cross_instance",
            "finegrained_perception_instance_level",
        ],
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "gpt4o",
            "gemini",
            "llava_next_interleave",
            "llava_one_vision",
            "mantis",
            "idefics2",
            "xcomposer2",
        ],
        default=["gpt4o"],
        nargs="+",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=1024,
        type=int,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--prompting_method",
        default="default",
        type=str,
        choices=[
            "default",
            "caption",
            "qg_caption",
            "ccot",
            "ddcot",
            "qgcot",
            "cocot",
        ],
        help="What kind of prompting method. (Default is Zero-shot)",
    )
    parser.add_argument("--task", type=str, help="What kind of task.", nargs="+")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--api_key", default=os.getenv("GEMINI_API_KEY"), type=str, help="API KEYS."
    )
    parser.add_argument("--device", default=0, type=int, help="Device index.")
    return parser.parse_args()


def eval_questions(args, queries, engine, task):
    results = []
    max_new_tokens = args.max_new_tokens

    pipe = load_model_pipeline(engine, device=args.device)

    for query in queries:

        if engine in ["gpt4o", "gemini"]:
            predicted_answer = closedsource_inference(
                args,
                engine,
                args.dataset,
                args.caption_dir,
                query,
                task,
                args.prompting_method,
                max_new_tokens,
            )
        else:
            predicted_answer = opensource_inference(
                args,
                engine,
                pipe,
                args.dataset,
                args.caption_dir,
                query,
                task,
                args.prompting_method,
                max_new_tokens,
            )
        print(predicted_answer)
        query["prediction"] = predicted_answer
        results.append(query)

    return results


if __name__ == "__main__":
    load_dotenv(verbose=True, override=True)
    args = parse_args()

    for engine, task in product(args.engine, args.task):
        print("Loaded model: {}\n".format(engine))

        query_meta, support_meta = utils.load_data(args, task)
        queries = query_meta + support_meta

        utils.set_random_seed(args.seed)

        results_dict = eval_questions(
            args,
            queries,
            engine,
            task,
        )

        dir_name = os.path.join(
            args.output_dir,
            engine,
            args.dataset,
            args.domain,
            task,
        )

        os.makedirs(dir_name, exist_ok=True)
        with open(f"{dir_name}/{args.prompting_method}.json", "w") as f:
            json.dump(results_dict, f, indent=4)

        torch.cuda.empty_cache()
        gc.collect()
