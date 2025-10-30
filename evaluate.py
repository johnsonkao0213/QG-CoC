import os
import json
import argparse
from utils import eval


metrics = {
    "MMIU": "Acc",
    "MUIR": "Acc",
    "ReMI": "Acc",
    "MMMU": "Acc",
    "ScienceQA": "Acc",
    "MMBench": "Acc",
}


def parse_args():
    parser = argparse.ArgumentParser(description="I2T ICL Evaluation")

    parser.add_argument(
        "--result_dir", default="./results", type=str
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
        required=True,
    )
    parser.add_argument(
        "--domain",
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
        required=True,
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "llava_next_interleave",
            "llava_one_vision",
            "mantis",
            "idefics2",
            "xcomposer2",
            "gpt4o",
            "gemini",
        ],
        default="gpt4o",
        type=str,
    )
    parser.add_argument(
        "--prompting_method",
        default=[
            "default",
            "caption",
            "qg_caption",
            "ccot",
            "ddcot",
            "qgcot",
            "cocot",
        ],
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
        nargs="+",
    )
    parser.add_argument(
        "--task", type=str, help="What kind of task."
    )
    parser.add_argument(
        "--max-new-tokens", default=15, type=int, help="Max new tokens for generation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir

    if args.dataset == "MMMU" or args.dataset == "MMBench":
        dir_name = os.path.join(
            result_dir,
            args.engine,
            args.dataset,
            args.domain,
        )
    else:
        dir_name = os.path.join(
            result_dir,
            args.engine,
            args.dataset,
            args.domain,
            args.task,
        )

    result_files = [
        f"{dir_name}/{method}.json"
        for method in args.prompting_method
    ]

    for result_file in result_files:
        method = result_file.split("/")[-1].replace(".json", "")
        with open(result_file, "r") as f:
            results_dict = json.load(f)

        score = eval.eval_scores(results_dict, args.dataset, args.task, model=args.engine)
        print(
            f"{args.dataset} {args.domain} {args.task} {method} {metrics[args.dataset]} of {args.engine}: ",
            f"{score * 100.0:.2f}",
            flush=True,
        )
