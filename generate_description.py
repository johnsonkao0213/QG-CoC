import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
from utils import utils, description_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Description Generator")

    parser.add_argument(
        "--output_dir", default="./caption", type=str, help="Output directory"
    )
    parser.add_argument(
        "--data_dir", default="./data_cache", type=str, help="Data cache directory"
    )
    parser.add_argument(
        "--dataset",
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
            "openflamingo",
            "otter-llama",
            "llava16-7b",
            "qwen-vl",
            "qwen-vl-chat",
            "internlm-x2",
            "emu2-chat",
            "idefics-9b-instruct",
            "idefics-80b-instruct",
            "llava_next_interleave",
            "llava_one_vision",
            "mantis",
            "idefics2",
            "xcomposer2",
        ],
        default="gpt4o",
        type=str,
    )
    parser.add_argument(
        "--task", default=["All"], type=str, help="What kind of task.", nargs="+"
    )
    parser.add_argument(
        "--method",
        choices=[
            "caption",
            "qg_caption",
            "ccot",
            "ddcot",
            "cocot",
            "qgcot",
        ],
        default="caption",
        type=str,
        help="What kind of caption.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    return parser.parse_args()


def write_to_json(result, file_path):
    with open(file_path, "w") as file:
        json.dump(result, file, indent=4)


def main():
    load_dotenv(verbose=True, override=True)
    args = parse_args()
    utils.set_random_seed(args.seed)
    for task in args.task:
        query_meta, support_meta = utils.load_data(args, task)
        descriptions_support = {}
        os.makedirs(
            f"{args.output_dir}/{args.dataset}-{args.method}/{args.engine}/{args.domain}/",
            exist_ok=True,
        )

        for data in tqdm(support_meta + query_meta):
            response = description_inference(args, args.engine, args.dataset, data)
            descriptions_support[data["id"]] = {
                "caption": response,
                "image": data["image"],
                "question": data["question"],
            }

            write_to_json(
                descriptions_support,
                f"{args.output_dir}/{args.dataset}-{args.method}/{args.engine}/{args.domain}/{task}.json",
            )


if __name__ == "__main__":
    main()
