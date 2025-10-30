import torch
import re
import io
import os
import ast
import math
import random
import numpy as np
import json
import base64
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# from vertexai.preview.generative_models import Part
# import google.generativeai as genai

def set_random_seed(seed_number):
    # position of setting seeds also matters
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def determine_option_length(option_string):
    # Split the string into lines
    lines = option_string.splitlines()

    # Initialize a set to store unique option identifiers
    options = set()

    # Iterate over each line
    for line in lines:
        # Check if the line starts with an option identifier (e.g., 'A:', 'B:', etc.)
        if line and line[0].isalpha() and line[1] == ":":
            # Add the option identifier to the set
            options.add(line[0])

    # The length of the set is the number of unique options
    return len(options)


def filter_dataset_mmiu(dataset, task_name):
    return [item for item in dataset if item["task"] == task_name]


def extract_option_block(option_string, ground_truth):
    # Define a regular expression pattern to match each option and its content
    pattern = r"([A-Z]):\s*(.*?)(?=(?:\n[A-Z]:|$))"

    # Find all matches in the option string
    matches = re.findall(pattern, option_string, re.DOTALL)
    # Create a dictionary to store the content for each option
    option_contents = {}

    for match in matches:
        option, content = match
        # Store the content in the dictionary, stripping any leading/trailing whitespace
        option_contents[option] = content.strip()

    return option_contents[ground_truth]


def transform_dataset_mmiu(dataset, data_dir):
    transformed = []
    tasks_exist = [
        "person_reid",
        "multiple_image_captioning",
        "spot_the_similarity",
        "face_retrieval",
        "sketch2image_retrieval",
        "handwritten_retrieval",
        "spot_the_diff",
        "image2image_retrieval",
        "vehicle_retrieval",
        "text2image_retrieval",
        "general_action_recognition",
        "video_captioning",
        "next_img_prediction",
        "temporal_ordering",
        "meme_vedio_understanding",
        "action_quality_assessment",
        "temporal_localization",
        "mevis",
        "ravens_progressive_matrices",
        "threed_indoor_recognition",
        "point_tracking",
        "threed_cad_recognition",
        "single_object_tracking",
    ]
    for item in dataset:
        if item["task"] in tasks_exist:
            question = (
                "Question: "
                + item["input"]["question"]
                + "\n"
                + item["input"]["context"]
            )
        else:
            question = (
                "Question: "
                + item["input"]["context"]
                + "\n"
                + item["input"]["question"]
            )

        try:
            gt = extract_option_block(item["options"], item["output"]["output_text"])
        except Exception as e:
            # Handle the exception or simply pass to continue with the next iteration
            print(f"An error occurred: {e}")
            continue
        transformed_item = {
            "id": item["task"]
            + "_"
            + item["input"]["input_image_path"][0].split("/")[-1].split("_")[-2],
            "image": [
                path.replace(
                    "/mnt/hwfile/gveval/mengfanqing/MMIU/",
                    f"{data_dir}/MMIU-Benchmark/",
                )
                for path in item["input"]["input_image_path"]
            ],
            "question": question,
            "answer": item["output"]["output_text"],
            "ground_truth": gt,
            "options_num": determine_option_length(item["options"]),
        }
        transformed.append(transformed_item)
    return transformed


## load data
def load_data(args, task):
    if args.dataset == "MMIU":
        with open(f"{args.data_dir}/MMIU-Benchmark/{args.domain}.json", "r") as file:
            dataset = json.load(file)
        filtered_dataset = filter_dataset_mmiu(dataset, task)
        train_set, test_set = train_test_split(
            filtered_dataset, test_size=0.2, random_state=args.seed
        )
        support_meta = transform_dataset_mmiu(train_set, args.data_dir)
        if (
            task == "visually_grounded_reasoning_nlvr2"
            and args.domain == "High-level-obj-semantic"
        ):
            support_meta = [
                support
                for support in support_meta
                if support["id"] != "visually_grounded_reasoning_nlvr2_126"
            ]
        query_meta = transform_dataset_mmiu(test_set, args.data_dir)

    elif args.dataset == "MUIR":
        if not os.path.exists(f"{args.data_dir}/MUIR/{task}"):
            dataset = load_dataset("MUIRBENCH/MUIRBENCH", split="test")
            task_name_dict = {
                "Geographic_Understanding": "Geographic Understanding",
                "Diagram_Understanding": "Diagram Understanding",
                "Image_Text_Matching": "Image-Text Matching",
                "Difference_Spotting": "Difference Spotting",
                "Visual_Retrieval": "Visual Retrieval",
                "Counting": "Counting",
                "Attribute_Similarity": "Attribute Similarity",
                "Scene_Understanding": "Scene Understanding",
                "Action_Understanding": "Action Understanding",
                "Visual_Grounding": "Visual Grounding",
                "Cartoon_Understanding": "Cartoon Understanding",
                "Ordering": "Ordering",
            }

            def create_prompt(sample, use_hint=True):
                question = sample["question"]
                choices = sample["options"]

                # Question
                question_text = f"Question: {question}"

                # Choices
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)

                # Hint
                if use_hint:
                    hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
                else:
                    hint_text = ""

                # Full Prompt
                elements = [question_text, choices_text, hint_text]
                query = "\n".join([e for e in elements if e != ""])
                query = query.strip()

                return query

            data = [
                item for item in dataset if item["task"] == task_name_dict[task]
            ]

            print("Storing MUIR dataset...")

            os.makedirs(f"{args.data_dir}/MUIR/{task}", exist_ok=True)
            transformed = []
            for idx, item in enumerate(data):
                images = []
                for i, img in enumerate(item["image_list"]):
                    img_bytes = pil_image_to_bytes(img)
                    if img_bytes:
                        # images.append(Image.open(io.BytesIO(img_bytes)))
                        img_path = f"{args.data_dir}/MUIR/{task}/idx_{idx}_{i+1}.png"
                        # img = Part.from_data(img_bytes, mime_type="image/png")
                        img = Image.open(io.BytesIO(img_bytes))
                        # Large file upload
                        # img = genai.upload_file(img_path)
                        img.save(img_path)
                        images.append(img_path)

                transformed_item = {
                    "id": item["task"] + "_" + item["idx"],
                    "image": images,
                    "question": create_prompt(item),
                    "answer": item["answer"],
                }
                transformed.append(transformed_item)

            with open(
                f"{args.data_dir}/MUIR/{task}/{task}.json", "w"
            ) as file:
                json.dump(transformed, file, indent=4)

        with open(
            f"{args.data_dir}/MUIR/{task}/{task}.json", "r"
        ) as file:
            dataset = json.load(file)
        print("Data Stored!!")
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "ReMI":
        if not os.path.exists(f"{args.data_dir}/ReMI/{task}"):
            dataset = load_dataset("mehrankazemi/ReMI", split="test")
            dataset = [x for x in dataset]

            data = [item for item in dataset if item["task"] == task]

            os.makedirs(f"{args.data_dir}/ReMI/{task}", exist_ok=True)
            transformed = []
            for idx, item in enumerate(data):
                images = []
                for i in range(6):
                    img_bytes = pil_image_to_bytes(item[f"image_{i+1}"])
                    if img_bytes:
                        # images.append(Image.open(io.BytesIO(img_bytes)))
                        img_path = (
                            f"{args.data_dir}/ReMI/{task}/idx_{idx}_{i+1}.png"
                        )
                        # img = Part.from_data(img_bytes, mime_type="image/png")
                        img = Image.open(io.BytesIO(img_bytes))
                        img.save(img_path)
                        images.append(img_path)

                # question = "Question: " + item["question"] + QUESTION_APPENDICES[task]
                question = "Question: " + item["question"]

                transformed_item = {
                    "id": item["task"] + "_" + str(idx),
                    "image": images,
                    "question": question,
                    "answer": item["label"],
                }
                transformed.append(transformed_item)

            with open(
                f"{args.data_dir}/ReMI/{task}/{task}.json", "w"
            ) as file:
                json.dump(transformed, file, indent=4)

        with open(f"{args.data_dir}/ReMI/{task}/{task}.json", "r") as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "ScienceQA":
        if not os.path.exists(f"{args.data_dir}/ScienceQA"):
            dataset = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
            options = ["A", "B", "C", "D", "E"]
            dataset = [x for x in dataset]

            def create_prompt(sample, use_hint=True):
                question = sample["question"]
                choices = sample["choices"]
                context = sample["hint"]

                # Question
                if context != "":
                    question_text = f"Question: {question}\n{context}"
                else:
                    question_text = f"Question: {question}"

                # Choices
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)

                # Hint
                if use_hint:
                    hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
                else:
                    hint_text = ""

                # Full Prompt
                elements = [question_text, choices_text, hint_text]
                query = "\n".join([e for e in elements if e != ""])
                query = query.strip()

                return query

            os.makedirs(f"{args.data_dir}/ScienceQA/", exist_ok=True)
            transformed = []
            for idx, item in enumerate(dataset):
                img_bytes = pil_image_to_bytes(item["image"])
                if img_bytes:
                    img_path = f"{args.data_dir}/ScienceQA/idx_{idx}.png"
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)

                transformed_item = {
                    "id": str(idx),
                    "image": [img_path],
                    "question": create_prompt(item),
                    "answer": options[item["answer"]],
                }
                transformed.append(transformed_item)

            with open(f"{args.data_dir}/ScienceQA/ScienceQA.json", "w") as file:
                json.dump(transformed, file, indent=4)

        with open(f"{args.data_dir}/ScienceQA/ScienceQA.json", "r") as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "MMMU":
        if not os.path.exists(f"{args.data_dir}/MMMU/{task}"):
            dataset = load_dataset("MMMU/MMMU", task, split="dev")

            def convert_to_sequence(example):
                example["options"] = ast.literal_eval(
                    example["options"]
                )  # Safely parse the string to a list
                return example

            dataset = dataset.map(convert_to_sequence)
            data = [
                item for item in dataset if item["question_type"] == "multiple-choice"
            ]

            def create_prompt(sample, use_hint=True):
                question = sample["question"]
                choices = sample["options"]

                # Question
                question_text = f"Question: {question}"

                # Choices
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)

                # Hint
                if use_hint:
                    hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
                else:
                    hint_text = ""

                # Full Prompt
                elements = [question_text, choices_text, hint_text]
                query = "\n".join([e for e in elements if e != ""])
                query = query.strip()

                return query

            os.makedirs(f"{args.data_dir}/MMMU/{task}", exist_ok=True)
            transformed = []
            for idx, item in enumerate(data):
                id = item["id"]
                images = []
                for i in range(7):
                    img_bytes = pil_image_to_bytes(item[f"image_{i+1}"])
                    if img_bytes:
                        img_path = f"{args.data_dir}/MMMU/{task}/{id}_{i+1}.png"
                        img = Image.open(io.BytesIO(img_bytes))
                        img.save(img_path)
                        images.append(img_path)
                transformed_item = {
                    "id": id,
                    "image": images,
                    "question": create_prompt(item),
                    "answer": item["answer"],
                }
                transformed.append(transformed_item)

            with open(
                f"{args.data_dir}/MMMU/{task}/{task}.json", "w"
            ) as file:
                json.dump(transformed, file, indent=4)

        with open(f"{args.data_dir}/MMMU/{task}/{task}.json", "r") as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "MMVet":
        if not os.path.exists(f"{args.data_dir}/MMVet"):
            dataset = load_dataset("lmms-lab/MMVet", split="test")
            dataset = [x for x in dataset]

            os.makedirs(f"{args.data_dir}/MMVet/", exist_ok=True)
            transformed = []
            for idx, item in enumerate(dataset):
                id = item["question_id"]
                img_bytes = pil_image_to_bytes(item["image"])
                if img_bytes:
                    img_path = f"{args.data_dir}/MMVet/{id}.png"
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)

                transformed_item = {
                    "id": id,
                    "image": [img_path],
                    "question": "Question: " + item["question"],
                    "answer": item["answer"],
                }
                transformed.append(transformed_item)

            with open(f"{args.data_dir}/MMVet/MMVet.json", "w") as file:
                json.dump(transformed, file, indent=4)

        with open(f"{args.data_dir}/MMVet/MMVet.json", "r") as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "MMBench":
        if not os.path.exists(f"{args.data_dir}/MMBench/{args.domain}/{task}"):
            dataset = load_dataset("lmms-lab/MMBench_EN", split="dev")
            options = ["A", "B", "C", "D"]
            domain_name_dict = {
                "relation_reasoning": "relation_reasoning",
                "logic_reasoning": "logic_reasoning",
                "attribute_reasoning": "attribute_reasoning",
                "coarse_perception": "coarse_perception",
                "finegrained_perception_cross_instance": "finegrained_perception (cross-instance)",
                "finegrained_perception_instance_level": "finegrained_perception (instance-level)",
            }
            dataset = [x for x in dataset]
            data = [
                item
                for item in dataset
                if item["category"] == task
                and item["l2-category"] == domain_name_dict[args.domain]
            ]

            def is_none(value):
                if value is None:
                    return True
                if type(value) is float and math.isnan(value):
                    return True
                if type(value) is str and value.lower() == "nan":
                    return True
                if type(value) is str and value.lower() == "none":
                    return True
                return False

            def get_options(row, options):
                parsed_options = []
                for option in options:
                    option_value = row[option]
                    if is_none(option_value):
                        break
                    parsed_options.append(option_value)
                return parsed_options

            def create_prompt(sample, use_hint=True):
                question = sample["question"]
                choices = get_options(sample, options)

                # Question
                question_text = f"Question: {question}"

                # Choices
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)

                # Hint
                if use_hint:
                    hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
                else:
                    hint_text = ""

                # Full Prompt
                elements = [question_text, choices_text, hint_text]
                query = "\n".join([e for e in elements if e != ""])
                query = query.strip()

                return query

            os.makedirs(
                f"{args.data_dir}/MMBench/{args.domain}/{task}", exist_ok=True
            )
            transformed = []
            for idx, item in enumerate(data):
                id = item["index"]
                img_bytes = pil_image_to_bytes(item["image"])
                if img_bytes:
                    img_path = (
                        f"{args.data_dir}/MMBench/{args.domain}/{task}/{id}.png"
                    )
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)

                transformed_item = {
                    "id": id,
                    "image": [img_path],
                    "question": create_prompt(item),
                    "answer": item["answer"],
                }
                transformed.append(transformed_item)

            with open(
                f"{args.data_dir}/MMBench/{args.domain}/{task}.json", "w"
            ) as file:
                json.dump(transformed, file, indent=4)

        with open(
            f"{args.data_dir}/MMBench/{args.domain}/{task}.json", "r"
        ) as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    elif args.dataset == "MathVista":
        if not os.path.exists(f"{args.data_dir}/MathVista/{task}"):
            dataset = load_dataset("AI4Math/MathVista", split="testmini")
            options = ["A", "B", "C", "D", "E"]
            task_name_dict = {
                "geometry_problem_solving": "geometry problem solving",
                "math_word_problem": "math word problem",
                "visual_question_answering": "visual question answering",
                "figure_question_answering": "figure question answering",
                "textbook_question_answering": "textbook question answering",
            }
            dataset = [x for x in dataset]
            data = [
                item
                for item in dataset
                if item["metadata"]["task"] == task_name_dict[task]
                and item["question_type"] == "multi_choice"
            ]

            def create_prompt(sample, use_hint=True):
                question = sample["question"]
                choices = sample["choices"]

                # Question
                question_text = f"Question: {question}"

                # Choices
                texts = ["Choices:"]
                for i, choice in enumerate(choices):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                choices_text = "\n".join(texts)

                # Hint
                if use_hint:
                    hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
                else:
                    hint_text = ""

                # Full Prompt
                elements = [question_text, choices_text, hint_text]
                query = "\n".join([e for e in elements if e != ""])
                query = query.strip()

                return query

            os.makedirs(f"{args.data_dir}/MathVista/{task}", exist_ok=True)
            transformed = []
            for idx, item in enumerate(data):
                id = item["pid"]
                img_bytes = pil_image_to_bytes(item["decoded_image"])
                if img_bytes:
                    img_path = f"{args.data_dir}/MathVista/{task}/{id}.png"
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)

                transformed_item = {
                    "id": id,
                    "image": [img_path],
                    "question": create_prompt(item),
                    "answer": chr(ord("A") + item["choices"].index(item["answer"])),
                }
                transformed.append(transformed_item)

            with open(f"{args.data_dir}/MathVista/{task}.json", "w") as file:
                json.dump(transformed, file, indent=4)

        with open(f"{args.data_dir}/MathVista/{task}.json", "r") as file:
            dataset = json.load(file)
        support_meta, query_meta = train_test_split(
            dataset, test_size=0.2, random_state=args.seed
        )
    else:
        data_dir = args.data_dir
        query_file = os.path.join(data_dir, args.dataset, "query.json")
        support_file = os.path.join(data_dir, args.dataset, "support.json")

        with open(query_file, "r") as f:
            query_meta = json.load(f)
        with open(support_file, "r") as f:
            support_meta = json.load(f)

    return query_meta, support_meta


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


# @title Utils


def pil_image_to_bytes(img, format="PNG"):
    if not img:
        return None
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


QUESTION_APPENDICES = {
    "MMIU": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "MUIR": ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "ScienceQA": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "MathVista": ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "MMMU": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "MMBench": ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "MMVet": ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "Collisions": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer. If it is a yes or no question, the answer field must be 0 for no and 1 for yes.',
    "Clocks": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    "Schedule": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string corresponding to your final answer.',
    "EmojiAlgebra": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    "Charts": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string or numerical value corresponding to your final answer.',
    "CodeEdit": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the line of code corresponding to your final answer.',
    "GeomShape": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    "GeomCost": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    "FuncRead": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string or numeric value corresponding to your final answer.',
    "RefCoco": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    "IQ": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    "Isomorphism": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field must be 1 if the two graphs are isomorphic and 0 otherwise.',
    "Maps": 'Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string corresponding to your final answer.',
}
