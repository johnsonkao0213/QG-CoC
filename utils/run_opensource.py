from utils.external_instruction import EXTERNAL_INSTRUCTION
from transformers.image_utils import load_image

import json


def parse_query(query, dataset):
    question = query["question"]
    images = [load_image(image_path) for image_path in query["image"]]
    content = []

    if dataset in ["ReMI", "MMMU"]:
        for index, image in enumerate(images):
            if dataset == "ReMI":
                question_parts = question.split(f"<image{index + 1}>")
            else:
                question_parts = question.split(f"<image {index+1}>")
            (
                content.append({"type": "text", "text": question_parts[0]})
                if question_parts[0]
                else None
            )
            content.append({"type": "image", "image": image})
            if len(question_parts) == 2:
                question = question_parts[1] if len(question_parts) > 1 else None
        content.append({"type": "text", "text": question}) if question else None
    elif dataset == "MUIR":
        for index, image in enumerate(images):
            question_parts = question.split("<image>", 1)
            (
                content.append({"type": "text", "text": question_parts[0]})
                if question_parts[0]
                else None
            )
            content.append({"type": "image", "image": image})
            question = question_parts[1] if len(question_parts) > 1 else None
        content.append({"type": "text", "text": question}) if question else None
    elif dataset in ["MMIU", "ScienceQA", "MathVista", "MMBench", "MMVet"]:
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question})
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet")

    if dataset in ["MMIU", "MMVet"]:
        content.append(
            {
                "type": "text",
                "text": "\nHint: Please provide the correct option letter, such as A, B, C, D, directly."
                + " The answer is",
            }
        )
    else:
        content.append({"type": "text", "text": "The answer is"})

    return content


def opensource_inference(
    args,
    engine,
    pipeline,
    dataset,
    caption_dir,
    query,
    task,
    prompting_method,
    max_new_tokens,
):
    content = []

    if prompting_method != "default":
        with open(
            f"{caption_dir}/{dataset}-{prompting_method}/gemini/{args.domain}/{task}.json",
            "r",
        ) as file:
            query_descriptions = json.load(file)

        content.append(
            {
                "type": "text",
                "text": EXTERNAL_INSTRUCTION[prompting_method].format(
                    preliminary_knowledge=query_descriptions[query["id"]]["caption"]
                ),
            }
        )

    content += parse_query(query, dataset)
    message = [{"role": "user", "content": content}]

    if engine == "llava_one_vision":
        outputs = pipeline(
            message,
            max_new_tokens,
        )
        return outputs[0]
    else:
        outputs = pipeline(
            text=message,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
        )
        return outputs[0]["generated_text"]
