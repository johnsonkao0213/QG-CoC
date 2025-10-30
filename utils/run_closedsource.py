import os
import json
from .utils import encode_image
from .brain import decoder_for_response
from .prompt import *
from .external_instruction import EXTERNAL_INSTRUCTION
from .question_appendices import QUESTION_APPENDICES
from .description_inference import *


def get_task_instruction(args):
    dataset = args.dataset
    task = args.task
    if dataset == "ReMI":
        instr = QUESTION_APPENDICES[task]
    else:
        instr = QUESTION_APPENDICES[dataset]
    return instr


def closedsource_inference(
    args,
    engine,
    dataset,
    caption_dir,
    query,
    task,
    prompting_method,
    max_new_tokens,
):
    task_instruction = get_task_instruction(args)
    if "gpt4o" in engine:
        query_image_paths = query["image"]
        query_text = query["question"]

        content = [
            f"{task_instruction}\n",
        ]
        if prompting_method != "default":
            with open(
                f"{caption_dir}/{dataset}-{prompting_method}/{engine}/{args.domain}/{task}.json",
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

        if dataset in ["ReMI", "MMMU"]:
            parts = []
            question = query_text
            for j, img_f in enumerate(query_image_paths):
                base64_image, mime_type = encode_image(img_f)
                img = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "low",
                    },
                }
                if dataset == "ReMI":
                    question_parts = question.split(f"<image{j+1}>")
                else:
                    question_parts = question.split(f"<image {j+1}>")
                if question_parts[0]:
                    parts.append(
                        {
                            "type": "text",
                            "text": question_parts[0],
                        }
                    )
                    parts.append(img)
                else:
                    parts.append(img)
                if len(question_parts) == 2:
                    question = question_parts[1]
            if question:
                content += parts
                content.append(
                    {
                        "type": "text",
                        "text": question,
                    }
                )
            else:
                content += parts
        elif dataset == "MUIR":
            parts = []
            question = query_text
            for j, img_f in enumerate(query_image_paths):
                base64_image, mime_type = encode_image(img_f)
                img = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "low",
                    },
                }
                question_parts = question.split(f"<image>", 1)
                if question_parts[0]:
                    parts.append(
                        {
                            "type": "text",
                            "text": question_parts[0],
                        }
                    )
                    parts.append(img)
                else:
                    parts.append(img)
                question = question_parts[1]
            if question:
                content += parts
                content.append(
                    {
                        "type": "text",
                        "text": question,
                    }
                )
            else:
                content += parts
        else:
            for query_image_path in query_image_paths:
                base64_image, mime_type = encode_image(query_image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": query_text,
                    }
                )

        if dataset in ["MMIU", "MMVet"]:
            content.append(
                {
                    "type": "text",
                    "text": "\nHint: Please provide the correct option letter, such as A, B, C, D, directly. The answer is",
                }
            )
        else:
            content.append({"type": "text", "text": "The answer is"})

        messages = [{"role": "user", "content": content}]
        predicted_answers = decoder_for_response(
            args, engine, messages, max_new_tokens, file_list=None
        )
        print(query["id"], "\t", predicted_answers)

    elif "gemini" in engine:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        query_image_paths = query["image"]
        query_text = query["question"]

        content = [f"{task_instruction}\n"]
        file_list = []

        if prompting_method != "default":
            with open(
                f"{caption_dir}/{dataset}-{prompting_method}/{engine}/{args.domain}/{task}.json",
                "r",
            ) as file:
                query_descriptions = json.load(file)
            content.append(
                EXTERNAL_INSTRUCTION[prompting_method].format(
                    preliminary_knowledge=query_descriptions[f"{query['id']}"][
                        "caption"
                    ]
                )
            )

        if dataset in ["ReMI", "MMMU"]:
            parts = []
            question = query_text
            for j, img_f in enumerate(query_image_paths):
                # Large file needs use this API
                img = genai.upload_file(img_f)
                file_list.append(img)
                if dataset == "ReMI":
                    question_parts = question.split(f"<image{j+1}>")
                else:
                    question_parts = question.split(f"<image {j+1}>")
                parts += [question_parts[0], img] if question_parts[0] else [img]
                if len(question_parts) == 2:
                    question = question_parts[1]
            content += parts + [question] if question else parts
        elif dataset == "MUIR":
            parts = []
            question = query_text
            for j, img_f in enumerate(query_image_paths):
                # Large file needs use this API
                img = genai.upload_file(img_f)
                file_list.append(img)
                question_parts = question.split(f"<image>", 1)
                parts += [question_parts[0], img] if question_parts[0] else [img]
                question = question_parts[1]
            content += parts + [question] if question else parts
        else:
            for query_image_path in query_image_paths:
                # Large file needs use this API
                img = genai.upload_file(query_image_path)
                file_list.append(img)
                content.append(img)
            content.append(query_text)

        if dataset in ["MMIU", "MMVet"]:
            content.append(
                "\nHint: Please provide the correct option letter, such as A, B, C, D, directly. The answer is"
            )
        else:
            content.append("The answer is")

        predicted_answers = decoder_for_response(
            args, engine, content, max_new_tokens, file_list=file_list
        )
        print(query["id"], "\t", predicted_answers)

    return predicted_answers
