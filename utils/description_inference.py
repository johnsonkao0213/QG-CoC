import os
from .utils import encode_image
from .prompt import *
from .brain import decoder_for_response


def description_inference(args, engine, dataset, data, max_new_tokens=2048):
    if args.method == "caption":
        instruction = caption
    elif args.method == "qg_caption":
        instruction = qg_caption
    elif args.method == "ccot":
        instruction = ccot
    elif args.method == "ddcot":
        instruction = ddcot
    elif args.method == "cocot":
        instruction = cocot
    elif args.method == "qgcot":
        instruction = qgcot

    if "gpt4o" in engine:
        content = [
            {
                "type": "text",
                "text": f"{instruction}",
            }
        ]
        if args.method == "caption":
            for image_path in data["image"]:
                base64_image, mime_type = encode_image(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                )
        else:
            if args.method == "ddcot":
                data["image"] = []
            if dataset in ["ReMI", "MMMU"]:
                parts = []
                question = data["question"]
                for j, img_f in enumerate(data["image"]):
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
                question = "".join(question.split("\n")[:-1])
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
                question = data["question"]
                for j, img_f in enumerate(data["image"]):
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
                question = "".join(question.split("\n")[:-1])
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
                for image_path in data["image"]:
                    base64_image, mime_type = encode_image(image_path)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "low",
                            },
                        }
                    )
                if dataset in ["MMIU", "MMVet"]:
                    question = data["question"]
                else:
                    question = "".join(data["question"].split("\n")[:-1])
                content.append(
                    {
                        "type": "text",
                        "text": question,
                    }
                )

        messages = [{"role": "user", "content": content}]

        predicted_answers = decoder_for_response(
            args, engine, messages, max_new_tokens, file_list=None
        )

    elif "gemini" in engine:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        content = [
            f"""{instruction}\n
            """
        ]
        file_list = []

        if args.method in ["caption"]:
            for image_path in data["image"]:
                # Large file needs use this API
                img = genai.upload_file(image_path)
                file_list.append(img)
                content.append(img)
        else:
            if args.method == "ddcot":
                data["image"] = []
            if dataset in ["ReMI", "MMMU"]:
                parts = []
                question = data["question"]
                for j, img_f in enumerate(data["image"]):
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
                question = "".join(question.split("\n")[:-1])
                content += parts + [question] if question else parts
            elif dataset == "MUIR":
                parts = []
                question = data["question"]
                for j, img_f in enumerate(data["image"]):
                    # Large file needs use this API
                    img = genai.upload_file(img_f)
                    file_list.append(img)
                    question_parts = question.split(f"<image>", 1)
                    parts += [question_parts[0], img] if question_parts[0] else [img]
                    question = question_parts[1]
                question = "".join(question.split("\n")[:-1])
                content += parts + [question] if question else parts
            else:
                for image_path in data["image"]:
                    # Large file needs use this API
                    img = genai.upload_file(image_path)
                    file_list.append(img)
                    content.append(img)
                if dataset in ["MMIU", "MMVet"]:
                    question = data["question"]
                else:
                    question = "".join(data["question"].split("\n")[:-1])
                content.append(question)

        predicted_answers = decoder_for_response(
            args, engine, content, max_new_tokens, file_list=file_list
        )

    if args.method == "ddcot":
        predicted_answers = ddcot_inference(
            args, predicted_answers, data, engine, True, max_new_tokens=2048
        )
    elif args.method == "qgcot":
        predicted_answers = extract_qa(predicted_answers)

    print(data["id"], "\t", predicted_answers)

    return predicted_answers


def ddcot_inference(args, answer, data, engine, single, max_new_tokens=2048):
    import re

    sub_question_pattern = re.compile(r"Sub-questions:(.*?)Sub-answers:", re.DOTALL)
    sub_answer_pattern = re.compile(r"Sub-answers:(.*?)Answer:", re.DOTALL)

    sub_questions_match = sub_question_pattern.search(answer)
    sub_answers_match = sub_answer_pattern.search(answer)

    sub_questions = []
    if sub_questions_match:
        sub_questions = [
            re.sub(r"^\d+\.\s*", "", sub.strip())
            for sub in sub_questions_match.group(1).split("\n")
            if sub.strip()
        ]
    sub_answers = []
    if sub_answers_match:
        sub_answers = [
            re.sub(r"^\d+\.\s*", "", sub.strip())
            for sub in sub_answers_match.group(1).split("\n")
            if sub.strip()
        ]
    if sub_questions == [] or sub_answers == []:
        return "no response"

    keywords = [
        "uncertain",
        "Uncertain",
        "insufficient",
        "Insufficient",
        "cannot be determined",
        "not provide",
        "not possible",
    ]

    preliminary_knowledge = ""
    for sub_q, sub_a in zip(sub_questions, sub_answers):
        reform_sub_q = f"Question: {sub_q} Answer: "
        if any(keyword in sub_a for keyword in keywords):
            if "gpt4o" in engine:
                if single:
                    content = [
                        {
                            "type": "text",
                            "text": f"Please answer the following question for each image individually.",
                        }
                    ]
                else:
                    content = [
                        {
                            "type": "text",
                            "text": f"Please answer the following question based on the given images.",
                        }
                    ]

                for image_path in data["image"]:
                    base64_image, mime_type = encode_image(image_path)
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
                        "text": reform_sub_q,
                    }
                )

                messages = [{"role": "user", "content": content}]

                generated_text = decoder_for_response(
                    args, engine, messages, max_new_tokens, file_list=None
                )
            elif "gemini" in engine:
                import google.generativeai as genai

                api_key = os.getenv("GEMINI_API_KEY")
                genai.configure(api_key=api_key)

                if single:
                    content = [
                        f"Please answer the following question for each image individually.\n"
                    ]
                else:
                    content = [
                        f"Please answer the following question based on the given images.\n"
                    ]
                file_list = []

                for image_path in data["image"]:
                    # Large file needs use this API
                    img = genai.upload_file(image_path)
                    file_list.append(img)
                    content.append(img)
                content.append(reform_sub_q)

                generated_text = decoder_for_response(
                    args, engine, content, max_new_tokens, file_list=file_list
                )

            preliminary_knowledge += reform_sub_q + generated_text + "\n"
        else:
            preliminary_knowledge += reform_sub_q + sub_a + "\n"

    return preliminary_knowledge


def extract_sub_items(text):
    import re

    items = []
    current_item = []
    for line in text.split("\n"):
        stripped_line = line.strip()
        if stripped_line:
            new_stripped_line = re.sub(r"^\d+\.\s*", "", stripped_line)
            if re.match(r"^\d+\.\s*", stripped_line):
                if current_item:
                    items.append("\n".join(current_item).strip())
                    current_item = []
            current_item.append(new_stripped_line)  # Preserve exact line formatting
    if current_item:
        items.append("\n".join(current_item).strip())
    return items


def extract_qa(answer):
    import re

    sub_question_pattern = re.compile(r"Sub-questions:(.*?)Sub-answers:", re.DOTALL)
    # sub_answer_pattern = re.compile(r"Sub-captions:(.*?)", re.DOTALL)
    if "Caption" in answer:
        sub_answer_pattern = re.compile(r"Sub-answers:(.*)\n\n", re.DOTALL)
    else:
        sub_answer_pattern = re.compile(r"Sub-answers:(.*)", re.DOTALL)

    sub_questions_match = sub_question_pattern.search(answer)
    sub_answers_match = sub_answer_pattern.search(answer)

    sub_questions = []
    if sub_questions_match:
        sub_questions_text = sub_questions_match.group(1)
        sub_questions = extract_sub_items(sub_questions_text)
    sub_answers = []
    if sub_answers_match:
        sub_answers_text = sub_answers_match.group(1)
        sub_answers = extract_sub_items(sub_answers_text)

    preliminary_knowledge = ""
    for sub_q, sub_a in zip(sub_questions, sub_answers):
        reform_sub_q = f"Question: {sub_q} Answer: "
        preliminary_knowledge += reform_sub_q + sub_a + "\n"

    return preliminary_knowledge
