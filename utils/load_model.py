from transformers import pipeline, AutoProcessor, AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass


@dataclass
class Engine:
    llava_1_5: str = "llava-hf/llava-1.5-7b-hf"
    llava_next_interleave: str = "llava-hf/llava-interleave-qwen-7b-hf"
    llava_one_vision: str = (
        "/nfs/data/hgzhou42/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd/"
    )
    mantis: str = (
        "/nfs/data/ethanhsu/huggingface/hub/models--TIGER-Lab--Mantis-8B-Idefics2/snapshots/a1ae928477b92a51b14d619b0932740cb122115b/"
    )
    idefics2: str = (
        "/nfs/data/ethanhsu/huggingface/hub/models--HuggingFaceM4--idefics2-8b/snapshots/2c42686c57fe21cf0348c9ce1077d094b72e7698/"
    )
    xgen_mm: str = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
    xcomposer2: str = "internlm/internlm-xcomposer2-7b"


def load_model_pipeline(engine, device):
    if engine in ["gpt4o", "gemini"]:
        return None
    elif engine == "llava_next_interleave":
        model_name = Engine().llava_next_interleave
        pipe = pipeline("image-text-to-text", model=model_name, device=device)
        return pipe
    elif engine == "llava_one_vision":
        model_name = Engine().llava_one_vision
        import copy
        from llava.model.builder import load_pretrained_model

        llava_model_args = {
            "multimodal": True,
        }
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_name,
            None,
            model_name="llava_qwen",
            device_map="auto",
            **llava_model_args,
        )

        model.eval()

        def pipe(inputs, max_new_tokens, **kwargs):
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
            )
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            query = ""
            images = []
            device = "cuda"
            for message in inputs:
                for content in message["content"]:
                    if content["type"] == "text":
                        query += content["text"]
                    elif content["type"] == "image":
                        query += f"{DEFAULT_IMAGE_TOKEN}\n"
                        images.append(content["image"])
            image_tensor = torch.stack(
                [
                    image_processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                    for image_file in images
                ]
            )
            image_tensor = image_tensor.to(dtype=torch.float16, device=device)
            conv_mode = "qwen_1_5"
            conv = copy.deepcopy(conv_templates[conv_mode])
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(device)
            )
            image_sizes = [image.size for image in images]
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
            predicted_answer = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return predicted_answer
        return pipe
    elif engine == "mantis":
        model_name = Engine().mantis
        pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
        )
        return pipe
    elif engine == "idefics2":
        pass
        model_name = Engine().idefics2
        processor = AutoProcessor.from_pretrained(
            model_name,
            image_splitting=False,
            size={"longest_edge": 448, "shortest_edge": 378},
        )
        pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            processor=processor,
        )
        return pipe
    elif engine == "xgen_mm":
        pass
    elif engine == "xcomposer2":
        model = AutoModel.from_pretrained(
            "internlm/internlm-xcomposer2-7b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=f"cuda:{device}",
        )
        tokenizer = AutoTokenizer.from_pretrained(engine, trust_remote_code=True)

        def pipe(inputs, max_new_tokens, **kwargs):
            query = ""
            images = []
            for message in inputs:
                for content in message["content"]:
                    if content["type"] == "text":
                        query += content["text"]
                    elif content["type"] == "image":
                        query += " <ImageHere> "
                        images.append(model.vis_processor(content["image"]))
            image = torch.stack(images).cuda()
            with torch.autocast("cuda"):
                predicted_answer, _ = model.chat(
                    tokenizer,
                    query=query,
                    image=image,
                    history=[],
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
                return predicted_answer

        return pipe

    else:
        raise NotImplementedError(f"Engine {engine} not implemented yet")
