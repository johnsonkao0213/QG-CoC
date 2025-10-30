import numpy as np
import re
import json
import random


def eval_scores(results, dataset, task, model=None):
    if model in ["gpt4o", "gemini"]:
        score = evaluate(dataset, task, results)
    else:
        score = evaluate_open_source(dataset, task, results)
    return score


def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result["prediction"].strip()
        prediction = prediction.strip("\n")
        trunc_index = prediction.find("\n")
        if trunc_index <= 0:
            trunc_index = prediction.find(".")
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result["answer"].lower() == "yes" and "yes" in str(prediction).lower():
            acc.append(1)
        elif result["answer"].lower() == "no" and "yes" not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


def exact_in_match(results):
    acc = []
    for result in results:
        prediction = result["prediction"].strip()
        prediction = prediction.strip("\n")
        trunc_index = prediction.find("\n")
        if trunc_index <= 0:
            trunc_index = prediction.find(".")
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if str(result["answer"]).lower() in str(prediction[0]).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


# From MMIU
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# @title Evaluate
def strip_json(s):
    """Takes a text possibly containing a json, and extrcts the json part."""
    return s[s.index("{") if "{" in s else 0 : s.rindex("}") + 1 if "}" in s else 0]


def is_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def exact_match(label, pred, eps=0.01):
    return (
        str(label) == str(pred)
        or (is_float(label) and is_float(pred) and float(label) == float(pred))
        or (
            (pred == "f(x)" and label == "f")
            or (pred == "g(x)" and label == "g")
            or (pred == "h(x)" and label == "h")
        )
        or (f"({pred})" == label or f"({label})" == pred)
        or (
            str(pred).replace(" ", "") == str(label)
            or str(label).replace(" ", "") == str(pred)
        )
        or relaxed_accuracy(label, pred, eps=eps)
    )


def relaxed_accuracy(label, pred, eps=0.03):
    if not is_float(label) or not is_float(pred):
        return False
    return (1 - eps) * float(label) <= float(pred) <= (1 + eps) * float(label)


def accuracy_with_tolerance(label, pred, tolerance=10):
    if not is_float(label) or not is_float(pred):
        return False
    return float(label) - tolerance <= float(pred) <= float(label) + tolerance


def get_pred(model_response):
    model_response_json = (
        strip_json(model_response).replace('\\"', "").replace("\\", "")
    )
    try:
        pred = str(json.loads(model_response_json)["answer"]).lower()
        return pred.split("%")[0].strip()
    except (KeyError, json.JSONDecodeError):
        return "BAD_JSON"


def prep_label(label):
    return label.lower().replace("\\", "").split("%")[0].strip()


def evaluate(dataset, task, results):
    correct = 0
    for result in results:
        model_response = result["prediction"]
        orig_label = result["answer"]
        pred, label = get_pred(model_response), prep_label(orig_label)
        if dataset == "ReMI":
            if task == "RefCoco":
                correct += (
                    1 if str(pred) in label.split(",") else 0
                )  # whether pred is in label
            elif task in ["GeomShape", "GeomCost"]:
                correct += 1 if relaxed_accuracy(label, pred, eps=0.03) else 0
            elif task == "Clocks":
                correct += (
                    1 if accuracy_with_tolerance(label, pred, tolerance=10) else 0
                )
            else:
                correct += 1 if exact_match(label, pred) else 0
        else:  #  MMIU & MUIR (multi-choice QA)
            correct += 1 if exact_match(label, pred) else 0
    return correct / len(results)


def evaluate_open_source(dataset, task, results):
    correct = 0
    for result in results:
        orig_label = result["answer"]
        model_response = result["prediction"].lower()
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            model_response = model_response.strip(char)
        pred, label = model_response.split(":")[0].strip(), prep_label(orig_label)
        if len(pred) > 1:
            pred = pred.split(",")[0].strip()
        if task == "RefCoco":
            correct += (
                1 if str(pred) in label.split(",") else 0
            )  # whether pred is in label
        elif task in ["GeomShape", "GeomCost"]:
            correct += 1 if relaxed_accuracy(label, pred, eps=0.03) else 0
        elif task == "Clocks":
            correct += 1 if accuracy_with_tolerance(label, pred, tolerance=10) else 0
        else:
            correct += 1 if exact_match(label, pred) else 0
    return correct / len(results)
