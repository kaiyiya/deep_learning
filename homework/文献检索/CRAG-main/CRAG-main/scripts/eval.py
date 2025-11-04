#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 说明：
# 本脚本用于对模型生成的预测结果进行离线评测。
# - 输入：原始评测数据（json/jsonl）与对应的预测结果文件（逐行一个预测）。
# - 处理：根据任务类型构造指令/输入格式，并按所选指标（match/accuracy）计算得分。
# - 输出：在控制台打印最终平均得分（Final result: ...）。

import jsonlines
import numpy as np
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy


def preprocess_input_data(dataset, task=None):
    """
    功能：将输入数据（原始样本）根据任务类型补充/改写成统一的 instruction 字段，
    便于后续与预测结果逐条对齐评测。
    - 对 arc_c（Arc Challenge 简写）题型：拼接四/五选项到 instruction；同步生成标准答案 labels。
    - 其他任务：若能在 TASK_INST 查到任务说明，则把问题拼进指令模板。
    返回：带 instruction（和必要时的 answers）的新数据列表。
    """
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            # 处理多选项题：把数值/字母标签统一映射到 A/B/C/D(/E) 文本
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            # 将题干 + 选项拼到 instruction，供评测时参照
            item["instruction"] = instruction + \
                                  "\n\n### Input:\n" + item["question"] + choices
            # 记录标准答案（如 'A'/'B' 等），用于匹配指标
            item["answers"] = [item["answerKey"]]
        else:
            # 非选择题：若有任务说明，则拼接到问题前；否则直接用问题作为 instruction
            prompt = instruction + "\n\n## Input:\n\n" + \
                     item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    # 预测结果文件路径（每一行一个预测）
    parser.add_argument('--eval_file', type=str, default=None)
    # 原始评测输入（json/jsonl），包含问题、选项、答案等信息
    parser.add_argument('--input_file', type=str)
    # 任务名称（如 fever/arc_c 等），用于确定指令模板与后处理逻辑
    parser.add_argument('--task', type=str)
    # Decoding hyperparams
    # 评测指标：目前支持 accuracy（精确匹配）与 match（子串匹配）
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()

    input_path = args.input_file
    # 支持 .json 或 .jsonl 两种输入格式
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    # 统一构造成带 instruction 的数据，方便与预测对齐
    input_data = preprocess_input_data(
        input_data, task=args.task)
    eval_file = args.eval_file
    # 逐行读取模型预测文本
    with open(eval_file, 'r') as f:
        resps = [l.strip()[:] for l in f.readlines()]
    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    # 遍历（预测，输入样本）对，逐条计算指标
    for i, (pred, row) in tqdm(enumerate(zip(resps[:], input_data[:]))):
        pred = pred.strip()
        prompts.append(None)
        preds.append(pred)
        all_results.append(None)
        # 若样本以 answer 字段给出标准答案，统一转到 answers 列表字段
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            # accuracy：严格相等（用于分类/封闭式）
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            # match：子串包含匹配（如 FEVER 任务，把 SUPPORTS/REFUTES 映射为 true/false 再匹配）
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if count % 10 == 0:
            # 中间态结果（未落盘，只用于调试/断点查看）
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results,
                             "all_results": all_results,
                             "golds": golds, "metric": args.metric, "metric_mean": np.mean(metric_results),
                             "scores": scores}
        count += 1

    # 最终统计并打印平均得分
    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds, "metric": args.metric, "metric_mean": np.mean(metric_results), "scores": scores}

    print("Final result: {0}".format(np.mean(metric_results)))


if __name__ == "__main__":
    main()
