import json
import os
import time

import pandas as pd
import requests
from transformers import AutoTokenizer


def get_prompt_content(prompt):
    if isinstance(prompt, list):
        return prompt[0]["content"]
    if hasattr(prompt, "tolist"):
        value = prompt.tolist()
        if isinstance(value, list):
            return value[0]["content"]
    return prompt[0]["content"]


def extract_question(content):
    if "Question:" not in content:
        raise ValueError(f"cannot find Question: in {content[:200]!r}")
    return content.rsplit("Question:", 1)[1].strip()


def make_prefix(question, context):
    return (
        "Answer the given question with some potentially useful context. "
        "You should analyze the question carefully, evaluate the given context "
        "(which may or may not be useful), and then generate an accurate and "
        "well-reasoned response. You should first have a reasoning process in "
        "mind and then provides the answer. Show your reasoning in <think> "
        "</think> tags and return the final answer in <answer> </answer> tags, "
        "for example <answer> Beijing </answer>. "
        f"Question: {question} Context: {context} \n"
    )


def main():
    out_dir = os.environ["OUT_DIR"]
    src = os.environ["SOURCE_PARQUET"]
    tokenizer_path = os.environ["TOKENIZER_PATH"]
    retriever_url = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
    topk = int(os.environ.get("TOPK", "3"))
    max_context_tokens = int(os.environ.get("MAX_CONTEXT_TOKENS", "500"))
    batch_size = int(os.environ.get("RETRIEVAL_BATCH_SIZE", "64"))

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(src)
    df = df[df["data_source"] == "nq"].reset_index(drop=True)
    print(f"source rows after nq filter: {len(df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def passages_to_string(retrieval_result):
        formatted = ""
        for idx, doc_item in enumerate(retrieval_result):
            doc = doc_item.get("document", doc_item)
            content = doc["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted += f"Doc {idx + 1}(Title: {title}) {text}\n"
        ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        if len(ids) > max_context_tokens:
            formatted = tokenizer.decode(ids[:max_context_tokens], skip_special_tokens=True)
        return formatted

    questions = [extract_question(get_prompt_content(prompt)) for prompt in df["prompt"]]
    contexts = []
    debug_path = os.path.join(out_dir, "retrieval_debug.jsonl")
    start = time.time()

    with open(debug_path, "w", encoding="utf-8") as debug_file:
        for start_idx in range(0, len(questions), batch_size):
            batch = questions[start_idx:start_idx + batch_size]
            response = requests.post(
                retriever_url,
                json={"queries": batch, "topk": topk, "return_scores": True},
                timeout=600,
            )
            response.raise_for_status()
            results = response.json()["result"]
            for question, result in zip(batch, results):
                contexts.append(passages_to_string(result))
                debug_file.write(json.dumps({
                    "question": question,
                    "top_titles": [
                        item["document"]["contents"].split("\n", 1)[0]
                        for item in result
                    ],
                }, ensure_ascii=False) + "\n")
            done = min(start_idx + len(batch), len(questions))
            print(f"retrieved {done}/{len(questions)} elapsed={time.time() - start:.1f}s", flush=True)

    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "data_source": "nq",
            "prompt": [{"role": "user", "content": make_prefix(questions[idx], contexts[idx])}],
            "ability": row.get("ability", "fact-reasoning"),
            "reward_model": row["reward_model"],
            "extra_info": {
                "split": "test",
                "index": int(idx),
                "rag_topk": topk,
                "max_context_tokens": max_context_tokens,
            },
        })

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(out_dir, "test.parquet"))
    out.head(8).to_parquet(os.path.join(out_dir, "train.parquet"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump({
            "source": src,
            "rows": len(out),
            "topk": topk,
            "max_context_tokens": max_context_tokens,
            "retriever": "wiki18_e5_flat_live",
        }, meta_file, ensure_ascii=False, indent=2)

    print(f"wrote {out_dir}/test.parquet rows={len(out)}", flush=True)
    print(f"sample prompt: {out.iloc[0]['prompt'][0]['content'][:800]}", flush=True)


if __name__ == "__main__":
    main()
