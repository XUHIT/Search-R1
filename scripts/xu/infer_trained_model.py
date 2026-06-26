import os
import re
import time

import requests
import torch
import transformers


DEFAULT_QUESTION = (
    "Mike Barnett negotiated many contracts including which player that went on "
    "to become general manager of CSKA Moscow of the Kontinental Hockey League?"
)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "/data/Search-R1/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
)
QUESTION = os.environ.get("QUESTION", DEFAULT_QUESTION).strip()
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
TOPK = env_int("TOPK", 3)
MAX_TURNS = env_int("MAX_TURNS", 5)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 512)
TEMPERATURE = env_float("TEMPERATURE", 0.7)

if QUESTION and QUESTION[-1] != "?":
    QUESTION += "?"

CURR_EOS = [151645, 151643]
SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for target_id, target_length in zip(self.target_ids, self.target_lengths):
            target = torch.as_tensor(target_id, device=input_ids.device)
            if torch.equal(input_ids[0, -target_length:], target):
                return True
        return False


def get_query(text):
    matches = SEARCH_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def search(query):
    payload = {"queries": [query], "topk": TOPK, "return_scores": True}
    started = time.time()
    response = requests.post(RETRIEVER_URL, json=payload, timeout=120)
    response.raise_for_status()
    elapsed = time.time() - started
    results = response.json()["result"][0]

    formatted = []
    for idx, doc_item in enumerate(results):
        content = doc_item["document"].get("contents") or ""
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        score = doc_item.get("score")
        score_text = f", score={score:.4f}" if isinstance(score, (int, float)) else ""
        formatted.append(f"Doc {idx + 1}(Title: {title}{score_text}) {text}")

    print(f"\n[retriever] query={query!r} topk={TOPK} seconds={elapsed:.3f}")
    return "\n".join(formatted) + "\n"


def build_prompt(question):
    prompt = (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. For example, "
        "<answer> Beijing </answer>. "
        f"Question: {question}\n"
    )
    return prompt


def main():
    print("model_id:", MODEL_ID)
    print("retriever_url:", RETRIEVER_URL)
    print("question:", QUESTION)
    print("max_turns:", MAX_TURNS)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    prompt = build_prompt(QUESTION)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    target_sequences = [
        "</search>",
        " </search>",
        "</search>\n",
        " </search>\n",
        "</search>\n\n",
        " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    print("\n################# [Start Reasoning + Searching] ##################\n")
    print(prompt)

    for turn in range(1, MAX_TURNS + 1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
        )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\n################# [Turn {turn} Model Output] ##################\n")
        print(output_text)

        if outputs[0][-1].item() in CURR_EOS or "<answer>" in output_text:
            print("\n################# [Finished] ##################\n")
            break

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        query = get_query(full_text)
        if not query:
            print("\n[stop] no valid <search> query found")
            break

        search_results = search(query)
        search_text = SEARCH_TEMPLATE.format(
            output_text=output_text,
            search_results=search_results,
        )
        print("\n################# [Injected Information] ##################\n")
        print(search_text)
        prompt += search_text
    else:
        print("\n[stop] reached MAX_TURNS")


if __name__ == "__main__":
    main()
