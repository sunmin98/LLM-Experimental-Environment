# *****************************************************************************
#
# brief : GPT-2 및 CodeGen 모델을 활용한 코드 퍼플렉시티 계산 및 코드 재작성 실험
#         - GPT-2 기본 모델 및 파인튜닝된 모델을 이용하여 퍼플렉시티를 계산
#         - CodeGen 모델로 긴 코드 블록에 대한 퍼플렉시티 계산 수행
#         - LLM을 사용하여 코드를 재작성하고, 재작성된 코드의 퍼플렉시티를 측정
#
# file : LLM_Control.py
# date : 2024/10/24
# author : 김선민
#
# *****************************************************************************


import ollama
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel


def calculate_perplexity_Gpt2(code):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(code, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity


###파인튜닝한 gpt2 사용 함수###
def calculate_perplexity_GPT2_Finetunig(code):
    model_name = "fine_tuned_gpt2(CODETEST_llama_70b)"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(code, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity


def calculate_perplexity_CodeGen(text, max_length=512):
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    if input_ids.size(1) > max_length:
        split_ids = torch.split(input_ids, max_length, dim=1)
    else:
        split_ids = [input_ids]

    total_loss = 0
    num_tokens = 0

    for split in split_ids:
        with torch.no_grad():
            outputs = model(split, labels=split)
            loss = outputs.loss
            total_loss += loss.item() * split.size(1)
            num_tokens += split.size(1)

    avg_loss = total_loss / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()


def LLM_rewrite_code(code, num_rewrites=3):
    for i in range(num_rewrites):
        # 코드 재작성
        result = ollama.generate(
            model="codellama:latest",  # LLM 모델 선택
            prompt=f"-{code}-\nWrite a Python function that performs the required task based on the provided description. The code should not include any comments, explanations, print statements, or examples. Provide only the function code, nothing else, and please return only one Python code block."
        )

        # response 키의 값을 가져옴
        code = result.get('response', '')
        print(f"재작성된 코드 {i + 1} ->", code)

        # 퍼플렉시티 계산
        perplexity_result = calculate_perplexity_Gpt2(code)
        print(f"퍼플렉시티 {i + 1}: {perplexity_result}")
        print("------------------------------------------------")

    return perplexity_result
