# *****************************************************************************
#
# brief : AI로 생성된 코드와 인간이 작성한 코드의 퍼플렉시티를 계산하여, AI 코드의 퍼플렉시티가 인간 코드보다 낮은지 여부를 실험.
# 퍼플렉시티가 낮을수록 AI code일 확률이 높기때문에 최종적으로 riginal_perplexity > ai_perplexity일시 TRUE.
#
# file : Perplexity_TEST.py
# date : 2024/09/26
# author : 김선민
#
# *****************************************************************************

import pandas as pd
import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# GPT2 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# 퍼플렉시티 계산 함수
def calculate_perplexity(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity


# CSV 파일 불러오기
file_path = 'MBPP_DATASET_AI_llama3.1:70b_(justcode).csv'
df = pd.read_csv(file_path)
num = 0

# 각 코드와 AI 코드에 대해 퍼플렉시티 계산 및 비교
results = []
for index, row in df.iterrows():
    task_id = row['task_id']
    original_code = row['code']
    ai_code = row['AI_code']

    # 원본 코드와 AI 코드의 퍼플렉시티 계산
    try:
        original_perplexity = calculate_perplexity(original_code)
    except:
        original_perplexity = None

    try:
        ai_perplexity = calculate_perplexity(ai_code)
    except:
        ai_perplexity = None

    # T/F 구별.
    if original_perplexity > ai_perplexity:
        Detecting = True
    else: Detecting = False

    # 결과 저장
    results.append({
        "task_id": task_id,
        "original_perplexity": original_perplexity,
        "ai_perplexity": ai_perplexity,
        "Detecting": Detecting
    })

    print(results[num])
    num += 1

# 결과를 데이터프레임으로 저장 및 출력
result_df = pd.DataFrame(results)
print(result_df)

# 결과를 CSV 파일로 저장
result_df.to_csv('perplexity_comparison_results.csv', index=False)