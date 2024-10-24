# *****************************************************************************
#
# brief :  LeetCode와 Codeforces의 문제를 이용하여 AI 코드를 파인튜닝한
# GPT2 모델을 실험하였으며, 실험 결과 3개의 오탐이 발생하여 최종 정확도는 99.6%로 평가됨
#
#
# file : Perplexity_Finetunig_TEST.py
# date : 2024/10/24
# author : 김선민
#
# *****************************************************************************



import pandas as pd
from LLM_Control import calculate_perplexity_GPT2_Finetunig

print("화이팅~")

# CSV 파일 불러오기
file_path = 'DATA/AI+/MBPP_DATASET_AI_llama3.1:70b_(justcode).csv'
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
        original_perplexity = calculate_perplexity_GPT2_Finetunig(original_code)
    except:
        original_perplexity = None

    try:
        ai_perplexity = calculate_perplexity_GPT2_Finetunig(ai_code)
    except:
        ai_perplexity = None

    # T/F 구별.
    if original_perplexity > ai_perplexity:
        Detecting = True
    else:
        Detecting = False

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
result_df.to_csv('파인튜닝안한거gpt2_MBPP에다실험.csv', index=False)
