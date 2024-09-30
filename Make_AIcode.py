# *****************************************************************************
#
# brief : CSV 파일에서 MBPP 데이터셋을 읽어 각 코드 프롬프트를 LLM에 보내 AI code를 생성
# 생성된 코드는 데이터셋의 새로운 열 'AI_code'에 추가되며, 업데이트된 데이터셋은 새로운 CSV 파일로 저장.
#
# file : Perplexity_TEST.py
# date : 2024/09/02
# author : 김선민
#
# *****************************************************************************

import ollama
import pandas as pd

# CSV 파일 경로
csv_file_path = "MBPP_Data(original).csv"

df = pd.read_csv(csv_file_path, encoding='utf-8')

# AI_code 열 추가
df['AI_code'] = ""

# prompt 열에 대해 코드 생성
for index, row in df.iterrows():
    prompt_text = row['text']
    print("------------------------------------------------")
    print(f"Processing prompt {index+1}/{len(df)}: {prompt_text}")
    result = ollama.generate(
        model="llama3.1:70b",
        prompt=f"-{prompt_text}-\d"
        f"Write a Python function that performs the required task based on the provided description. The code should not include any comments, explanations, print statements, or examples. Provide only the function code, nothing else, and please return only one Python code block."

    )

    # response 키의 값을 가져옴
    ai_code = result.get('response', '')
    print(ai_code)
    print("------------------------------------------------")
    # AI_code 열에 코드 저장
    df.at[index, 'AI_code'] = ai_code

# 새로운 CSV 파일로 저장
output_file_path = "MBPP_DATASET_AI_llama3.1:70b_(justcode).csv"
df.to_csv(output_file_path, encoding='utf-8', index=False)

print(f"AI 코드가 추가된 파일이 '{output_file_path}'로 저장되었습니다.")