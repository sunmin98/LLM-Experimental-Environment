# *****************************************************************************
#
# brief : 캐글에서 leetcode(1825개), codeforce(8343개)를 합친 10168문제를 읽어 llama3.1
# 으로 AI코드를 생성
#
# file : Make_ai_CODETEST(llama3.1).py.py
# date : 2024/10/24
# author : 김선민
#
# *****************************************************************************

import ollama
import pandas as pd

# CSV 파일 경로
csv_file_path = "DATA/code/CODETEST_dataset.csv"

df = pd.read_csv(csv_file_path, encoding='utf-8')

# AI_code 열 추가
df['AI_code'] = ""

# prompt 열에 대해 코드 생성
for index, row in df.iterrows():
    prompt_text = row['prompt']
    print("------------------------------------------------")
    print(f"Processing prompt {index+1}/{len(df)}: {prompt_text}")
    result = ollama.generate(
        model="llama3.1:70b", #LLM 모델 선택 사용
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
output_file_path = "DATA/AI+/CODETEST:llama3.1:70b_(justcode).csv"
df.to_csv(output_file_path, encoding='utf-8', index=False)

print(f"AI 코드가 추가된 파일이 '{output_file_path}'로 저장되었습니다.")