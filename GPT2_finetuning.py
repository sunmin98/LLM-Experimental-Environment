# *****************************************************************************
#
# brief : GPT-2 모델을 MBPP 데이터셋을 활용해 파인튜닝하기 위한 코드
#          - GPT-2 기본 모델을 이용해 특정 코드 데이터셋에 맞게 파인튜닝 진행
#          - 파인튜닝 후 학습된 모델과 토크나이저를 저장
#
# file : GPT2_Finetuning.py
# date : 2024/10/24
# author : 김선민
#
# *****************************************************************************


import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# 1. CSV 파일 불러오기
data = pd.read_csv("DATA/AI+/CODETEST_llama3.1_70b_(ONLY_AI).csv")  # 여기서 'your_dataset.csv'는 네 CSV 파일의 경로
texts = data['AI_code'].tolist()  # 열 이름에 맞게 수정해줘야 해 (예: 'AI_code')

# 2. 데이터셋을 Hugging Face의 Dataset 형식으로 변환
dataset = Dataset.from_dict({'text': texts})

# 3. 토크나이저 및 모델 불러오기 (padding token 설정 추가)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 패딩 토큰이 없기 때문에 eos_token을 pad_token으로 설정
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # 토크나이저 업데이트 후 모델의 임베딩 크기 재조정

# 4. 텍스트를 토크나이즈
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

# 5. 데이터 컬레이터 설정 (마스크 언어 모델링에 필요)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2는 마스크 언어 모델이 아니므로 False로 설정
)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # 학습 반복 횟수
    per_device_train_batch_size=4,  # 배치 크기 (메모리 여유에 맞게 조절 가능)
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
)

# 7. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 8. 학습 시작
trainer.train()

# 9. 학습된 모델 및 토크나이저 저장
trainer.save_model("./fine_tuned_gpt2(CODETEST_llama_70b)")
tokenizer.save_pretrained("./fine_tuned_gpt2(CODETEST_llama_70b)")  # 토크나이저도 함께 저장
