import token
from requests import get
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.fine_tuning import ContrastiveTrainer, CustomTrainer
from utils.helper import load_multiple_data, make_train_dataset, TokenizerHelper

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import math
from utils.config import get_secret


# 2. 허깅페이스 Hub에 로그인 (명령줄 또는 코드에서 수행 가능)
# huggingface-cli login

# 또는 코드에서 토큰을 사용하여 로그인
from huggingface_hub import login

login(token=get_secret("HF_TOKEN"))  # Hugging Face Hub에 로그인

# 3. 다운로드할 모델 이름 정의
model_name = "BAAI/bge-m3"

# 4. 모델 및 토크나이저 로드
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SentenceTransformer(model_name)


# 5. Fine-tuning을 위한 데이터셋 준비
data_dir = "dataset/"
if isinstance(data_dir, str):
    data = load_multiple_data(data_dir)
else:
    data = load_multiple_data(data_dir)

train_examples = make_train_dataset(data)
# 데이터셋 생성
train_dataset = Dataset.from_dict(
    {
        "text": [item["text"] for item in train_examples],
        "labels": [item["label"] for item in train_examples],
    }
)
print("Hugging Face Dataset 생성 완료.")
print(f"Dataset 예시 (첫 번째 항목): {train_dataset[0] if train_dataset else None}")
tokenizer_helper = TokenizerHelper(tokenizer)
tokenized_train_dataset = train_dataset.map(
    tokenizer_helper.tokenize_function, batched=True
)
print("데이터셋 토큰화 완료.")

# # 7. 학습을 위한 설정 (TrainingArguments)
# output_dir = "./bge-m3-finetuned"  # 모델 저장 경로
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     push_to_hub=True,  # 허깅페이스 Hub에 업로드 활성화
#     hub_model_id="hyoseok1989/bge-m3-finetuned",
# )

# 9. Trainer 객체 생성 및 학습
# trainer = ContrastiveTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_train_dataset,
# )
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_train_dataset,
#     compute_loss_func=
# )
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_train_dataset,
#     # Pass custom parameters
#     target_norm=1,
#     lambda_scale=0.02,  # Tune this
#     loss_type="cosine",  # Choose 'cosine', 'mse', or 'infonce'
# )
# print("Trainer 객체 생성 완료.")

# # 모델 파인튜닝 시작
# print("모델 파인튜닝 시작...")
# trainer.train()

# 10. 학습된 모델을 허깅페이스 Hub에 업로드
# trainer.push_to_hub()

try:
    # repo_id는 "사용자명/모델명" 또는 "기관명/모델명" 형식으로 지정
    repo_id = "hyoseok1989/bge-m3-finetuned"
    commit_msg = "Fine-tuned BGE-M3 for Korean-English code-switching task"

    model.save_to_hub(
        repo_id=repo_id,
        commit_message=commit_msg,
        private=False,
        exist_ok=True,  # True if overwrite the existing model
    )
    print(f"모델이 Hugging Face Hub에 성공적으로 업로드되었습니다: {repo_id}")

except Exception as e:
    print(f"모델 업로드 중 오류 발생: {e}")


# print(
#     f"Fine-tuned 모델이 허깅페이스 Hub에 업로드되었습니다: https://huggingface.co/{training_args.hub_model_id}"
# )
