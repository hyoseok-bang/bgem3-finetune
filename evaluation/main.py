import torch
import torch.nn as nn
import mteb
import os 



task_list = [
    "BitextMining",
    "Classification",
    "MultilabelClassification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
    "InstructionRetrieval",
    "Speed",
]

"""
1천 개 이상의 task (여기서 task 라는게 어느 한 데이터에 대한 모델 평가 실험) 가
위 task_list 와 같이 카테고리 별로 나뉘어져 있습니다. 
카테고리는 총 11개 입니다. 

Classification 카테고리에 속하는 task 의 경우 해당 task 이름 내 데이터에 대한 모델의 
텍스트 분류 평가 실험이라 보시면 될거 같습니다.

"""

os.makedirs("results",exist_ok=True)

model_name = "ksyint/checkpoint1" # upskyy/bge-m3-korean # BAAI/bge-m3 # ksyint/checkpoint1
model = mteb.get_model(model_name)

"""
Default 한 코드에서는 우리가 Fine tune 하기 이전의 bge-m3 를 가져옵니다. 
우리의 학습된 bge-m3 로 대체를 하고 싶으면 huggingface model 에 따로 push를 한 이후에 
user_name/bge-m3_checkpoint278 형태로 model_name 에 온라인에서 다운로드 하는 형태로 가져오거나

아니면 로컬에서 이미 저장되어있는 checkpoint를 바로 연결을 하려면 만약 main.py 가 
/workspace/eval 안에 있다면 eval안에 user_name 폴더를 만들고 그 user_name 폴더 밑에 파인튠 한 bge-m3_checkpoint278 를 넣습니다. 
즉 model_name 은 무조건 "/" 가 하나 있어야하며 "A/B" 형태입니다.
로컬에서 model_name = "user_name/bge-m3_checkpoint278" 로 하고 /workspace/eval 내의 main.py 를 실행시킵니다.

"""

with open("available_experiments.txt", "r") as lists:
    possible_list = [line.strip() for line in lists]

"""
available_experiments.txt 를 제외한 task (데이터 셋) 은 404 connection error 가 일어나고 있습니다. 
즉 해당 txt 파일 내 데이터 셋 (벤치마크) 에 대해서만 evaluation을 할수 있습니다.

그래도 txt 안에 없는 데이터 셋에 대해서 eval 성공을 하였다면 카카오톡에 말씀 부탁드립니다.

"""



for i in range(len(task_list)):
    
    tasks = mteb.get_tasks(task_types=[task_list[i]], languages = ["eng", "kor"])

    """
    task 타입 카테고리 하나하나 씩 순차적으로 evaluation 이 돌아갑니다.
    즉 예를 들면, 모든 classification 카테고리 데이터셋 에 대한 모든 eval 이 끝난 이후에야 
    다음 모든 retrieval 데이터셋에 대한 실험으로 넘어갑니다.
    
    한국어, 영어만 지정하였습니다.  

    """

    for i2 in range(len(tasks)):

        if str(tasks[i2]) in possible_list:
            index=str(tasks[i2]).index("(")
            task_name=str(tasks[i2])[0:index]
            A=model_name.replace("/","_")

            evaluation = mteb.MTEB(tasks=[tasks[i2]])  
            results = evaluation.run(model, output_folder=f"results/{A}/{task_list[i]}/{task_name}")

            """
            어느 한 카테고리 내의 어느 한 데이터셋 (한 벤치마크) 에 대한 실험을 진행합니다. 

            """

        else:
            print(f"{str(tasks[i2])} currently not available")