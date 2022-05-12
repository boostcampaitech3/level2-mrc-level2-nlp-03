import json
import heapq

def most_frequent(data):
    return max(data, key=data.count)

def main():
    """
        model.py에 정의된 RobertaQA, BertQA, ElectraQA에 대해 inference를 진행한 결과를 가지고 앙상블을 시도합니다.
    """
    with open("./outputs/test_dataset/.json") as f:
        one = json.load(f)

    with open("./outputs/test_dataset/.json") as f:
        two = json.load(f)

    with open("./outputs/test_dataset/.json") as f:
        three = json.load(f)

    query_id = list(one.keys())
    dic = {}
    
    for id in query_id:
        # 각 결과에 대해 같은 id를 가지는 답변을 모두 가져옵니다
        answer1 = one.get(id)
        answer2 = two.get(id)
        answer3 = three.get(id)

        arr = [answer1, answer2, answer3]

        answer = most_frequent(arr)
        
        
        dic[id] = answer

    # 결과를 prediction.json으로 저장합니다
    with open(
        "./outputs/test_dataset/ensemble_predictions.json", "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(dic, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
