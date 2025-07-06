from datasets import load_dataset

def load_medmcqa_subset(limit=5000):
    dataset = load_dataset("medmcqa", split=f"train[:{limit}]")

    def format_entry(entry):
        answer_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
        correct = answer_map.get(entry["cop"], entry["cop"].upper())
        return {
            "question": entry["question"],
            "formatted": (
                f"Q: {entry['question']}\n"
                f"A. {entry['opa']}  B. {entry['opb']}  C. {entry['opc']}  D. {entry['opd']}\n"
                f"Correct Answer: {correct}\n"
                f"Explanation: {entry['exp'] or 'No explanation provided.'}"
            )
        }

    return [format_entry(entry) for entry in dataset]
