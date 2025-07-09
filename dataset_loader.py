# dataset_loader.py
from datasets import load_dataset

def load_medmcqa_subset(limit=5000):
    dataset = load_dataset("medmcqa", split=f"train[:{limit}]")

    def format_entry(entry):
        return {
            "question": entry["question"],
            "formatted": (
                f"Q: {entry['question']}\n"
                f"A. {entry['opa']}  B. {entry['opb']}  C. {entry['opc']}  D. {entry['opd']}\n"
                f"Correct Answer: {entry['cop']}\n"
                f"Explanation: {entry['exp']}"
            )
        }

    return [format_entry(entry) for entry in dataset]
