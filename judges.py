import json

from harness import Case, Label, Verdict


class BaselineJudge:
    def __init__(self, client , model="gpt-4o"):
        self.client = client
        self.model = model

    def __call__(self, case: Case) -> Verdict:
        convo = f"Conversation:\n{case.context}\n" if case.context else ""
        prompt = ("You are going to be given a question, gold answer, and model answer. "
                "Compare the model answer to the gold answer.\n"
                + convo
                + "Question: " + case.question + "\n"
                + "Gold answer: " + case.expected + "\n"
                + "Model answer: " + case.actual + "\n"
                + "Please respond with 'yes' if the model answer is correct, or 'no' if it is incorrect.")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            )
        text = resp.choices[0].message.content
        passed = "yes" in text.lower()
        return Verdict(
            label=Label.SUPPORTED if passed else Label.MISSED,
            reasoning=text,
        )


class RubricJudge:
    def __init__(self, client , model="gpt-4o"):
        self.client = client
        self.model = model

    def __call__(self, case: Case) -> Verdict:
        convo = f"Conversation:\n{case.context}\n" if case.context else ""
        prompt = (
                "You are going to be given a question, gold answer, and model answer. "
                "Compare the model answer to the gold answer.\n"
                f"{convo}"
                f"Question: {case.question}\nGold answer: {case.expected}\nModel answer: {case.actual}\n"
                "You will return a label like this:\n"
                f"{Label.SUPPORTED.value}  - the answer is supported by the conversation\n"
                f"{Label.MISSED.value}  - the answer is missing / not found\n"
                f"{Label.NO_BASIS.value}  - the answer has no basis in the conversation\n"
                f"{Label.CORRECT_IDK.value}  - correctly declines when the info is absent\n"
                f"{Label.CONTRADICTED.value}  - the answer contradicts the conversation\n"
                'Reply ONLY with JSON: {"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}'
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            )
        data = json.loads(resp.choices[0].message.content)
        return Verdict(
            label=Label(data["label"].strip().lower()),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
        )