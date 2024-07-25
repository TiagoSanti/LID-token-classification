from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from transformers import BertForTokenClassification, BertTokenizerFast, pipeline
import torch

ID2TAG = {
    0: "B-banco",
    1: "B-cidade",
    2: "B-empresa",
    3: "B-empresario",
    4: "B-estado",
    5: "B-organização",
    6: "B-outras_pessoas",
    7: "B-pais",
    8: "B-politico",
    9: "B-valor_financeiro",
    10: "I-banco",
    11: "I-cidade",
    12: "I-empresa",
    13: "I-empresario",
    14: "I-estado",
    15: "I-organização",
    16: "I-outras_pessoas",
    17: "I-pais",
    18: "I-politico",
    19: "I-valor_financeiro",
    20: "O",
}


class NewModel(LabelStudioMLBase):
    model_version = "1.0"
    model_name = "TiagoSanti/bert-ner-finetuned"
    max_length = 510
    stride = 256

    def setup(self):
        self.model = BertForTokenClassification.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model.eval().cuda() if torch.cuda.is_available() else self.model
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        predictions = [self.process_task(task) for task in tasks]
        print("\n", predictions)
        return ModelResponse(predictions=predictions)

    def process_task(self, task: Dict) -> Dict:
        text = task["data"]["full_content"]
        results, entity_ranges = [], []
        state = {
            "current_entity": None,
            "accumulated_score": 0,
            "parts_count": 0,
            "full_word": "",
        }

        for i in range(0, len(text), self.max_length - self.stride):
            chunk = text[i : i + self.max_length]
            offset = i
            raw_predictions = self.ner_pipeline(chunk)
            results, entity_ranges = self.process_predictions(
                raw_predictions, results, entity_ranges, offset, state
            )

        results.sort(key=lambda x: x["value"]["start"], reverse=True)
        return {"result": results}

    def process_predictions(self, predictions, results, entity_ranges, offset, state):
        for pred in predictions:
            label_id = int(pred["entity"].split("_")[-1])
            label = ID2TAG[label_id]
            if label == "O":
                continue

            entity_type = label[2:]
            entity_begin = label[:1] == "B"
            if entity_begin or (
                state["current_entity"]
                and state["current_entity"]["label"] != entity_type
            ):
                self.finalize_current_entity(state, results, entity_ranges)

            self.update_or_start_entity(pred, entity_type, entity_begin, offset, state)
        self.finalize_current_entity(state, results, entity_ranges)
        return results, entity_ranges

    def finalize_current_entity(self, state, results, entity_ranges):
        if state["current_entity"] is not None:
            mean_score = state["accumulated_score"] / state["parts_count"]
            start, end = (
                state["current_entity"]["start"],
                state["current_entity"]["end"],
            )
            self.append_or_replace_entity(
                state["current_entity"]["label"],
                start,
                end,
                mean_score,
                state["full_word"],
                results,
                entity_ranges,
            )
            state["current_entity"] = None

    def append_or_replace_entity(
        self, label, start, end, score, word, results, entity_ranges
    ):
        overlap_found = False
        for idx, (rng_start, rng_end, rng_score) in enumerate(entity_ranges):
            if start <= rng_end and end >= rng_start:
                overlap_found = True
                if score > rng_score:
                    entity_ranges[idx] = (start, end, score)
                    results[idx] = self.create_result(label, start, end, word, score)
                    print(
                        f"Replaced existing entity due to higher score at index {idx}. New entity: '{label}' with score {score:.2f}: {word}"
                    )
                else:
                    print(
                        f"Discarded new entity '{label}' due to lower score compared to existing entity at index {idx}: {word}"
                    )
                break
        if not overlap_found:
            entity_ranges.append((start, end, score))
            results.append(self.create_result(label, start, end, word, score))
            print(
                f"Added new entity: '{label}' from index {start} to {end} with score {score:.2f}: {word}"
            )

    def create_result(self, label, start, end, text, score):
        return {
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "labels": [label],
                "start": start,
                "end": end,
                "text": text,
                "score": score,
            },
        }

    def update_or_start_entity(self, pred, entity_type, entity_begin, offset, state):
        if entity_begin:
            state["current_entity"] = {
                "label": entity_type,
                "start": pred["start"] + offset,
                "end": pred["end"] + offset,
            }
            state["accumulated_score"] = pred["score"]
            state["parts_count"] = 1
            state["full_word"] = pred["word"].replace("##", "")
        else:
            if state["current_entity"]:
                state["current_entity"]["end"] = pred["end"] + offset
                state["accumulated_score"] += pred["score"]
                state["parts_count"] += 1
                state["full_word"] += pred["word"].replace("##", "")
