import torch
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)
from datasets import Dataset, DatasetDict
import schedulefree
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import wandb
from dotenv import load_dotenv
import os


class CustomTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=self.args.learning_rate
        )


class NERModelTrainer:
    def __init__(self, dataset_path, model_name="bert-base-multilingual-cased"):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.tag2id = {}
        self.id2tag = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.full_dataset = self.load_and_split_data()
        self.model = self.setup_model()

    def load_and_split_data(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()

        grouped_lines = []
        current_tokens = []
        current_tags = []
        unique_tags = set()

        for line in lines:
            if line.startswith("-DOCSTART-") or not line.strip():
                if current_tokens:
                    grouped_lines.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue
            parts = line.split()
            current_tokens.append(parts[0])
            current_tags.append(parts[-1])
            unique_tags.add(parts[-1])

        if current_tokens:
            grouped_lines.append((current_tokens, current_tags))

        self.tag2id = {tag: i for i, tag in enumerate(sorted(unique_tags))}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}
        data = {
            "tokens": [x[0] for x in grouped_lines],
            "ner_tags": [[self.tag2id[tag] for tag in x[1]] for x in grouped_lines],
        }

        dataset = Dataset.from_dict(data)
        train_test = dataset.train_test_split(test_size=0.1)
        train_val = train_test["train"].train_test_split(test_size=0.1)
        return DatasetDict(
            {
                "train": train_val["train"],
                "validation": train_val["test"],
                "test": train_test["test"],
            }
        )

    def setup_model(self):
        num_labels = len(self.tag2id)
        return BertForTokenClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(
                        label[word_idx] if word_idx != previous_word_idx else -100
                    )
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        true_predictions = []

        for i in range(labels.shape[0]):
            label_list = labels[i]
            prediction_list = predictions[i]

            temp_true_labels = []
            temp_true_predictions = []

            for j in range(label_list.shape[0]):
                label_id = label_list[j]
                prediction_id = prediction_list[j]

                if label_id != -100:
                    temp_true_labels.append(self.id2tag[label_id])
                    temp_true_predictions.append(self.id2tag[prediction_id])

            true_labels.append(temp_true_labels)
            true_predictions.append(temp_true_predictions)

        return {
            "precision": precision_score(
                true_labels, true_predictions, zero_division=0
            ),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
            "report": classification_report(
                true_labels, true_predictions, zero_division=0
            ),
        }

    def train(self):
        load_dotenv()
        wandb.login()
        wandb.init(project="ner-finetuning", entity="your_wandb_username")

        dataset_info = {
            "train_size": len(self.full_dataset["train"]),
            "validation_size": len(self.full_dataset["validation"]),
            "test_size": len(self.full_dataset["test"]),
        }
        wandb.log(dataset_info)

        tokenized_datasets = self.full_dataset.map(
            self.tokenize_and_align_labels, batched=True
        )
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=5e-5,
            num_train_epochs=150,
            per_device_train_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=10,
            report_to="wandb",
            run_name="ner-finetuning",
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.test_evaluation = trainer.evaluate(tokenized_datasets["test"])
        wandb.log(self.test_evaluation)
        wandb.finish()

        self.model.save_pretrained("./model_finetuned")
        self.tokenizer.save_pretrained("./model_finetuned")

        self.model.push_to_hub(
            "bert-ner-finetuned", use_auth_token=os.getenv("HUGGINGFACE_API_KEY")
        )
        self.tokenizer.push_to_hub(
            "bert-ner-finetuned", use_auth_token=os.getenv("HUGGINGFACE_API_KEY")
        )

    def predict(self, text, score_threshold=0.5):
        model = BertForTokenClassification.from_pretrained("./model_finetuned")
        tokenizer = BertTokenizerFast.from_pretrained("./model_finetuned")

        model.eval()
        model.to(self.device)

        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )

        raw_predictions = ner_pipeline(text)

        translated_predictions = []
        for pred in raw_predictions:
            if pred["score"] < score_threshold:
                continue

            label_id = int(pred["entity"].split("_")[-1])
            pred["entity"] = self.id2tag[label_id]

            if pred["entity"] == "O":
                continue

            translated_predictions.append(pred)

        return translated_predictions


trainer = NERModelTrainer("datasets/labelled_dataset.conll")
trainer.train()
