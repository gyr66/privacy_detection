import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoModelForTokenClassification,
)
from model import BertCrfForTokenClassification
import evaluate
import numpy as np
import argparse
from torch.optim import AdamW


def align_labels_with_tokens(labels, word_ids):
    return [-100 if word_id is None else labels[word_id] for word_id in word_ids]


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = evaluate.load("seqeval")
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--check_point", type=str, default="hfl/chinese-roberta-wwm-ext-large"
    )
    parser.add_argument(
        "--repo_name", type=str, default="RoBERTa-ext-large-chinese-finetuned-crf-ner"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--crf_lr", type=float, default=2e-3)
    parser.add_argument("--freeze_base_model", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    dataset = load_dataset("gyr66/privacy_detection")

    dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)
    dataset["validation"] = dataset.pop("test")

    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    check_point = args.check_point
    tokenizer = AutoTokenizer.from_pretrained(check_point, ignore_mismatched_sizes=True)
    optimizer = None
    if args.use_crf:
        model = BertCrfForTokenClassification.from_pretrained(
            check_point,
            num_labels=len(id2label),
        )
        crf_parameters = [p for n, p in model.named_parameters() if "crf" in n]
        remaining_parameters = [
            p for n, p in model.named_parameters() if "crf" not in n
        ]
        optimizer = AdamW(
            [
                {"params": remaining_parameters, "lr": args.lr},
                {"params": crf_parameters, "lr": args.crf_lr},
            ]
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            check_point,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )
    model.config.id2label = id2label
    model.config.label2id = label2id

    if args.freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        args.repo_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        logging_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, None) if optimizer else None,
    )

    if not args.dry_run:
        trainer.train()
        trainer.save_model(args.repo_name)
    metric = trainer.evaluate()
    print("Evaluate the best model on the validation set:")
    print(metric)

    if args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

# python train.py --check_point ./RoBERTa-ext-large-chinese-finetuned-ner --repo_name RoBERTa-ext-large-crf-chinese-finetuned-ner --push_to_hub --epochs 10
