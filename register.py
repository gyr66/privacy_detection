from model import BertCrfForTokenClassification

BertCrfForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
model = BertCrfForTokenClassification.from_pretrained(
    "gyr66/RoBERTa-ext-large-crf-chinese-finetuned-ner"
)
model.push_to_hub("gyr66/RoBERTa-ext-large-crf-chinese-finetuned-ner")
