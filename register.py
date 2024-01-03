from model import BertCrfForTokenClassification

BertCrfForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
model = BertCrfForTokenClassification.from_pretrained(
    "gyr66/relation_extraction_bert_base_uncased"
)
model.push_to_hub("gyr66/relation_extraction_bert_base_uncased")
