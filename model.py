import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertCrfForTokenClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        dummy_logits = torch.zeros_like(logits).to(logits.device)

        valid_lens = attention_mask.sum(dim=1) - 2
        logits = logits[:, 1:]
        labels_mask = torch.arange(logits.size(1)).to(
            valid_lens.device
        ) < valid_lens.unsqueeze(1)

        seq_label_ids = self.crf.decode(logits, mask=labels_mask)

        loss = None
        if labels is not None:
            labels = labels[:, 1:]
            is_pad = labels == -100
            labels.masked_fill_(is_pad, 0)
            assert torch.eq(~is_pad, labels_mask).all().item(), "mask assertion failed "
            loss = -self.crf(logits, labels, mask=labels_mask, reduction="mean")

        padded_list = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lst) for lst in seq_label_ids],
            batch_first=True,
            padding_value=0,
        )
        padded_list = torch.nn.functional.pad(
            padded_list, (0, logits.size(1) - padded_list.shape[1])
        )
        padded_list = torch.nn.functional.one_hot(
            padded_list, num_classes=logits.size(2)
        )
        assert dummy_logits.size(1) == padded_list.size(1) + 1, "size assertion failed"
        dummy_logits[:, 1:] = padded_list

        if not return_dict:
            output = (dummy_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=dummy_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
