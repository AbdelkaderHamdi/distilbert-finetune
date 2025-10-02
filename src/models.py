from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def build_model(model_name, num_labels=2, freeze_base=False):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if freeze_base:
        # DistilBERT / many HF models store backbone as `.distilbert` or `.bert`. Try both.
        if hasattr(model, "distilbert"):
            backbone = model.distilbert
        elif hasattr(model, "bert"):
            backbone = model.bert
        else:
            backbone = None

        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
    return model
