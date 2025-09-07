import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
import mirdata
import os
import numpy as np
import wandb

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data import split_dataset

class BiLSTMChordClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_classes=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_values, labels=None):
        out, _ = self.lstm(input_values)
        logits = self.classifier(out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"logits": logits, "loss": loss}
        
        return {"logits": logits}
    
    

def collate_fn(batch):
    Xs, Ys, _ = zip(*batch)

    Xs = [x for x in Xs]  # [seq_len, input_dim]
    Ys = [y for y in Ys]  # [seq_len]

    X_padded = pad_sequence(Xs, batch_first=True)  # [batch, max_len, input_dim]
    Y_padded = pad_sequence(Ys, batch_first=True, padding_value=-100)  # [batch, max_len]

    return {"input_values": X_padded, "labels": Y_padded}




def get_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        preds_flat, labels_flat = [], []
        for p, l in zip(preds, labels):
            mask = l != -100
            preds_flat.extend(p[mask])
            labels_flat.extend(l[mask])

        acc = accuracy_score(labels_flat, preds_flat)
        f1_macro = f1_score(labels_flat, preds_flat, average="macro")
        f1_weighted = f1_score(labels_flat, preds_flat, average="weighted")

        metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }


        return metrics

    return compute_metrics


from transformers import TrainerCallback

class ConfusionMatrixCallback(TrainerCallback):
    def __init__(self, label_encoder, save_dir="confusion_matrices"):
        self.LE = label_encoder
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        dataloader = kwargs["eval_dataloader"]
        device = args.device if hasattr(args, "device") else "cuda" if torch.cuda.is_available() else "cpu"

        save_path = os.path.join(self.save_dir, f"confmat_epoch_{int(state.epoch)}.png")
        title = f"Confusion Matrix (Epoch {int(state.epoch)})"

        save_confusion_matrix(
            model=model,
            dataloader=dataloader,
            label_encoder=self.LE,
            device=device,
            save_path=save_path,
            title=title
        )


def save_confusion_matrix(model, dataloader, label_encoder, device, save_path, title="Confusion Matrix"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values=inputs)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)

            for p, l in zip(preds, labels):
                mask = l != -100
                all_preds.extend(p[mask].cpu().numpy())
                all_labels.extend(l[mask].cpu().numpy())



    cm = confusion_matrix(all_labels, all_preds, labels=range(len(label_encoder.classes_)))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[TEST] Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    billboard = mirdata.initialize('billboard')
    model = BiLSTMChordClassifier(input_dim=24, hidden_dim=128, num_classes=24)
    train_dataset, val_dataset, test_dataset = split_dataset(billboard)
    label_encoder = train_dataset.label_encoder
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    
    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=512,
        learning_rate=1e-3,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        num_train_epochs=20,
        report_to="wandb"
    )
    exp = f"bilstm_batch_{args.per_device_train_batch_size}_lr_{args.learning_rate}_epochs_{args.num_train_epochs}"
    wandb.init(project="chord_recognition", name=exp)
    wandb.config.update(args)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=get_metrics(),
        callbacks=[ConfusionMatrixCallback(label_encoder)]
    )
    
    trainer.train()
    trainer.save_model(f"./checkpoints/{exp}")
    
    
    metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test"
    )
    print(metrics)
    wandb.log(metrics)
    save_confusion_matrix(
        model=model,
        dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=512, collate_fn=collate_fn),
        label_encoder=label_encoder,
        device=args.device if hasattr(args, "device") else "cuda" if torch.cuda.is_available() else "cpu",
        save_path="./confusion_matrices/confmat_test.png",
        title="Confusion Matrix (Test Set)"
    )
    