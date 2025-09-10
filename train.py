import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
import mirdata
import os
import numpy as np
import wandb

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data import split_dataset
from model import BiLSTMChordClassifier, BiLSTMCRFChordClassifier, TransformerChordClassifier, TCNChordClassifier, MambaChordClassifier

def get_model(model_name, **kwargs):
    if model_name == "bilstm":
        return BiLSTMChordClassifier(**kwargs)
    elif model_name == "bilstm_crf":
        return BiLSTMCRFChordClassifier(**kwargs)
    elif model_name == "transformer":
        return TransformerChordClassifier(**kwargs)
    elif model_name == "tcn":
        return TCNChordClassifier(**kwargs)
    elif model_name == "mamba":
        return MambaChordClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
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

    print(f"[DEBUG] Confusion matrix saved to {save_path}")


class LogEpochCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["epoch"] = state.epoch  # 添加 epoch 到 wandb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a chord recognition model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size for LSTM.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--model", type=str, default="bilstm", help="Model type to use.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    billboard = mirdata.initialize('billboard')
    model = get_model(args.model, input_dim=24, hidden_dim=args.hidden_dim, num_classes=25)
    train_dataset, val_dataset, test_dataset = split_dataset(billboard)
    label_encoder = train_dataset.label_encoder
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    exp = f"{args.model}_{args.hidden_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_epochs_{args.epochs}"
    wandb.init(project="chord_recognition", name=exp)
    wandb.config.update(args)
    batch_size = args.batch_size
    
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        num_train_epochs=args.epochs,
        report_to="wandb"
    )
    wandb.config.update(train_args)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=get_metrics(),
        callbacks=[ConfusionMatrixCallback(label_encoder), LogEpochCallback()]
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
        dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn),
        label_encoder=label_encoder,
        device=args.device if hasattr(args, "device") else "cuda" if torch.cuda.is_available() else "cpu",
        save_path="./confusion_matrices/confmat_test.png",
        title="Confusion Matrix (Test Set)"
    )
    