from torchcrf import CRF  # pip install pytorch-crf
from torch import nn
import torch
from mamba_ssm import Mamba  # pip install mamba-ssm


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

class BiLSTMCRFChordClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_classes=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, input_values, labels=None):
        lstm_out, _ = self.lstm(input_values)  # (B, T, 2*H)
        emissions = self.classifier(lstm_out)  # (B, T, C)

        if labels is not None:
            # mask out -100s
            mask = labels != -100
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return {"logits": emissions, "loss": loss}
        else:
            # decode returns best path
            pred = self.crf.decode(emissions)
            # pad prediction to match input length
            max_len = emissions.size(1)
            logits = torch.full_like(labels, fill_value=-100)
            for i, seq in enumerate(pred):
                logits[i, :len(seq)] = torch.tensor(seq)
            return {"logits": logits}
        

class TransformerChordClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_classes=24, num_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_values, labels=None):
        x = self.input_proj(input_values)  # (B, T, H)
        x = self.encoder(x)                # (B, T, H)
        logits = self.classifier(x)        # (B, T, C)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"logits": logits, "loss": loss}
        return {"logits": logits}
    
    
class TCNChordClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_classes=24, num_layers=4):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(num_layers):
            layers.append(nn.Conv1d(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            ))
            layers.append(nn.ReLU())
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_values, labels=None):
        x = input_values.transpose(1, 2)  # (B, D, T)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (B, T, D)
        logits = self.classifier(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"logits": logits, "loss": loss}
        return {"logits": logits}
    
    

class MambaChordClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_classes=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_values, labels=None):
        x = self.input_proj(input_values)  # (B, T, H)
        x = self.mamba(x)                  # (B, T, H)
        logits = self.classifier(x)        # (B, T, C)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"logits": logits, "loss": loss}
        return {"logits": logits}