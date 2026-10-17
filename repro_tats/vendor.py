"""Validated mechanisms vendored from the reference pack (TaTS reproduction).

Sources: TaTS/utils/reproduction.py (encode_texts), TaTS/utils/tools.py
(EarlyStopping with the non-finite guard), TaTS/utils/metrics.py
(full-element MSE/MAE). Ported verbatim in behavior; the fork's own copies
are left untouched so the git diff stays minimal and reviewable.
"""
import torch
import numpy as np


def encode_texts(texts, tokenizer, model, pool_type="avg", batch_size=32,
                 use_fullmodel=False):
    """Encode in bounded batches, masking padding identically in train/val/test."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size],
                               return_tensors="pt", padding=True,
                               truncation=True, max_length=256)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)
            if ids.numel() == 0 or (mask.sum(dim=1) == 0).any():
                raise ValueError("Tokenizer produced an empty text representation")
            if use_fullmodel:
                embeddings = model(input_ids=ids,
                                   attention_mask=mask).last_hidden_state
            else:
                embeddings = model.get_input_embeddings()(ids)
            mask = mask.unsqueeze(-1)
            if pool_type == "avg":
                pooled = (embeddings * mask).sum(1) / mask.sum(1)
            elif pool_type == "max":
                pooled = embeddings.masked_fill(mask == 0,
                                                -torch.inf).amax(1)
            elif pool_type == "min":
                pooled = embeddings.masked_fill(mask == 0,
                                                torch.inf).amin(1)
            else:
                raise ValueError("Unsupported pooling: %s" % pool_type)
            if not torch.isfinite(pooled).all():
                raise ValueError("Non-finite text embeddings")
            chunks.append(pooled.detach())
    model.train(was_training)
    return torch.cat(chunks)


class EarlyStopping:
    """Validation-only stopping; refuses to checkpoint non-finite losses."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        if not np.isfinite(val_loss):
            raise ValueError("Validation loss is not finite; refusing to "
                             "save a checkpoint")
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: %d out of %d"
                  % (self.counter, self.patience), flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print("Validation loss decreased (%.6f --> %.6f). Saving model ..."
                  % (self.val_loss_min, val_loss), flush=True)
        torch.save(model.state_dict(), str(path) + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def metric(pred, true):
    """Full-element MSE/MAE in the standardized space (reference metrics)."""
    diff = pred - true
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    return mae, mse
