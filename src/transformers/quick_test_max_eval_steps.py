# quick_test_max_eval_steps.py
# A tiny test that mocks the Trainer.prediction_step so we don't need a real model.
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

class TinyDataset(Dataset):
    def __init__(self, n=50, seq_len=8):
        self.n = n
        self.seq_len = seq_len
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {
            "input_ids": torch.zeros(self.seq_len, dtype=torch.long),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.tensor(0, dtype=torch.long),
        }

class MockTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        batch_size = 4
        logits = torch.randn(batch_size, 2)
        labels = torch.zeros(batch_size, dtype=torch.long)
        return None, logits, labels

# dummy model (not used by mock prediction_step)
import torch.nn as nn
class DummyModel(nn.Module):
    def forward(self, *a, **k):
        return None

ds = TinyDataset(n=50, seq_len=8)
model = DummyModel()
args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=8, do_train=False)

trainer = MockTrainer(model=model, args=args, eval_dataset=ds)
print("Running evaluate(max_eval_steps=2) ... should stop after ~2 batches.")
res = trainer.evaluate(max_eval_steps=2)
print("Returned metrics keys:", list(res.keys()))
print("Test complete.")