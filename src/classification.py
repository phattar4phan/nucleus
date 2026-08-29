import sys
import time
import json
import signal
import numpy as np

from pathlib import Path
from typing import Any
from colorama import Fore, init
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

init(autoreset=True)

class ChunkInDexDataset(Dataset):
    def __init__(self, memmap, labels, seq_length, chunk_indices):
        self.data = memmap
        self.labels = labels
        self.seq_length = seq_length
        self.chunk_indices = chunk_indices
        
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        chunk_idx = self.chunk_indices[idx]
        start = chunk_idx * self.seq_length
        end = start + self.seq_length
        chunk = np.array(self.data[start:end], dtype=np.int64)
        label = self.labels[chunk_idx]
        return torch.from_numpy(chunk), torch.tensor(label, dtype=torch.long)

def create_dataloaders(binary_file: Path | str, labels_file: Path | str, record_count_files: Path | str, dtype: np.dtype, seq_length: int, batch_size: int, num_workers: int, seed: int = 42):
    raw_memmap = np.memmap(binary_file, dtype=dtype, mode="r")
    labels = np.load(labels_file)
    record_chunk_counts = np.load(record_count_files)
    
    boundaries = [0]
    for count in record_chunk_counts:
        boundaries.append(boundaries[-1] + count)
        
    records = list(zip(boundaries[:-1], boundaries[1:]))
    
    class_records = defaultdict(list)
    for start, end in records:
        if start == end:
            continue
        
        class_id = int(labels[start])
        class_records[class_id].append((start, end))
        
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    
    for class_id, recs in class_records.items():
        rng.shuffle(recs)
        n = len(recs)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - n_test)
        
        for start, end in recs[:n_train]:
            train_idx.extend(range(start, end))
        for start, end in recs[n_train:n_train+n_val]:
            val_idx.extend(range(start, end))
        for start, end in recs[n_train+n_val:]:
            test_idx.extend(range(start, end))
            
    train_dataset = ChunkInDexDataset(raw_memmap, labels, seq_length, train_idx)
    val_dataset = ChunkInDexDataset(raw_memmap, labels, seq_length, val_idx)
    test_dataset = ChunkInDexDataset(raw_memmap, labels, seq_length, test_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    return train_dataloader, val_dataloader, test_dataloader

class CNNTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, channels, blocks, layers):
        super().__init__()
        
        # embed tokens to vector space with positional embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, 512, embedding_dim) * 0.02)
        
        # intial projection to cnn channels
        self.initial_conv = nn.Sequential(
            nn.Conv1d(embedding_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
        # multilayer cnn with residual connections
        # as many layer as defined in 'blocks'
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels)
            ]) for _ in range(blocks)
        ])
        # simply another GELU activation function
        self.res_act = nn.GELU()
        
        # dynamic sequence compression
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        # transformer encoder
        encoder = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=layers)
        
        # multiclass output head
        self.fc = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        B, S = x.size()
        
        # embedding (batch, seq_length, embedded_dim)
        x = self.embedding(x) + self.positional_embedding[:, :S, :]
        
        # forward pass (batch, channels, seq_length) *cnn
        x = x.permute(0, 2, 1)
        x = self.initial_conv(x)
        
        for conv1, bn1, act, conv2, bn2 in self.conv_layers:
            residual = x
            out = act(bn1(conv1(x)))
            out = bn2(conv2(out))
            x = self.res_act(out + residual)
        
        # compress sequence to 32 tokens (batch, channels, 32)
        x = self.adaptive_pool(x)
        
        # forward pass (batch, 32, seq_length) *transformer
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        return self.fc(x)

def checkpoint(model, optimizer, scheduler, epoch, val_accuracy, val_loss, filename: Path | str):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": epoch,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_acc": val_accuracy,
        "val_loss": val_loss
    }
    torch.save(checkpoint, filename)
    
def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item() * inputs.size(0)
            prediction = outputs.argmax(dim=-1)
            correct += (prediction == targets).sum().item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    accuracy = (correct / total_samples) * 100.00
    return avg_loss, accuracy

def format(seconds):
    minute, second = divmod(int(seconds), 60)
    return f"{minute:02d}:{second:02d}"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # get num_classes and the mapping
    with open("mapped.json", "r") as file:
        mapping = json.load(file)
    num_classes = len(mapping)
    
    print(f'{Fore.LIGHTBLUE_EX}Device{Fore.RESET}: {(str(device)).upper()}')
    print(f'{Fore.LIGHTBLUE_EX}Classes{Fore.RESET}: {num_classes}')
    
    VOCAB_SIZE = 390625
    SEQ_LENGTH = 128
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        binary_file="tokens.bin",
        labels_file="chunks.npy",
        record_count_files="records.npy",
        dtype=np.uint32,
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    model = CNNTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=num_classes,
        embedding_dim=256,
        channels=256,
        blocks=4, # conv1d block
        layers=2 # transformer layer
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'{Fore.LIGHTBLUE_EX}Paramters{Fore.RESET}: {total_params:,}')
    print(f'{Fore.LIGHTBLUE_EX}Model{Fore.RESET}: {model}')
    
    model = torch.compile(model, mode="reduce-overhead")
    
    # defined:
    #   - criterion: CrossEntropyLoss
    #   - Optimizer: AdamW
    #   - Scheduler: CosineAnnealingLR
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # state variables
    best_val_acc = 0.0
    last_val_acc = 0.0
    last_val_loss = 0.0
    current_epoch = 0
    
    # incase of any interruptions, it automatically saves the best model
    def interruption(signal, frame):
        print(f'\n{Fore.LIGHTRED_EX}Interrupted{Fore.RESET}: auto-saving checkpoint...')
        checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=current_epoch,
            val_accuracy=last_val_acc,
            val_loss=last_val_loss,
            filename="checkpoints/interrupted.pt"
        )
        sys.exit(0)
        
    # bind system signals
    signal.signal(signal.SIGINT, interruption)
    signal.signal(signal.SIGTERM, interruption)
    
    total_batches = len(train_dataloader)
    try:
        for epoch in range(1, EPOCHS+1):
            current_epoch = epoch
            model.train()
            
            running_loss = torch.zeros(1, device=device)
            correct = torch.zeros(1, device=device)
            total_samples = 0
            epoch_start_time = time.perf_counter()
            
            for current_batch, (inputs, targets) in enumerate(train_dataloader, 1):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # sanity check for holy loss
                if not torch.isfinite(loss):
                    print(f'\n{Fore.LIGHTRED_EX}Error{Fore.RESET}: non-infinite loss at epoch: {epoch}, batch: {current_batch}, {loss.item()}')
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_size = inputs.size(0)
                running_loss += loss.detach() * batch_size
                prediction = outputs.argmax(dim=-1)
                correct += (prediction == targets).sum()
                total_samples += batch_size
                
                avg_loss = running_loss / total_samples
                train_accuracy = (correct / total_samples) * 100.0 # train accuracy
                percent = (current_batch / total_batches) * 100.0 # percent completion (current_batch/total_batch)
                
                elasped = time.perf_counter() - epoch_start_time
                second_per_batch = elasped / current_batch
                estimate_seconds = second_per_batch * total_batches
                
                current_time = format(elasped)
                estimated_time = format(estimate_seconds)
                
                logs = (
                    f"\r[{Fore.LIGHTWHITE_EX}{epoch}{Fore.RESET}/{EPOCHS}]: {Fore.LIGHTCYAN_EX}{current_batch}{Fore.RESET}/{Fore.LIGHTCYAN_EX}{total_batches}{Fore.RESET} "
                    f"{Fore.LIGHTBLUE_EX}{percent:.2f}%{Fore.RESET}, {Fore.LIGHTYELLOW_EX}{current_time}{Fore.RESET}/{Fore.LIGHTMAGENTA_EX}{estimated_time}{Fore.RESET} "
                    f"Loss: {Fore.LIGHTRED_EX}{(avg_loss.item()):.4f}{Fore.RESET}, Accuracy: {Fore.LIGHTYELLOW_EX}{(train_accuracy.item()):.2f}%{Fore.RESET} "
                    f"VL: {Fore.LIGHTGREEN_EX}{last_val_loss:.4f}{Fore.RESET}, VA: {Fore.LIGHTBLUE_EX}{last_val_acc:.2f}%"
                )
                # use sys.stdout to bypass I/O buffer and execution speed
                sys.stdout.write(logs)
                sys.stdout.flush()
        
            # first epoch is expected to show 0.0000, 0.00% VL and VA
            last_val_loss, last_val_acc = evaluate(model, val_dataloader, criterion, device)
            scheduler.step()    
        
            logs = (
                f"\r[{Fore.LIGHTWHITE_EX}{epoch}{Fore.RESET}/{EPOCHS}]: {Fore.LIGHTCYAN_EX}{current_batch}{Fore.RESET}/{Fore.LIGHTCYAN_EX}{total_batches}{Fore.RESET} "
                    f"{Fore.LIGHTBLUE_EX}{percent:.2f}%{Fore.RESET}, {Fore.LIGHTYELLOW_EX}{current_time}{Fore.RESET}/{Fore.LIGHTMAGENTA_EX}{estimated_time}{Fore.RESET} "
                    f"Loss: {Fore.LIGHTRED_EX}{(avg_loss.item()):.4f}{Fore.RESET}, Accuracy: {Fore.LIGHTYELLOW_EX}{(train_accuracy.item()):.2f}%{Fore.RESET} "
                    f"VL: {Fore.LIGHTGREEN_EX}{last_val_loss:.4f}{Fore.RESET}, VA: {Fore.LIGHTBLUE_EX}{last_val_acc:.2f}%"
                )
            # use sys.stdout to bypass I/O buffer and execution speed
            sys.stdout.write(logs)
            sys.stdout.flush()
        
            # checkpoint saving if new epoch validation accuracy is higher than last validation accuracy
            if last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
                
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(raw_model.state_dict(), "checkpoints/best.pt")
                    
            checkpoint(model, optimizer, scheduler, epoch, last_val_acc, last_val_loss, filename="checkpoints/checkpoints.pt")
              
    except KeyboardInterrupt:
        print(f'{Fore.LIGHTRED_EX}Error{Fore.RESET}: Keyboard interrupted, saving progress...')
        checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=current_epoch,
            val_accuracy=last_val_acc,
            val_loss=last_val_loss,
            filename="checkpoints/keyboarded.pt"
        )

    # rebuild the same model used during training
    model = CNNTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=num_classes,
        embedding_dim=256,
        channels=256,
        blocks=4,
        layers=2
    ).to(device)

    # load the raw state_dict directly into uncomplied model
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt)

    # only compile if you actually want compiled inference speed;
    # not required for a one-off evaluation
    # model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
    print(f"\n[test]: Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")