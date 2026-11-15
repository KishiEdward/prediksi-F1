"""Definisi arsitektur neural network (PyTorch, CPU).

Sesuai laporan: LSTM SATU ARAH (bukan BiLSTM, karena prediksi pada lap N hanya
boleh memakai informasi sampai lap N) dengan dua head keluaran dari representasi
temporal yang sama (multi-task):
- head regresi   : laps remaining (1 nilai)
- head klasifikasi: compound berikutnya (5 kelas + softmax)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModelLSTM(nn.Module):
    """LSTM multi-task: encoder sekuensial + 2 head (regresi & klasifikasi).

    Dipakai untuk model LSTM tunggal maupun hybrid. Pada hybrid, vektor input
    per timestep sudah memuat estimasi awal XGBoost + fitur pra-balapan
    (mekanisme integrasi: fitur tambahan di tiap timestep).
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, n_classes: int = 5) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Head regresi: estimasi sisa lap menuju pit berikutnya
        self.head_regresi = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
        # Head klasifikasi: probabilitas compound berikutnya
        self.head_klasifikasi = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, window, fitur) -> (pred_laps: (batch,), logits: (batch, kelas))."""
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])  # representasi temporal H_t pada lap terakhir
        pred_laps = self.head_regresi(h).squeeze(-1)
        logits = self.head_klasifikasi(h)
        return pred_laps, logits

    def config(self) -> dict:
        """Konfigurasi arsitektur (untuk disimpan bersama artifact)."""
        return {
            "input_size": self.input_size, "hidden_size": self.hidden_size,
            "num_layers": self.num_layers, "n_classes": self.n_classes,
        }


def muat_model(path: str | object, config: dict) -> ModelLSTM:
    """Muat ModelLSTM dari file state_dict + config arsitektur."""
    model = ModelLSTM(**{k: config[k] for k in
                         ("input_size", "hidden_size", "num_layers", "n_classes")})
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
