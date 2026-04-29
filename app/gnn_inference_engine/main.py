#!/usr/bin/env python3
"""
GNN Inference Engine (FR4) – обучение и инференс гетерогенной графовой нейронной сети.

Принимает HeteroData граф (после Vulnerability Enricher) с обогащёнными признаками.
Обучает гетерогенную GNN (HeteroConv) на рёбрах DEPENDS_ON, VULNERABLE_TO и HAS_VERSION.
Вычисляет risk_score для каждого узла Version.

Использование:
    python gnn_inference_engine.py --nx-input enriched/nx_graph.json \
                                   --het-input enriched/hetero_data.pt \
                                   --output-dir ./scored
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from networkx.readwrite import json_graph
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logger = logging.getLogger("gnn_inference_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class HeteroVulnerabilityGNN(torch.nn.Module):
    def __init__(self, version_in_channels: int, package_in_channels: int,
                 vuln_in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()

        self.conv1 = HeteroConv({
            ('version', 'DEPENDS_ON', 'version'): SAGEConv(version_in_channels, hidden_channels),
            ('version', 'VULNERABLE_TO', 'vulnerability'): GATConv(
                (version_in_channels, vuln_in_channels), hidden_channels,
                heads=1, concat=False, add_self_loops=False
            ),
            ('package', 'HAS_VERSION', 'version'): SAGEConv(
                (package_in_channels, version_in_channels), hidden_channels
            ),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('version', 'DEPENDS_ON', 'version'): SAGEConv(hidden_channels, hidden_channels),
            ('version', 'VULNERABLE_TO', 'vulnerability'): GATConv(
                (hidden_channels, hidden_channels), hidden_channels,
                heads=1, concat=False, add_self_loops=False
            ),
        }, aggr='mean')

        self.classifier = Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        if 'version' in x_dict:
            return self.classifier(x_dict['version']).squeeze(-1)
        else:
            raise KeyError("No 'version' node type in output")

# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train_model(hetero_data: HeteroData, epochs: int = 200, lr: float = 0.001,
                device: str = 'cpu') -> HeteroVulnerabilityGNN:
    """Обучает гетерогенную модель на графе.

    Args:
        hetero_data: Граф с заполненными признаками и полем y для version.
        epochs: Количество эпох обучения.
        lr: Скорость обучения.
        device: Устройство ('cpu' или 'cuda').

    Returns:
        Обученная модель.
    """
    # Перенос на устройство
    hetero_data = hetero_data.to(device)

    # Извлечение целевой переменной
    y = hetero_data['version'].y.float().to(device)
    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()
    if num_pos == 0:
        logger.error("No positive examples (vulnerable versions) found. Cannot train.")
        sys.exit(1)
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)

    # Инициализация модели
    version_in_channels = hetero_data['version'].x.size(1)
    package_in_channels = hetero_data['package'].x.size(1) if 'package' in hetero_data.node_types else 0
    vuln_in_channels = hetero_data['vulnerability'].x.size(1) if 'vulnerability' in hetero_data.node_types else 0

    model = HeteroVulnerabilityGNN(
        version_in_channels=version_in_channels,
        package_in_channels=package_in_channels,
        vuln_in_channels=vuln_in_channels,
        hidden_channels=64,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = (torch.sigmoid(logits) > 0.5).float()
            acc = (pred == y).float().mean().item()
            logger.info("Epoch %d/%d, Loss: %.4f, Accuracy: %.4f", epoch + 1, epochs, loss.item(), acc)

    return model


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def run_inference(hetero_data: HeteroData, nx_graph: nx.MultiDiGraph,
                  model: HeteroVulnerabilityGNN, device: str = 'cpu') -> Tuple[HeteroData, nx.MultiDiGraph]:
    """Выполняет инференс и добавляет risk_score в оба представления графа."""
    hetero_data = hetero_data.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        risk_scores = torch.sigmoid(logits)

    hetero_data['version'].risk_score = risk_scores.cpu()

    # Синхронизация с NetworkX
    version_names = hetero_data['version'].names
    for idx, name in enumerate(version_names):
        node_id = f"version/{name}"
        if node_id in nx_graph:
            nx_graph.nodes[node_id]["risk_score"] = round(risk_scores[idx].item(), 4)

    logger.info("Inference completed. risk_score added to %d version nodes.", risk_scores.size(0))
    return hetero_data, nx_graph


# ---------------------------------------------------------------------------
# Подготовка разметки
# ---------------------------------------------------------------------------

def add_labels(hetero_data: HeteroData) -> HeteroData:
    """Создаёт целевую переменную y для узлов Version.

    Узел Version считается уязвимым (y=1), если он имеет хотя бы одно
    исходящее ребро VULNERABLE_TO.
    """
    num_versions = hetero_data['version'].x.size(0)
    y = torch.zeros(num_versions, dtype=torch.long)

    edge_type = ('version', 'VULNERABLE_TO', 'vulnerability')
    if edge_type in hetero_data.edge_index_dict:
        edge_index = hetero_data[edge_type].edge_index
        if edge_index.size(1) > 0:
            vulnerable_indices = edge_index[0].unique()
            y[vulnerable_indices] = 1

    hetero_data['version'].y = y
    logger.info("Labels added: %d vulnerable out of %d versions.", y.sum().item(), num_versions)
    return hetero_data


# ---------------------------------------------------------------------------
# Нормализация признаков
# ---------------------------------------------------------------------------

def normalize_features(hetero_data: HeteroData) -> HeteroData:
    """Применяет StandardScaler к признакам всех узлов.

    Для каждого типа узлов нормализация по среднему и стандартному отклонению.
    Пропускает константные признаки (std == 0).
    """
    for node_type in hetero_data.node_types:
        x = hetero_data[node_type].x.numpy()
        # Проверка на константные столбцы
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # чтобы не делить на ноль
        mean = x.mean(axis=0, keepdims=True)
        x_norm = (x - mean) / std
        hetero_data[node_type].x = torch.from_numpy(x_norm).float()
        logger.debug("Normalized features for '%s': mean=%s, std=%s", node_type, mean, std)
    return hetero_data


# ---------------------------------------------------------------------------
# Загрузка NetworkX графа
# ---------------------------------------------------------------------------

def load_nx_graph(path: Path) -> nx.MultiDiGraph:
    """Загружает NetworkX граф из JSON-файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN Inference Engine (FR4) – обучение и инференс гетерогенной модели."
    )
    parser.add_argument("--nx-input", type=Path, required=True,
                        help="Путь к NetworkX JSON-файлу (после Vulnerability Enricher).")
    parser.add_argument("--het-input", type=Path, required=True,
                        help="Путь к HeteroData файлу (после Vulnerability Enricher).")
    parser.add_argument("--output-dir", type=Path, default=Path("./scored"),
                        help="Директория для сохранения результата.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Количество эпох обучения.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Устройство для вычислений ('cpu' или 'cuda').")
    parser.add_argument("--verbose", action="store_true",
                        help="Подробное логирование.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.het_input.is_file():
        logger.error("HeteroData file '%s' not found.", args.het_input)
        sys.exit(1)
    if not args.nx_input.is_file():
        logger.error("NetworkX JSON file '%s' not found.", args.nx_input)
        sys.exit(1)

    # Загрузка графов
    hetero_data = torch.load(args.het_input, weights_only=False)
    nx_graph = load_nx_graph(args.nx_input)
    logger.info("Graphs loaded.")

    # Нормализация признаков
    hetero_data = normalize_features(hetero_data)

    # Разметка
    hetero_data = add_labels(hetero_data)

    # Обучение
    logger.info("Starting training...")
    model = train_model(hetero_data, epochs=args.epochs, device=args.device)

    # Сохранение модели
    model_path = args.output_dir / "gnn_model.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # Инференс (обновляет оба графа)
    hetero_data, nx_graph = run_inference(hetero_data, nx_graph, model, args.device)

    # Сохраняем обновлённые графы
    output_het = args.output_dir / "hetero_data.pt"
    torch.save(hetero_data, output_het)
    logger.info("Scored HeteroData saved to %s", output_het)

    output_nx = args.output_dir / "nx_graph_scored.json"
    with open(output_nx, "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(nx_graph), f, indent=2, ensure_ascii=False)
    logger.info("Scored NetworkX graph saved to %s", output_nx)


if __name__ == "__main__":
    main()