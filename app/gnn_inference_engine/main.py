#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from networkx.readwrite import json_graph
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logger = logging.getLogger("gnn_inference_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Константы модели
# ---------------------------------------------------------------------------
HIDDEN_CHANNELS = 64
DROPOUT = 0.3


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class HeteroVulnerabilityGNN(torch.nn.Module):
    """Гетерогенная GNN для прогнозирования уязвимых версий пакетов.

    Архитектура: два слоя HeteroConv с SAGE и GAT, затем линейный классификатор.
    """

    def __init__(
        self,
        version_in_channels: int,
        package_in_channels: int,
        vuln_in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        dropout: float = DROPOUT,
    ) -> None:
        """Инициализация гетерогенной GNN.

        Args:
            version_in_channels: Размер входных признаков узлов‑версий.
            package_in_channels: Размер входных признаков узлов‑пакетов.
            vuln_in_channels: Размер входных признаков узлов‑уязвимостей.
            hidden_channels: Размер скрытого представления.
            dropout: Вероятность дропаута.
        """
        super().__init__()

        self.conv1 = HeteroConv(
            {
                ("version", "DEPENDS_ON", "version"): SAGEConv(
                    version_in_channels, hidden_channels
                ),
                ("version", "VULNERABLE_TO", "vulnerability"): GATConv(
                    (version_in_channels, vuln_in_channels),
                    hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                ),
                ("package", "HAS_VERSION", "version"): SAGEConv(
                    (package_in_channels, version_in_channels), hidden_channels
                ),
            },
            aggr="mean",
        )

        self.conv2 = HeteroConv(
            {
                ("version", "DEPENDS_ON", "version"): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                ("version", "VULNERABLE_TO", "vulnerability"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                ),
            },
            aggr="mean",
        )

        self.classifier = Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple, torch.Tensor]
    ) -> torch.Tensor:
        """Прямой проход модели.

        Args:
            x_dict: Словарь признаков по типам узлов.
            edge_index_dict: Словарь индексов рёбер по типам отношений.

        Returns:
            Логиты для узлов типа 'version' (размер [num_versions]).

        Raises:
            KeyError: Если в выходном словаре отсутствует ключ 'version'.
        """
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {
            key: F.dropout(x, p=self.dropout, training=self.training)
            for key, x in x_dict.items()
        }

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        if "version" not in x_dict:
            raise KeyError("No 'version' node type in output")
        return self.classifier(x_dict["version"]).squeeze(-1)


# ---------------------------------------------------------------------------
# Вспомогательные функции обработки данных
# ---------------------------------------------------------------------------
def normalize_features(hetero_data: HeteroData) -> HeteroData:
    """Нормализует признаки узлов «per‑graph» (Z‑score).

    Используется в трансдуктивном режиме. Для каждого типа узлов вычисляется
    среднее и стандартное отклонение по текущему графу, после чего
    выполняется нормализация.

    Args:
        hetero_data: Гетерогенный граф с необработанными признаками.

    Returns:
        Тот же объект HeteroData с нормализованными признаками.
    """
    for node_type in hetero_data.node_types:
        x = hetero_data[node_type].x.numpy()
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        mean = x.mean(axis=0, keepdims=True)
        x_norm = (x - mean) / std
        hetero_data[node_type].x = torch.from_numpy(x_norm).float()
        logger.debug("Normalized features for '%s'", node_type)
    return hetero_data


def normalize_with_stats(
    hetero_data: HeteroData, stats: Dict[str, Dict[str, float]]
) -> HeteroData:
    """Нормализует признаки с использованием глобальных статистик.

    Применяется в индуктивном режиме, чтобы согласовать масштаб признаков
    с обучающей выборкой.

    Args:
        hetero_data: Гетерогенный граф.
        stats: Словарь вида {node_type: {'mean': ..., 'std': ...}}.

    Returns:
        HeteroData с нормализованными признаками.
    """
    for node_type in hetero_data.node_types:
        if node_type not in stats:
            logger.debug("No norm stats for node type '%s', skipping.", node_type)
            continue
        x = hetero_data[node_type].x.numpy()
        mean = stats[node_type]["mean"]
        std = stats[node_type]["std"]
        x_norm = (x - mean) / std
        hetero_data[node_type].x = torch.from_numpy(x_norm).float()
    return hetero_data


def load_nx_graph(path: Path) -> nx.MultiDiGraph:
    """Загружает NetworkX граф из JSON‑файла.

    Args:
        path: Путь к файлу.

    Returns:
        Восстановленный MultiDiGraph.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------
def run_inference(
    hetero_data: HeteroData,
    nx_graph: nx.MultiDiGraph,
    model: HeteroVulnerabilityGNN,
    device: str = "cpu",
) -> Tuple[HeteroData, nx.MultiDiGraph]:
    """Выполняет инференс обученной модели и обновляет оба графа.

    Добавляет предсказанный `risk_score` в узлы‑версии HeteroData и в
    соответствующие узлы NetworkX графа.

    Args:
        hetero_data: Нормализованный граф с признаками.
        nx_graph: Параллельный NetworkX граф.
        model: Обученная GNN.
        device: Устройство выполнения ('cpu' или 'cuda').

    Returns:
        Кортеж (hetero_data, nx_graph) с добавленными атрибутами risk_score.
    """
    hetero_data = hetero_data.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        risk_scores = torch.sigmoid(logits)

    hetero_data["version"].risk_score = risk_scores.cpu()
    version_names = hetero_data["version"].names
    for idx, name in enumerate(version_names):
        node_id = f"version/{name}"
        if node_id in nx_graph:
            nx_graph.nodes[node_id]["risk_score"] = round(
                risk_scores[idx].item(), 4
            )

    logger.info(
        "Inference completed. risk_score added to %d version nodes.",
        risk_scores.size(0),
    )
    return hetero_data, nx_graph


# ---------------------------------------------------------------------------
# Трансдуктивное обучение
# ---------------------------------------------------------------------------
def train_transductive(
    hetero_data: HeteroData,
    epochs: int,
    lr: float,
    device: str,
) -> HeteroVulnerabilityGNN:
    """Обучает модель с нуля на единственном графе (трансдуктивный режим).

    Args:
        hetero_data: Нормализованный граф с метками y.
        epochs: Количество эпох обучения.
        lr: Скорость обучения.
        device: Устройство для вычислений.

    Returns:
        Обученная модель.

    Raises:
        SystemExit: Если в данных нет положительных примеров уязвимых версий.
    """
    hetero_data = hetero_data.to(device)
    y = hetero_data["version"].y.float().to(device)
    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()
    if num_pos == 0:
        logger.warning("No positive examples found. Training skipped; all risks will be 0.")
        return None

    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)

    version_in = hetero_data["version"].x.size(1)
    package_in = (
        hetero_data["package"].x.size(1)
        if "package" in hetero_data.node_types
        else 0
    )
    vuln_in = (
        hetero_data["vulnerability"].x.size(1)
        if "vulnerability" in hetero_data.node_types
        else 0
    )

    model = HeteroVulnerabilityGNN(
        version_in, package_in, vuln_in, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT
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
            logger.info(
                "Epoch %d/%d, Loss: %.4f, Accuracy: %.4f",
                epoch + 1,
                epochs,
                loss.item(),
                acc,
            )
    return model


# ---------------------------------------------------------------------------
# Индуктивное дообучение
# ---------------------------------------------------------------------------
def fine_tune(
    model: HeteroVulnerabilityGNN,
    hetero_data: HeteroData,
    epochs: int,
    lr: float,
    device: str,
    patience: int = 5,
) -> None:
    """Дообучает предобученную модель на новом графе.

    Ранние слои обучаются с пониженной скоростью, классификатор – с обычной.
    Используется AdamW с шедулером ReduceLROnPlateau.

    Args:
        model: Предобученная модель.
        hetero_data: Новый граф с метками.
        epochs: Количество эпох дообучения.
        lr: Базовая скорость обучения.
        device: Устройство вычислений.
        patience: Терпение для шедулера (по умолчанию 5).
    """
    hetero_data = hetero_data.to(device)
    y = hetero_data["version"].y.float().to(device)
    pos_weight = torch.tensor(
        [(y.size(0) - y.sum()) / (y.sum() + 1e-5)]
    ).to(device)

    param_groups = [
        {"params": model.conv1.parameters(), "lr": lr * 0.1},
        {"params": model.conv2.parameters(), "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=0.5
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Fine-tune epoch %d/%d, loss: %.4f (best: %.4f)",
                epoch + 1,
                epochs,
                loss.item(),
                best_loss,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Точка входа – парсинг аргументов и запуск обучения / инференса.

    Возможные режимы:
      - transductive: обучение с нуля и инференс.
      - inductive: загрузка модели, опциональное дообучение и инференс.
    """
    parser = argparse.ArgumentParser(
        description="GNN Inference Engine (FR4) – обучение и инференс."
    )
    parser.add_argument(
        "--nx-input",
        type=Path,
        required=True,
        help="Путь к JSON‑файлу NetworkX графа.",
    )
    parser.add_argument(
        "--het-input",
        type=Path,
        required=True,
        help="Путь к HeteroData .pt файлу.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./scored"),
        help="Директория для сохранения результатов.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Количество эпох обучения (трансдуктивный режим).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство выполнения: 'cpu' или 'cuda'.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Подробное логирование."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transductive", "inductive"],
        default="transductive",
        help="transductive: обучить с нуля; inductive: загрузить модель.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Путь к предобученной модели (обязателен для inductive).",
    )
    parser.add_argument(
        "--fine-tune",
        type=int,
        default=0,
        help="Число эпох дообучения в индуктивном режиме (0 – без дообучения).",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=0.001,
        help="Скорость обучения для дообучения.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Путь к файлу norm_stats.pt (обязателен для inductive).",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Проверка входных файлов
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

    # Нормализация в зависимости от режима
    if args.mode == "inductive":
        if args.norm_stats is None or not args.norm_stats.is_file():
            logger.error(
                "For inductive mode, a valid --norm-stats file is required."
            )
            sys.exit(1)
        stats = torch.load(args.norm_stats, weights_only=False)
        hetero_data = normalize_with_stats(hetero_data, stats)
    else:
        hetero_data = normalize_features(hetero_data)

    # Метки достижимости уже должны быть в hetero_data (добавлены Vulnerability Enricher)
    if 'y' not in hetero_data['version']:
        logger.warning("No reachability labels found, assuming all versions safe.")
        hetero_data['version'].y = torch.zeros(hetero_data['version'].num_nodes, dtype=torch.long)

    # Получение модели
    model = None
    if args.mode == "transductive":
        logger.info("Transductive mode: training from scratch...")
        model = train_transductive(
            hetero_data,
            epochs=args.epochs,
            lr=0.001,
            device=args.device,
        )
    else:  # inductive
        if args.model_path is None or not args.model_path.is_file():
            logger.error(
                "For inductive mode, a valid --model-path is required."
            )
            return
        logger.info("Inductive mode: loading model from %s", args.model_path)

        version_in = hetero_data["version"].x.size(1)
        package_in = (
            hetero_data["package"].x.size(1)
            if "package" in hetero_data.node_types
            else 0
        )
        vuln_in = (
            hetero_data["vulnerability"].x.size(1)
            if "vulnerability" in hetero_data.node_types
            else 0
        )
        model = HeteroVulnerabilityGNN(
            version_in, package_in, vuln_in, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT
        ).to(args.device)
        model.load_state_dict(
            torch.load(args.model_path, map_location=args.device)
        )

        if args.fine_tune > 0:
            logger.info("Fine-tuning for %d epochs...", args.fine_tune)
            fine_tune(
                model,
                hetero_data,
                epochs=args.fine_tune,
                lr=args.fine_tune_lr,
                device=args.device,
            )

    # Инференс или заполнение нулями, если модель не обучена
    if model is not None:
        hetero_data, nx_graph = run_inference(
            hetero_data, nx_graph, model, args.device
        )
    else:
        # Нет ни одной уязвимой версии → все риски = 0
        risk_scores = torch.zeros(hetero_data["version"].num_nodes)
        hetero_data["version"].risk_score = risk_scores
        version_names = hetero_data["version"].names
        for idx, name in enumerate(version_names):
            node_id = f"version/{name}"
            if node_id in nx_graph:
                nx_graph.nodes[node_id]["risk_score"] = 0.0
        logger.info("All version nodes marked as safe (risk_score=0.0).")

    # Сохранение результатов
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель только если она обучалась
    if (args.mode == "transductive" or args.fine_tune > 0) and model is not None:
        model_path = args.output_dir / "gnn_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved to %s", model_path)

    output_het = args.output_dir / "hetero_data.pt"
    torch.save(hetero_data, output_het)
    logger.info("Scored HeteroData saved to %s", output_het)

    output_nx = args.output_dir / "nx_graph_scored.json"
    with open(output_nx, "w", encoding="utf-8") as f:
        json.dump(
            json_graph.node_link_data(nx_graph),
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Scored NetworkX graph saved to %s", output_nx)


if __name__ == "__main__":
    main()