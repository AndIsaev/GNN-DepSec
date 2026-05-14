#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_multigraph")
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
# Модель (аналогична используемой в gnn_inference_engine)
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
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]
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
# Нормализация на основе глобальных статистик
# ---------------------------------------------------------------------------
def compute_global_stats(graphs: List[HeteroData]) -> Dict[str, Dict[str, np.ndarray]]:
    """Вычисляет средние и стандартные отклонения по всем узлам всех типов.

    Проходит по всем предоставленным необработанным графам, накапливает
    значения признаков для каждого типа узлов и вычисляет общую статистику,
    устойчивую к нулевому std (заменяется на 1.0).

    Args:
        graphs: Список ненормализованных HeteroData.

    Returns:
        Словарь вида:
        { node_type: {'mean': np.ndarray, 'std': np.ndarray} }
    """
    accum: Dict[str, List[np.ndarray]] = {}
    for data in graphs:
        for node_type in data.node_types:
            x = data[node_type].x.numpy()
            accum.setdefault(node_type, []).append(x)

    stats = {}
    for node_type, arrays in accum.items():
        all_data = np.concatenate(arrays, axis=0)
        mean = all_data.mean(axis=0, keepdims=True)
        std = all_data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        stats[node_type] = {"mean": mean, "std": std}
        logger.info(
            "Global stats for '%s': mean shape %s, std shape %s",
            node_type,
            mean.shape,
            std.shape,
        )
    return stats


def normalize_with_stats(
    hetero_data: HeteroData, stats: Dict[str, Dict[str, np.ndarray]]
) -> HeteroData:
    """Нормализует признаки графа, используя глобальные статистики.

    Для каждого типа узлов, если он присутствует в словаре статистик,
    выполняется Z‑score нормализация: (x - mean) / std.

    Args:
        hetero_data: Гетерогенный граф с необработанными признаками.
        stats: Словарь со средними и стандартными отклонениями.

    Returns:
        Тот же объект HeteroData с нормализованными признаками.
    """
    for node_type in hetero_data.node_types:
        if node_type not in stats:
            continue
        x = hetero_data[node_type].x.numpy()
        mean = stats[node_type]["mean"]
        std = stats[node_type]["std"]
        x_norm = (x - mean) / std
        hetero_data[node_type].x = torch.from_numpy(x_norm).float()
    return hetero_data


def load_graphs(
    data_dir: Path, stats: Dict[str, Dict[str, np.ndarray]] | None = None
) -> List[HeteroData]:
    """Загружает все гетерогенные графы из подпапок директории.

    Рекурсивно ищет файлы `hetero_data.pt`. Если переданы глобальные
    статистики, нормализует признаки с их помощью.
    Метки достижимости (y) уже должны быть в данных; если отсутствуют,
    предполагаем отсутствие уязвимостей (все версии безопасны).

    Args:
        data_dir: Директория, содержащая подпапки проектов.
        stats: Опциональный словарь глобальных статистик для нормализации.

    Returns:
        Список загруженных и подготовленных HeteroData.

    Raises:
        FileNotFoundError: Если не найдено ни одного корректного файла.
    """
    graphs = []
    for pt_file in data_dir.rglob("hetero_data.pt"):
        try:
            data = torch.load(pt_file, weights_only=False)
            # Проверка, что граф не пустой и содержит рёбра для обновления версий
            has_version = 'version' in data.node_types and data['version'].x.size(0) > 0
            if not has_version:
                logger.warning("Graph %s has no version nodes, skipping.", pt_file)
                continue
            dst_to_version = False
            for e in data.edge_index_dict.keys():
                if len(e) == 3 and e[2] == 'version':
                    dst_to_version = True
                    break
            if not dst_to_version:
                logger.warning("Graph %s lacks edges targeting 'version', skipping.", pt_file)
                continue
            if stats is not None:
                data = normalize_with_stats(data, stats)
            # Убедимся, что метки y существуют; если нет – заполняем нулями
            if 'y' not in data['version']:
                logger.warning("No labels in %s, assuming all versions safe.", pt_file.parent.name)
                data['version'].y = torch.zeros(data['version'].num_nodes, dtype=torch.long)
            graphs.append(data)
            logger.info("Loaded project from %s", pt_file.parent.name)
        except Exception as e:
            logger.warning("Skipping %s: %s", pt_file, e)

    if not graphs:
        raise FileNotFoundError(f"No valid hetero_data.pt files found in {data_dir}")
    logger.info("Total loaded projects: %d", len(graphs))
    return graphs


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def train_on_multigraphs(
    data_dir: Path,
    epochs: int,
    lr: float,
    device: str,
    model_output: Path,
    val_path: Path | None = None,
) -> None:
    """Обучает модель на множестве проектов в индуктивном стиле.

    Последовательность действий:
    1. Загрузка сырых графов для вычисления глобальных статистик.
    2. Вычисление и сохранение нормализационных параметров.
    3. Нормализация всех графов этими параметрами.
    4. Обучение модели на всех тренировочных графах.
    5. Опциональная оценка на валидационном проекте.
    6. Сохранение обученной модели и нормализационных статистик.

    Args:
        data_dir: Директория с тренировочными проектами.
        epochs: Количество эпох обучения.
        lr: Скорость обучения.
        device: Устройство для вычислений ('cpu' или 'cuda').
        model_output: Путь для сохранения файла модели.
        val_path: Опциональный путь к проекту для валидации.
    """
    # Шаг 1: загрузка без нормализации для вычисления статистик
    raw_graphs = load_graphs(data_dir, stats=None)

    # Шаг 2: глобальные статистики
    global_stats = compute_global_stats(raw_graphs)

    # Шаг 3: нормализация всех тренировочных графов
    train_graphs = [normalize_with_stats(g, global_stats) for g in raw_graphs]

    # Шаг 4: загрузка валидационного проекта (если указан)
    val_graph = None
    if val_path:
        val_raw = load_graphs(val_path, stats=None)
        if val_raw:
            val_graph = normalize_with_stats(val_raw[0], global_stats)
            logger.info("Validation project loaded and normalized.")

    # Определение размерностей по первому графу
    sample = train_graphs[0]
    version_in = sample["version"].x.size(1)
    package_in = (
        sample["package"].x.size(1) if "package" in sample.node_types else 0
    )
    vuln_in = (
        sample["vulnerability"].x.size(1)
        if "vulnerability" in sample.node_types
        else 0
    )

    # Инициализация модели и оптимизатора
    model = HeteroVulnerabilityGNN(
        version_in,
        package_in,
        vuln_in,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Обучение
    logger.info("Starting training on %d projects...", len(train_graphs))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_pos = 0

        for data in train_graphs:
            data = data.to(device)
            y = data["version"].y.float()
            num_pos = y.sum().item()
            if num_pos == 0:
                continue
            pos_weight = torch.tensor(
                [(y.size(0) - num_pos) / max(num_pos, 1)], device=device
            )
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer.zero_grad()
            try:
                logits = model(data.x_dict, data.edge_index_dict)
            except KeyError as ke:
                # logger.warning("Skipping project due to KeyError: %s. Probably empty graph.", ke)
                continue
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pos += num_pos

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_graphs)
            logger.info(
                "Epoch %3d/%d | avg loss: %.4f | total positive nodes: %d",
                epoch,
                epochs,
                avg_loss,
                total_pos,
            )

        # Оценка на валидации
        if val_graph and epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_data = val_graph.to(device)
                y_val = val_data["version"].y.float()
                logits = model(val_data.x_dict, val_data.edge_index_dict)
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y_val).float().mean().item()
                logger.info("  Validation accuracy: %.4f", acc)

    # Сохранение результатов
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output)
    logger.info("Model saved to %s", model_output)

    stats_path = model_output.parent / "norm_stats.pt"
    torch.save(global_stats, stats_path)
    logger.info("Global normalization stats saved to %s", stats_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Точка входа – парсинг аргументов и запуск обучения на множестве проектов."""
    parser = argparse.ArgumentParser(
        description="Train GNN on multiple projects (inductive pretraining)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Директория с подпапками проектов (в каждой hetero_data.pt).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство для обучения ('cpu' или 'cuda').",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("./pretrained/gnn_model.pt"),
        help="Куда сохранить обученную модель.",
    )
    parser.add_argument(
        "--val-project",
        type=Path,
        default=None,
        help="Путь к папке с одним проектом для оценки качества.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error("Data directory %s not found.", args.data_dir)
        sys.exit(1)

    train_on_multigraphs(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_output=args.output_model,
        val_path=args.val_project,
    )


if __name__ == "__main__":
    main()