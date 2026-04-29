#!/usr/bin/env python3
"""
Graph Builder (FR2) – построение гетерогенного графа зависимостей.

Принимает JSON-файл от Data Collector и строит два представления графа:
  - NetworkX MultiDiGraph (для визуализации и анализа путей),
  - PyTorch Geometric HeteroData (для машинного обучения).

Использование:
    python graph_builder.py --input deps.json --output-dir ./graph_output [--github-token ghp_...]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
import networkx as nx
import torch
from networkx.readwrite import json_graph
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logger = logging.getLogger("graph_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ---------------------------------------------------------------------------
# Взаимодействие с внешними API
# ---------------------------------------------------------------------------

def fetch_github_timestamps(project_urls: dict, auth_headers: dict) -> Dict[str, Optional[str]]:
    """Извлекает ключевые даты из GitHub API для репозитория пакета.

    Args:
        project_urls: Словарь Project-URLs из метаданных PyPI.
        auth_headers: Словарь с заголовком Authorization (если передан токен).

    Returns:
        Словарь с ключами 'created_at', 'updated_at', 'pushed_at'
        (строки ISO-8601 или None).
    """
    default = {'created_at': None, 'updated_at': None, 'pushed_at': None}
    if not project_urls:
        return default
    for url in project_urls.values():
        if isinstance(url, str) and 'github.com' in url:
            parts = url.rstrip('/').split('/')
            if len(parts) >= 2:
                owner, repo = parts[-2], parts[-1]
                if repo.endswith('.git'):
                    repo = repo[:-4]
                try:
                    resp = requests.get(
                        f'https://api.github.com/repos/{owner}/{repo}',
                        timeout=5,
                        headers=auth_headers
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return {
                            'created_at': data.get('created_at'),
                            'updated_at': data.get('updated_at'),
                            'pushed_at': data.get('pushed_at'),
                        }
                    else:
                        logger.debug("GitHub API returned %d for %s/%s", resp.status_code, owner, repo)
                except Exception as e:
                    logger.debug("GitHub request failed for %s/%s: %s", owner, repo, e)
    return default


def fetch_package_metadata(name: str, auth_headers: dict) -> Dict[str, Any]:
    """Получает агрегированные метаданные пакета из PyPI и GitHub.

    Args:
        name: Нормализованное имя пакета.
        auth_headers: Заголовки для GitHub API.

    Returns:
        Словарь с ключами:
          - creation_date: дата первого релиза на PyPI,
          - total_releases: общее число версий,
          - created_at: дата создания репозитория (GitHub),
          - updated_at: дата последнего обновления (GitHub),
          - pushed_at: дата последнего коммита (GitHub),
          - is_deprecated: True, если пакет помечен как устаревший.
    """
    url = f'https://pypi.org/pypi/{name}/json'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            info = data.get('info', {})
            releases = data.get('releases', {})
            first_release_dates = [
                rel.get('upload_time')
                for rels in releases.values()
                for rel in rels
                if rel.get('upload_time')
            ]
            creation_date = min(first_release_dates) if first_release_dates else None
            gh_dates = fetch_github_timestamps(info.get('project_urls', {}), auth_headers)
            return {
                'creation_date': creation_date,
                'total_releases': len(releases),
                'created_at': gh_dates['created_at'],
                'updated_at': gh_dates['updated_at'],
                'pushed_at': gh_dates['pushed_at'],
                'is_deprecated': bool(info.get('deprecated')),
            }
        else:
            logger.warning("PyPI API returned %d for package %s", resp.status_code, name)
    except requests.RequestException as e:
        logger.warning("Failed to fetch package metadata for %s: %s", name, e)
    return {
        "creation_date": None,
        "total_releases": 0,
        "created_at": None,
        "updated_at": None,
        "pushed_at": None,
        "is_deprecated": False,
    }


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------

def build_graph(deps_json: Dict[str, Any], auth_headers: dict) -> Tuple[nx.MultiDiGraph, HeteroData]:
    """Строит гетерогенный граф зависимостей из JSON, полученного от Data Collector.

    Args:
        deps_json: Словарь с ключами 'project_path' и 'packages'.
        auth_headers: Заголовки для GitHub API.

    Returns:
        Кортеж (nx_graph, hetero_data).
    """
    packages = deps_json["packages"]

    nx_graph = nx.MultiDiGraph()
    hetero_data = HeteroData()

    version_features: Dict[str, List[Any]] = {
        "name": [], "version": [], "pushed_at": [],
        "is_root": [],
    }
    package_features: Dict[str, List[Any]] = {
        "name": [], "creation_date": [], "total_releases": [],
        "created_at": [], "updated_at": [], "pushed_at": [], "is_deprecated": [],
    }

    created_packages: Set[str] = set()
    version_idx: Dict[str, int] = {}
    package_idx: Dict[str, int] = {}

    logger.info("Building graph...")

    for pkg in packages:
        name = pkg["name"]
        version = pkg["version"]
        is_root = pkg["is_root"]

        if name not in version_idx:
            version_idx[name] = len(version_idx)
            version_features["name"].append(name)
            version_features["version"].append(version)
            version_features["is_root"].append(is_root)
            # pushed_at будет добавлен позже, пока None
            version_features["pushed_at"].append(None)

            nx_graph.add_node(
                f"version/{name}",
                node_type="version",
                name=name,
                version=version,
                is_root=is_root,
            )

        if name not in created_packages:
            created_packages.add(name)
            package_idx[name] = len(package_idx)
            pkg_meta = fetch_package_metadata(name, auth_headers)
            package_features["name"].append(name)
            package_features["creation_date"].append(pkg_meta["creation_date"])
            package_features["total_releases"].append(pkg_meta["total_releases"])
            package_features["created_at"].append(pkg_meta["created_at"])
            package_features["updated_at"].append(pkg_meta["updated_at"])
            package_features["pushed_at"].append(pkg_meta["pushed_at"])
            package_features["is_deprecated"].append(pkg_meta["is_deprecated"])

            nx_graph.add_node(
                f"package/{name}",
                node_type="package",
                **pkg_meta
            )

    # Перенос pushed_at из узлов Package в узлы Version (NetworkX)
    # и одновременно заполняем version_features["pushed_at"]
    for node, attrs in nx_graph.nodes(data=True):
        if attrs.get("node_type") == "version":
            pkg_name = attrs["name"]
            pkg_node = f"package/{pkg_name}"
            if pkg_node in nx_graph:
                pushed_at = nx_graph.nodes[pkg_node].get("pushed_at")
                if pushed_at:
                    nx_graph.nodes[node]["pushed_at"] = pushed_at
                    # Обновляем признак в version_features по индексу
                    idx = version_idx.get(pkg_name)
                    if idx is not None:
                        version_features["pushed_at"][idx] = pushed_at

    for pkg in packages:
        source_name = pkg["name"]
        source_is_root = pkg["is_root"]

        if source_name in package_idx and source_name in version_idx:
            nx_graph.add_edge(
                f"package/{source_name}",
                f"version/{source_name}",
                type="HAS_VERSION"
            )

        for dep_name in pkg["requires"]:
            if dep_name not in version_idx:
                logger.warning("Dependency '%s' (required by '%s') not found in installed packages, skipping edge.",
                               dep_name, source_name)
                continue
            dep_type = "direct" if source_is_root else "transitive"
            nx_graph.add_edge(
                f"version/{source_name}",
                f"version/{dep_name}",
                type="DEPENDS_ON",
                dep_type=dep_type
            )

    logger.info("Graph built: %d versions, %d packages", len(version_idx), len(package_idx))

    # Формирование тензоров для HeteroData
    version_names = version_features["name"]
    x_version = torch.eye(len(version_names))
    hetero_data['version'].x = x_version
    hetero_data['version'].names = version_names
    # Сохраняем pushed_at как дополнительный атрибут (для последующего использования в GNN)
    hetero_data['version'].pushed_at = version_features["pushed_at"]

    package_names = package_features["name"]
    x_package = torch.eye(len(package_names))
    hetero_data['package'].x = x_package
    hetero_data['package'].names = package_names

    dep_sources, dep_targets = [], []
    for pkg in packages:
        src_name = pkg["name"]
        if src_name not in version_idx:
            continue
        for dep_name in pkg["requires"]:
            if dep_name in version_idx:
                dep_sources.append(version_idx[src_name])
                dep_targets.append(version_idx[dep_name])

    if dep_sources:
        dep_edge_index = torch.tensor([dep_sources, dep_targets], dtype=torch.long)
        hetero_data['version', 'DEPENDS_ON', 'version'].edge_index = dep_edge_index
        logger.info("Added %d DEPENDS_ON edges.", len(dep_sources))

    has_sources, has_targets = [], []
    for pkg_name in created_packages:
        if pkg_name in package_idx and pkg_name in version_idx:
            has_sources.append(package_idx[pkg_name])
            has_targets.append(version_idx[pkg_name])
    if has_sources:
        has_edge_index = torch.tensor([has_sources, has_targets], dtype=torch.long)
        hetero_data['package', 'HAS_VERSION', 'version'].edge_index = has_edge_index
        logger.info("Added %d HAS_VERSION edges.", len(has_sources))

    return nx_graph, hetero_data


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def save_graphs(nx_graph: nx.MultiDiGraph, hetero_data: HeteroData, output_dir: Path) -> None:
    """Сохраняет оба представления графа в файлы."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nx_path = output_dir / "nx_graph.json"
    data = json_graph.node_link_data(nx_graph)
    with open(nx_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("NetworkX graph saved to %s", nx_path)

    het_path = output_dir / "hetero_data.pt"
    torch.save(hetero_data, het_path)
    logger.info("HeteroData saved to %s", het_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Graph Builder (FR2) – построение гетерогенного графа."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Путь к JSON-файлу от Data Collector.")
    parser.add_argument("--output-dir", type=Path, default=Path("./graph_output"),
                        help="Директория для сохранения графов.")
    parser.add_argument("--github-token", type=str, default=os.getenv("GITHUB_TOKEN"),
                        help="GitHub токен (или установите переменную GITHUB_TOKEN).")
    parser.add_argument("--verbose", action="store_true",
                        help="Подробное логирование.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.is_file():
        logger.error("Input file '%s' not found.", args.input)
        sys.exit(1)

    # Подготовка заголовков для GitHub API
    auth_headers = {}
    if args.github_token:
        auth_headers["Authorization"] = f"Bearer {args.github_token}"
        logger.info("Using authenticated GitHub requests.")
    else:
        logger.warning("No GitHub token provided – rate limits may be exceeded quickly.")

    with open(args.input, "r", encoding="utf-8") as f:
        deps = json.load(f)

    nx_graph, hetero_data = build_graph(deps, auth_headers)
    save_graphs(nx_graph, hetero_data, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()