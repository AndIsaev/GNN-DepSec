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
import re
import sys
from datetime import datetime, timezone
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
# Вспомогательные функции для признаков
# ---------------------------------------------------------------------------

def parse_version(ver_str: str) -> Tuple[int, int, int]:
    """Извлекает major, minor, patch из строки версии PEP 440.
    Для простоты обрезаем до трёх чисел, остальное игнорируем.
    Возвращает 0 для отсутствующих частей.
    """
    parts = re.split(r'[.\-]', ver_str)
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
        else:
            break
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def is_prerelease(ver_str: str) -> bool:
    """True, если версия содержит a, b, rc, dev, pre и т.п."""
    return bool(re.search(r'[a-zA-Z]', ver_str))


# ---------------------------------------------------------------------------
# Взаимодействие с внешними API
# ---------------------------------------------------------------------------

def fetch_github_timestamps(project_urls: dict, auth_headers: dict) -> Dict[str, Optional[str]]:
    """Извлекает ключевые даты из GitHub API для репозитория пакета."""
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
    """Получает агрегированные метаданные пакета из PyPI и GitHub,
    а также информацию о всех версиях (даты и флаги пререлизов).

    Returns:
        Словарь с ключами:
          - creation_date, total_releases, created_at, updated_at, pushed_at, is_deprecated,
          - releases_info: словарь {version: {'upload_time': str или None, 'is_prerelease': bool}}
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

            # Собираем информацию по каждой версии
            releases_info = {}
            for ver, files in releases.items():
                upload_time = None
                if files:
                    upload_time = files[0].get('upload_time')
                releases_info[ver] = {
                    'upload_time': upload_time,
                    'is_prerelease': is_prerelease(ver)
                }

            return {
                'creation_date': creation_date,
                'total_releases': len(releases),
                'created_at': gh_dates['created_at'],
                'updated_at': gh_dates['updated_at'],
                'pushed_at': gh_dates['pushed_at'],
                'is_deprecated': bool(info.get('deprecated')),
                'releases_info': releases_info
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
        "releases_info": {}
    }


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------

def build_graph(deps_json: Dict[str, Any], auth_headers: dict) -> Tuple[nx.MultiDiGraph, HeteroData]:
    """Строит гетерогенный граф зависимостей из JSON, полученного от Data Collector."""
    packages = deps_json["packages"]

    nx_graph = nx.MultiDiGraph()
    hetero_data = HeteroData()

    # Временные структуры для сбора информации об узлах
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
    # Словарь для сохранения полных метаданных пакетов (включая releases_info)
    package_meta_dict: Dict[str, Dict[str, Any]] = {}

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
            version_features["pushed_at"].append(None)  # заполнится позже

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
            package_meta_dict[name] = pkg_meta  # сохраняем для признаков
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
                **{k: v for k, v in pkg_meta.items() if k != "releases_info"}
            )

    # Перенос pushed_at из узлов Package в узлы Version (NetworkX)
    for node, attrs in nx_graph.nodes(data=True):
        if attrs.get("node_type") == "version":
            pkg_name = attrs["name"]
            pkg_node = f"package/{pkg_name}"
            if pkg_node in nx_graph:
                pushed_at = nx_graph.nodes[pkg_node].get("pushed_at")
                if pushed_at:
                    nx_graph.nodes[node]["pushed_at"] = pushed_at
                    idx = version_idx.get(pkg_name)
                    if idx is not None:
                        version_features["pushed_at"][idx] = pushed_at

    # Добавляем рёбра HAS_VERSION и DEPENDS_ON
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

    # ---------- ФОРМИРОВАНИЕ ПРИЗНАКОВЫХ ТЕНЗОРОВ ----------
    version_names = version_features["name"]
    # Признаки Version: major, minor, patch, is_prerelease, release_timestamp, is_root, pkg_pushed_at
    version_feat_list = []
    for i, name in enumerate(version_names):
        ver_str = version_features["version"][i]
        major, minor, patch = parse_version(ver_str)
        pre = 1.0 if is_prerelease(ver_str) else 0.0

        # Дата релиза конкретной версии
        rinfo = None
        if name in package_meta_dict:
            rinfo = package_meta_dict[name].get('releases_info', {}).get(ver_str)
        if rinfo and rinfo['upload_time']:
            dt = datetime.fromisoformat(rinfo['upload_time'].replace('Z', '+00:00'))
            release_ts = dt.timestamp() / 1e9  # нормализация грубая, позже сделаем StandardScaler
        else:
            release_ts = 0.0

        is_root_val = 1.0 if version_features["is_root"][i] else 0.0

        # pushed_at пакета (уже есть в version_features["pushed_at"] либо None)
        pkg_pushed_at_str = version_features["pushed_at"][i]
        if pkg_pushed_at_str:
            dt = datetime.fromisoformat(pkg_pushed_at_str.replace('Z', '+00:00'))
            pkg_pushed_ts = dt.timestamp() / 1e9
        else:
            pkg_pushed_ts = 0.0

        version_feat_list.append([major, minor, patch, pre, release_ts, is_root_val, pkg_pushed_ts])

    x_version = torch.tensor(version_feat_list, dtype=torch.float)
    hetero_data['version'].x = x_version
    hetero_data['version'].names = version_names

    # Признаки Package: total_releases, creation_timestamp, pushed_at_timestamp, is_deprecated
    package_names = package_features["name"]
    package_feat_list = []
    for i, name in enumerate(package_names):
        meta = package_meta_dict.get(name, {})
        total_rel = float(meta.get('total_releases', 0))
        cr_date = meta.get('creation_date')
        if cr_date:
            dt = datetime.fromisoformat(cr_date.replace('Z', '+00:00'))
            cr_ts = dt.timestamp() / 1e9
        else:
            cr_ts = 0.0
        push_date = meta.get('pushed_at')
        if push_date:
            dt = datetime.fromisoformat(push_date.replace('Z', '+00:00'))
            push_ts = dt.timestamp() / 1e9
        else:
            push_ts = 0.0

        is_depr = 1.0 if meta.get('is_deprecated') else 0.0
        package_feat_list.append([total_rel, cr_ts, push_ts, is_depr])

    x_package = torch.tensor(package_feat_list, dtype=torch.float)
    hetero_data['package'].x = x_package
    hetero_data['package'].names = package_names

    # Рёбра DEPENDS_ON в HeteroData
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

    # Рёбра HAS_VERSION
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