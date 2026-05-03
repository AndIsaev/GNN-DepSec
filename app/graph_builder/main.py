#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
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
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Коэффициент нормализации временных меток (секунды → «миллиардные доли»)
TIMESTAMP_SCALE = 1e9

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def parse_version(ver_str: str) -> Tuple[int, int, int]:
    """Возвращает кортеж (major, minor, patch) из строки версии PEP 440.

    Если какая-либо часть отсутствует, подставляется 0.

    Args:
        ver_str: Строка версии, например "2.4.1" или "1.0a2".

    Returns:
        Кортеж из трёх целых чисел.
    """
    parts = re.split(r"[.\-]", ver_str)
    numbers = []
    for p in parts:
        if p.isdigit():
            numbers.append(int(p))
        else:
            break
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def is_prerelease(ver_str: str) -> bool:
    """Проверяет, является ли версия предварительным релизом.

    Args:
        ver_str: Строка версии.

    Returns:
        True, если версия содержит буквенные маркеры (a, b, rc, dev и т.п.).
    """
    return bool(re.search(r"[a-zA-Z]", ver_str))


def iso_to_timestamp(iso_str: Optional[str]) -> float:
    """Преобразует ISO-строку даты в нормализованную временную метку.

    Args:
        iso_str: Строка в формате ISO 8601, возможно с 'Z' на конце.

    Returns:
        Число секунд с начала эпохи, делённое на 1e9, или 0.0, если
        строка отсутствует или не может быть разобрана.
    """
    if not iso_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp() / TIMESTAMP_SCALE
    except (ValueError, TypeError):
        logger.debug("Failed to parse date '%s'", iso_str)
        return 0.0


# ---------------------------------------------------------------------------
# Асинхронные запросы к внешним API
# ---------------------------------------------------------------------------

async def fetch_github_timestamps_async(
    session: aiohttp.ClientSession,
    project_urls: Dict[str, str],
    auth_headers: Dict[str, str],
) -> Dict[str, Optional[str]]:
    """Асинхронно получает ключевые даты репозитория из GitHub API.

    Ищет в словаре project_urls URL, содержащий 'github.com', и делает
    запрос к API.

    Args:
        session: Асинхронная сессия aiohttp.
        project_urls: Словарь с URL проекта (ключи – названия, значения – ссылки).
        auth_headers: Заголовки для авторизации (может содержать Bearer токен).

    Returns:
        Словарь с ключами 'created_at', 'updated_at', 'pushed_at'.
        Если данные получить не удалось, все значения – None.
    """
    default = {"created_at": None, "updated_at": None, "pushed_at": None}
    if not project_urls:
        return default

    for url in project_urls.values():
        if not isinstance(url, str) or "github.com" not in url:
            continue
        parts = url.rstrip("/").split("/")
        if len(parts) < 2:
            continue
        owner, repo = parts[-2], parts[-1]
        if repo.endswith(".git"):
            repo = repo[:-4]

        try:
            async with session.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=auth_headers,
                timeout=5,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "pushed_at": data.get("pushed_at"),
                    }
                logger.debug(
                    "GitHub API returned %d for %s/%s", resp.status, owner, repo
                )
        except Exception as e:
            logger.debug("GitHub request failed for %s/%s: %s", owner, repo, e)

    return default


async def fetch_package_metadata_async(
    session: aiohttp.ClientSession,
    name: str,
    auth_headers: Dict[str, str],
) -> Dict[str, Any]:
    """Асинхронно собирает агрегированные метаданные пакета из PyPI и GitHub.

    Для каждого пакета получает общую информацию с PyPI, даты с GitHub,
    а также сведения о всех доступных версиях (дата загрузки и признак пререлиза).

    Args:
        session: Асинхронная сессия aiohttp.
        name: Имя пакета в PyPI.
        auth_headers: Заголовки авторизации для GitHub API.

    Returns:
        Словарь с ключами:
            creation_date, total_releases, created_at, updated_at,
            pushed_at, is_deprecated, releases_info.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                info = data.get("info", {})
                releases = data.get("releases", {})

                first_release_dates = [
                    rel.get("upload_time")
                    for files in releases.values()
                    for rel in files
                    if rel.get("upload_time")
                ]
                creation_date = min(first_release_dates) if first_release_dates else None

                gh_dates = await fetch_github_timestamps_async(
                    session, info.get("project_urls", {}), auth_headers
                )

                releases_info: Dict[str, Dict[str, Any]] = {}
                for ver, files in releases.items():
                    upload_time = files[0].get("upload_time") if files else None
                    releases_info[ver] = {
                        "upload_time": upload_time,
                        "is_prerelease": is_prerelease(ver),
                    }

                return {
                    "creation_date": creation_date,
                    "total_releases": len(releases),
                    "created_at": gh_dates.get("created_at"),
                    "updated_at": gh_dates.get("updated_at"),
                    "pushed_at": gh_dates.get("pushed_at"),
                    "is_deprecated": bool(info.get("deprecated")),
                    "releases_info": releases_info,
                }
            logger.warning("PyPI API returned %d for package %s", resp.status, name)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning("Failed to fetch package metadata for %s: %s", name, e)

    return {
        "creation_date": None,
        "total_releases": 0,
        "created_at": None,
        "updated_at": None,
        "pushed_at": None,
        "is_deprecated": False,
        "releases_info": {},
    }


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------

def build_graph(
    deps_json: Dict[str, Any],
    package_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[nx.MultiDiGraph, HeteroData]:
    """Строит гетерогенный граф зависимостей из данных о пакетах и метаданных.

    Создаёт два представления:
      - NetworkX MultiDiGraph с узлами типов 'version' и 'package',
        рёбрами 'HAS_VERSION' и 'DEPENDS_ON'.
      - PyTorch Geometric HeteroData с аналогичной структурой и
        числовыми признаками для каждого типа узлов.

    Args:
        deps_json: Словарь, загруженный из JSON-файла Data Collector.
            Ожидается ключ 'packages' со списком установленных пакетов.
        package_metadata: Словарь, сопоставляющий имени пакета его
            метаданные (результат fetch_package_metadata_async).

    Returns:
        Кортеж из двух элементов: NetworkX MultiDiGraph и HeteroData.
    """
    packages: List[Dict[str, Any]] = deps_json["packages"]
    unique_names: List[str] = sorted(package_metadata.keys())

    nx_graph = nx.MultiDiGraph()
    hetero_data = HeteroData()

    # -------- Версии (узлы типа 'version') и их признаки --------
    version_idx: Dict[str, int] = {}
    version_names: List[str] = []
    version_features: List[List[float]] = []

    for pkg in packages:
        name = pkg["name"]
        if name in version_idx:
            continue
        version_idx[name] = len(version_names)
        version_names.append(name)

        ver_str: str = pkg["version"]
        major, minor, patch = parse_version(ver_str)
        pre_flag = 1.0 if is_prerelease(ver_str) else 0.0

        # Дата загрузки конкретной версии
        rinfo = package_metadata.get(name, {}).get("releases_info", {}).get(ver_str, {})
        release_timestamp = iso_to_timestamp(rinfo.get("upload_time"))

        is_root_val = 1.0 if pkg["is_root"] else 0.0
        pkg_pushed_ts = iso_to_timestamp(package_metadata.get(name, {}).get("pushed_at"))

        version_features.append(
            [major, minor, patch, pre_flag, release_timestamp, is_root_val, pkg_pushed_ts]
        )

        nx_graph.add_node(
            f"version/{name}",
            node_type="version",
            name=name,
            version=ver_str,
            is_root=is_root_val,
            pushed_at=package_metadata.get(name, {}).get("pushed_at"),
        )

    # -------- Пакеты (узлы типа 'package') и их признаки --------
    package_idx: Dict[str, int] = {}
    package_names: List[str] = []
    package_features: List[List[float]] = []

    for name in unique_names:
        package_idx[name] = len(package_names)
        package_names.append(name)
        meta = package_metadata.get(name, {})

        total_rel = float(meta.get("total_releases", 0))
        cr_ts = iso_to_timestamp(meta.get("creation_date"))
        push_ts = iso_to_timestamp(meta.get("pushed_at"))
        is_depr = 1.0 if meta.get("is_deprecated") else 0.0

        package_features.append([total_rel, cr_ts, push_ts, is_depr])

        nx_graph.add_node(
            f"package/{name}",
            node_type="package",
            name=name,
            creation_date=meta.get("creation_date"),
            total_releases=total_rel,
            created_at=meta.get("created_at"),
            updated_at=meta.get("updated_at"),
            pushed_at=meta.get("pushed_at"),
            is_deprecated=is_depr,
        )

    # -------- Рёбра HAS_VERSION --------
    for name in unique_names:
        if name in version_idx:
            nx_graph.add_edge(
                f"package/{name}", f"version/{name}", type="HAS_VERSION"
            )

    # -------- Рёбра DEPENDS_ON --------
    dep_sources: List[int] = []
    dep_targets: List[int] = []

    for pkg in packages:
        src_name = pkg["name"]
        src_is_root = pkg["is_root"]
        if src_name not in version_idx:
            continue
        for dep_name in pkg["requires"]:
            if dep_name not in version_idx:
                logger.warning(
                    "Dependency '%s' (required by '%s') not found in installed packages, skipping.",
                    dep_name,
                    src_name,
                )
                continue
            dep_type = "direct" if src_is_root else "transitive"
            nx_graph.add_edge(
                f"version/{src_name}",
                f"version/{dep_name}",
                type="DEPENDS_ON",
                dep_type=dep_type,
            )
            dep_sources.append(version_idx[src_name])
            dep_targets.append(version_idx[dep_name])

    logger.info(
        "Graph built: %d versions, %d packages, %d DEPENDS_ON edges",
        len(version_names),
        len(package_names),
        len(dep_sources),
    )

    # -------- Тензоры для HeteroData --------
    hetero_data["version"].x = torch.tensor(version_features, dtype=torch.float)
    hetero_data["version"].names = version_names

    hetero_data["package"].x = torch.tensor(package_features, dtype=torch.float)
    hetero_data["package"].names = package_names

    if dep_sources:
        edge_index = torch.tensor([dep_sources, dep_targets], dtype=torch.long)
        hetero_data["version", "DEPENDS_ON", "version"].edge_index = edge_index

    has_sources, has_targets = [], []
    for name in unique_names:
        if name in version_idx:
            has_sources.append(package_idx[name])
            has_targets.append(version_idx[name])
    if has_sources:
        hetero_data["package", "HAS_VERSION", "version"].edge_index = torch.tensor(
            [has_sources, has_targets], dtype=torch.long
        )

    return nx_graph, hetero_data


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def save_graphs(
    nx_graph: nx.MultiDiGraph, hetero_data: HeteroData, output_dir: Path
) -> None:
    """Сохраняет оба представления графа в файлы.

    Args:
        nx_graph: NetworkX граф.
        hetero_data: PyTorch Geometric HeteroData.
        output_dir: Путь к директории для сохранения.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nx_path = output_dir / "nx_graph.json"
    with open(nx_path, "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(nx_graph), f, indent=2, ensure_ascii=False)
    logger.info("NetworkX graph saved to %s", nx_path)

    het_path = output_dir / "hetero_data.pt"
    torch.save(hetero_data, het_path)
    logger.info("HeteroData saved to %s", het_path)


# ---------------------------------------------------------------------------
# Асинхронная обвязка
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    """Основная асинхронная логика: загрузка метаданных и построение графа.

    Args:
        args: Аргументы командной строки после парсинга.
    """
    auth_headers: Dict[str, str] = {}
    if args.github_token:
        auth_headers["Authorization"] = f"Bearer {args.github_token}"
        logger.info("Using authenticated GitHub requests.")
    else:
        logger.warning(
            "No GitHub token provided – rate limits may be exceeded quickly."
        )

    with open(args.input, "r", encoding="utf-8") as f:
        deps = json.load(f)

    packages = deps["packages"]
    unique_names = sorted({pkg["name"] for pkg in packages})
    logger.info(
        "Fetching metadata for %d unique packages asynchronously...",
        len(unique_names),
    )

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_package_metadata_async(session, name, auth_headers)
            for name in unique_names
        ]
        results = await asyncio.gather(*tasks)

    package_metadata = dict(zip(unique_names, results))
    logger.info("Metadata fetched for %d packages.", len(package_metadata))

    nx_graph, hetero_data = build_graph(deps, package_metadata)
    save_graphs(nx_graph, hetero_data, args.output_dir)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа в приложение.

    Парсит аргументы командной строки, настраивает логирование и запускает
    асинхронный рабочий процесс.
    """
    parser = argparse.ArgumentParser(
        description="Graph Builder (FR2) – построение гетерогенного графа."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Путь к JSON-файлу от Data Collector.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./graph_output"),
        help="Директория для сохранения графов.",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub токен (или установите переменную GITHUB_TOKEN).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Подробное логирование."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.is_file():
        logger.error("Input file '%s' not found.", args.input)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()