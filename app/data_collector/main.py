#!/usr/bin/env python3
"""
Data Collector (FR1) – сбор списка пакетов и их зависимостей
с использованием pipdeptree.

Работает только в активном виртуальном окружении.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("data_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

@dataclass
class PackageInfo:
    """Информация об одном установленном пакете.

    Attributes:
        name: Нормализованное имя пакета (PEP 503).
        version: Установленная версия пакета.
        requires: Список имён пакетов, от которых зависит данный (может быть пустым).
        is_root: True, если пакет является корневой зависимостью проекта.
    """
    name: str
    version: str
    requires: List[str] = field(default_factory=list)
    is_root: bool = False


@dataclass
class CollectorOutput:
    """Результат сбора зависимостей.

    Attributes:
        project_path: Абсолютный путь к проекту, в окружении которого выполняется сбор.
        packages: Список собранных пакетов.
    """
    project_path: str
    packages: List[PackageInfo]


def normalize_name(name: str) -> str:
    """Приводит имя пакета к каноническому виду согласно PEP 503.

    Заменяет точки, подчёркивания и дефисы на одиночные дефисы,
    приводит к нижнему регистру.

    Args:
        name: Исходное имя пакета.

    Returns:
        Нормализованное имя.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def find_venv_python(project_root: Path) -> Optional[Path]:
    """Ищет интерпретатор Python внутри стандартных директорий виртуального окружения.

    Проверяет наличие python (или python.exe) в `venv/bin`, `.venv/bin`,
    `venv/Scripts`, `.venv/Scripts`.

    Args:
        project_root: Корневая директория проекта.

    Returns:
        Путь к найденному интерпретатору либо None, если окружение не обнаружено.
    """
    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / "venv" / "bin" / "python",
        project_root / "venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Сбор зависимостей
# ---------------------------------------------------------------------------

def collect_dependencies() -> CollectorOutput:
    """Собирает информацию о зависимостях через ``pipdeptree --json-tree``.

    Рекурсивно обходит дерево, полученное от pipdeptree, выделяя корневые
    пакеты (is_root=True) и транзитивные зависимости (is_root=False).

    Returns:
        Объект CollectorOutput с заполненным списком пакетов.
        В случае ошибки возвращается CollectorOutput с пустым списком пакетов.
    """
    logger.info("Collecting dependencies via pipdeptree...")
    try:
        result = subprocess.run(
            ["pipdeptree", "--json-tree"],
            capture_output=True,
            text=True,
            check=True
        )
        tree = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("pipdeptree failed: %s", e.stderr)
        return CollectorOutput(project_path=str(Path.cwd().resolve()), packages=[])
    except FileNotFoundError:
        logger.error("pipdeptree not found. Install it with 'pip install pipdeptree'.")
        return CollectorOutput(project_path=str(Path.cwd().resolve()), packages=[])

    packages: List[PackageInfo] = []

    def walk(node: dict, is_root: bool) -> None:
        """Рекурсивно обходит дерево pipdeptree и добавляет пакеты в список.

        Args:
            node: Узел дерева (словарь с ключами key, installed_version, dependencies).
            is_root: Является ли текущий пакет корневым.
        """
        name = normalize_name(node["key"])
        version = node["installed_version"]
        deps = [normalize_name(dep["key"]) for dep in node.get("dependencies", [])]
        packages.append(PackageInfo(
            name=name,
            version=version,
            requires=sorted(set(deps)),
            is_root=is_root
        ))
        for dep in node.get("dependencies", []):
            walk(dep, is_root=False)

    for entry in tree:
        if entry.get("key") == "pipdeptree":
            continue  # пропускаем инструментальный пакет и все его зависимости
        walk(entry, is_root=True)

    packages.sort(key=lambda p: p.name)
    logger.info("Collected %d packages via pipdeptree.", len(packages))
    return CollectorOutput(
        project_path=str(Path.cwd().resolve()),
        packages=packages
    )


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_json(output: CollectorOutput, output_path: Path) -> None:
    """Сохраняет результат сбора в JSON-файл.

    Использует ``dataclasses.asdict`` для сериализации структуры CollectorOutput.

    Args:
        output: Объект с собранными пакетами.
        output_path: Путь для сохранения файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(output), f, indent=2, ensure_ascii=False)
    logger.info("Data saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Data Collector – извлекает список пакетов и их зависимости."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dependencies.json"),
        help="Путь для сохранения JSON-файла с результатами."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Включить подробное логирование."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if sys.prefix == sys.base_prefix:
        logger.warning("Script is not running inside a virtual environment.")
        venv_python = find_venv_python(Path.cwd())
        if venv_python:
            logger.info("Found Python in %s. Please run inside the venv.", venv_python)
        else:
            logger.error("No virtual environment found.")
        sys.exit(1)

    output = collect_dependencies()
    if not output.packages:
        logger.error("No packages collected. Exiting.")
        sys.exit(1)
    save_json(output, args.output)


if __name__ == "__main__":
    main()