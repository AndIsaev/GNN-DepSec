#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Расширенные пулы пакетов (имя, версия) — все версии реальные
# ---------------------------------------------------------------------------
VULNERABLE_POOL: List[Tuple[str, str]] = [
    # Web / HTTP
    ("requests", "2.22.0"), ("requests", "2.25.1"), ("urllib3", "1.25.11"),
    ("urllib3", "1.26.5"), ("aiohttp", "3.8.1"), ("httpx", "0.22.0"),
    # Frameworks
    ("django", "3.2.0"), ("django", "4.0.0"), ("flask", "1.1.2"),
    ("fastapi", "0.65.0"), ("starlette", "0.19.0"), ("werkzeug", "2.0.3"),
    # Data / Science
    ("numpy", "1.16.0"), ("scipy", "1.2.0"), ("pandas", "1.3.0"),
    ("scikit-learn", "1.0.2"), ("matplotlib", "3.5.0"), ("pillow", "9.3.0"),
    # Security / Crypto
    ("cryptography", "36.0.0"), ("pyopenssl", "22.0.0"), ("bcrypt", "3.1.7"),
    # Utils / Others
    ("pyyaml", "5.3.1"), ("lxml", "4.6.3"), ("jinja2", "2.11.3"),
    ("sqlalchemy", "1.3.20"), ("celery", "5.0.5"), ("redis", "4.0.2"),
    ("boto3", "1.17.0"), ("uvicorn", "0.18.0"), ("pytest", "7.0.0"),
    ("wheel", "0.36.0"), ("certifi", "2020.12.5"), ("idna", "2.9"),
    ("more-itertools", "8.10.0"), ("packaging", "21.0"), ("pluggy", "0.13.0"),
    ("py", "1.10.0"), ("attrs", "21.2.0"), ("chardet", "5.0.0"),
    ("virtualenv", "20.4.0"), ("pip", "22.0.4"), ("setuptools-scm", "6.4.2"),
    ("filelock", "3.6.0"), ("distlib", "0.3.4"), ("six", "1.16.0"),
    ("wcwidth", "0.2.5"), ("sqlparse", "0.4.1"), ("tomli", "2.0.1"),
    ("ansible", "2.9.27"),
]

SAFE_POOL: List[Tuple[str, str]] = [
    ("requests", "2.32.4"), ("urllib3", "2.6.0"), ("aiohttp", "3.13.2"),
    ("httpx", "0.28.0"), ("django", "4.2.20"), ("flask", "3.1.0"),
    ("fastapi", "0.115.0"), ("starlette", "0.42.0"), ("werkzeug", "3.0.4"),
    ("numpy", "2.2.0"), ("scipy", "1.15.0"), ("pandas", "2.3.0"),
    ("scikit-learn", "1.7.2"), ("matplotlib", "3.10.0"), ("pillow", "11.0.0"),
    ("cryptography", "44.0.3"), ("pyopenssl", "25.0.0"), ("bcrypt", "4.2.0"),
    ("pyyaml", "6.0.2"), ("lxml", "5.3.0"), ("jinja2", "3.1.5"),
    ("sqlalchemy", "2.0.38"), ("celery", "5.5.0"), ("redis", "5.2.0"),
    ("boto3", "1.38.0"), ("uvicorn", "0.34.0"), ("pytest", "8.3.5"),
    ("wheel", "0.46.2"), ("certifi", "2026.4.22"), ("idna", "3.10"),
    ("more-itertools", "10.8.0"), ("packaging", "26.2"), ("pluggy", "1.6.0"),
    ("py", "1.11.0"), ("attrs", "25.3.0"), ("chardet", "5.2.0"),
    ("virtualenv", "20.30.0"), ("pip", "25.2"), ("setuptools-scm", "8.2.0"),
    ("filelock", "3.18.0"), ("distlib", "0.4.0"), ("six", "1.17.0"),
    ("wcwidth", "0.6.0"), ("sqlparse", "0.5.5"), ("tomli", "2.3.0"),
    ("ansible", "11.8.0"),
]

# ---------------------------------------------------------------------------
# Конфигурация генерации
# ---------------------------------------------------------------------------
DEFAULT_NUM_CRITICAL = 10
DEFAULT_NUM_MIXED = 20
DEFAULT_NUM_SAFE = 70
DEFAULT_MIN_DEPS = 5
DEFAULT_OUTPUT_DIR = Path("datasets/fake_projects")


def generate_requirements(project_id: int, category: str, min_deps: int = DEFAULT_MIN_DEPS) -> str:
    """Генерирует содержимое requirements.txt для синтетического проекта.

    Для каждого идентификатора проекта фиксируется начальное состояние
    генератора случайных чисел, что даёт воспроизводимый набор зависимостей.

    Args:
        project_id: Целочисленный идентификатор проекта (задаёт seed).
        category: Тип проекта: 'critical' – только уязвимые версии,
            'mixed' – смесь уязвимых и безопасных,
            'safe' – только безопасные.
        min_deps: Минимальное количество зависимостей.

    Returns:
        Строка с зависимостями вида `package==version`, по одной на строку.
    """
    random.seed(project_id)
    # Количество зависимостей варьируется от min_deps до min_deps+5
    num_dependencies = min_deps + (project_id % 6)

    if category == "critical":
        chosen = random.sample(VULNERABLE_POOL, k=num_dependencies)
    elif category == "safe":
        chosen = random.sample(SAFE_POOL, k=num_dependencies)
    else:  # mixed
        half = num_dependencies // 2
        vuln_part = random.sample(VULNERABLE_POOL, k=half)
        safe_part = random.sample(SAFE_POOL, k=num_dependencies - len(vuln_part))
        chosen = vuln_part + safe_part
        random.shuffle(chosen)

    lines = [f"{name}=={ver}" for name, ver in chosen]
    return "\n".join(lines)


def main() -> None:
    """Точка входа: создаёт директорию с синтетическими проектами."""
    parser = argparse.ArgumentParser(
        description="Генератор синтетических проектов с уязвимыми и безопасными зависимостями."
    )
    parser.add_argument(
        "--critical",
        type=int,
        default=DEFAULT_NUM_CRITICAL,
        help=f"Количество проектов только с уязвимыми зависимостями (по умолчанию {DEFAULT_NUM_CRITICAL})",
    )
    parser.add_argument(
        "--mixed",
        type=int,
        default=DEFAULT_NUM_MIXED,
        help=f"Количество смешанных проектов (по умолчанию {DEFAULT_NUM_MIXED})",
    )
    parser.add_argument(
        "--safe",
        type=int,
        default=DEFAULT_NUM_SAFE,
        help=f"Количество проектов только с безопасными зависимостями (по умолчанию {DEFAULT_NUM_SAFE})",
    )
    parser.add_argument(
        "--min-deps",
        type=int,
        default=DEFAULT_MIN_DEPS,
        help=f"Минимальное количество зависимостей в проекте (по умолчанию {DEFAULT_MIN_DEPS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Директория для сохранения проектов (по умолчанию {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Глобальное случайное зерно для воспроизводимости порядка категорий",
    )
    args = parser.parse_args()

    # Очистка старых проектов
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Формируем список категорий и перемешиваем для случайного порядка
    categories = (
        ["critical"] * args.critical
        + ["mixed"] * args.mixed
        + ["safe"] * args.safe
    )
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(categories)

    for idx, category in enumerate(categories):
        project_dir = args.output_dir / f"project_{idx:03d}"
        project_dir.mkdir(exist_ok=True)
        req_path = project_dir / "requirements.txt"
        content = generate_requirements(idx, category, args.min_deps)
        req_path.write_text(content)
        num_lines = content.count("\n") + 1
        print(f"Created {req_path} ({category}) with {num_lines} deps")


if __name__ == "__main__":
    main()