#!/usr/bin/env python3
"""
Поиск до 5000 Python-репозиториев с requirements.txt через разбивку по звёздам.
Требуется: pip install requests
"""

import os
import time
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import quote

# --- Настройки ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BASE_URL = "https://api.github.com"
PER_PAGE = 100
TARGET_REPOS_WITH_FILE = 5000
MAX_REPOS_TO_CHECK = 10000
OUTPUT_JSON = "python_projects_with_requirements.json"
DOWNLOAD_DIR = "requirements_files"
DOWNLOAD_DELAY = 0.2
REPO_CHECK_DELAY = 0.1
SEARCH_PAGE_DELAY = 10

# Базовый запрос без звёзд
BASE_REPO_QUERY = "requirements.txt language:python"

HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def search_repos(query: str, page: int = 1) -> dict:
    """Поиск репозиториев."""
    params = {"q": query, "per_page": PER_PAGE, "page": page}
    resp = requests.get(f"{BASE_URL}/search/repositories", headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def check_file_exists(owner: str, repo: str, path: str = "requirements.txt") -> bool:
    """Быстрая проверка наличия файла в корне."""
    url = f"{BASE_URL}/repos/{owner}/{repo}/contents/{quote(path)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def download_requirements(owner: str, repo: str, dest_dir: Path) -> bool:
    """Скачивает файл в папку owner__repo и проверяет, не пуст ли он."""
    for branch in ("main", "master"):
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/requirements.txt"
        try:
            resp = requests.get(raw_url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                project_dir = dest_dir / f"{owner}__{repo}"
                project_dir.mkdir(parents=True, exist_ok=True)
                file_path = project_dir / "requirements.txt"
                file_path.write_bytes(resp.content)

                content = file_path.read_text(encoding="utf-8")
                has_pkg = any(
                    stripped and not stripped.startswith('#') and re.search(r'[a-zA-Z0-9]', stripped)
                    for stripped in (line.strip() for line in content.splitlines())
                )
                if has_pkg:
                    return True
                else:
                    file_path.unlink()
                    try:
                        project_dir.rmdir()
                    except OSError:
                        pass
        except Exception:
            continue
    return False


def generate_star_ranges(min_stars: int = 0, max_stars: int = 50000,
                         step: int = 200) -> List[Tuple[int, int]]:
    """Создаёт список диапазонов звёзд: (0..200, 200..400, ...)."""
    ranges = []
    low = min_stars
    while low < max_stars:
        high = low + step
        ranges.append((low, high))
        low = high
    return ranges


def collect_repos_with_requirements(ranges: List[Tuple[int, int]]) -> List[Dict]:
    """Перебирает диапазоны звёзд, собирает репозитории с requirements.txt."""
    collected = []
    seen = set()

    for low, high in ranges:
        if len(collected) >= TARGET_REPOS_WITH_FILE:
            break

        query = f"{BASE_REPO_QUERY} stars:{low}..{high}"
        print(f"\n=== Диапазон звёзд {low}..{high} ===")
        print(f"Запрос: {query}")

        page = 1
        while len(collected) < TARGET_REPOS_WITH_FILE:
            try:
                data = search_repos(query, page=page)
            except requests.exceptions.RequestException as e:
                print(f"Ошибка поиска: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            print(f"  Страница {page}: {len(items)} репозиториев")

            for repo in items:
                owner = repo["owner"]["login"]
                name = repo["name"]
                full_name = f"{owner}/{name}"
                if full_name in seen:
                    continue
                seen.add(full_name)

                # Проверяем наличие requirements.txt в корне
                if check_file_exists(owner, name):
                    collected.append({
                        "full_name": full_name,
                        "html_url": repo["html_url"],
                        "description": repo.get("description"),
                        "stars": repo["stargazers_count"],
                        "language": repo.get("language"),
                    })
                    print(f"    ✓ {full_name} (всего {len(collected)})")
                    if len(collected) >= TARGET_REPOS_WITH_FILE:
                        break
                time.sleep(REPO_CHECK_DELAY)

            if data.get("total_count", 0) <= page * PER_PAGE:
                break
            page += 1
            time.sleep(SEARCH_PAGE_DELAY)

    return collected[:TARGET_REPOS_WITH_FILE]


def main():
    print(f"Цель: найти {TARGET_REPOS_WITH_FILE} репозиториев с requirements.txt")
    print("Генерация диапазонов звёзд (0..50000 с шагом 200)...")
    star_ranges = generate_star_ranges(min_stars=0, max_stars=50000, step=200)
    print(f"Будет проверено {len(star_ranges)} диапазонов.")

    projects = collect_repos_with_requirements(star_ranges)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Найдено проектов: {len(projects)} (сохранено в {OUTPUT_JSON})")

    # Скачивание файлов
    download_dir = Path(DOWNLOAD_DIR)
    download_dir.mkdir(exist_ok=True)
    success = 0
    for idx, proj in enumerate(projects, 1):
        owner, repo = proj["full_name"].split("/")
        if download_requirements(owner, repo, download_dir):
            print(f"[{idx}/{len(projects)}] {proj['full_name']} – скачан ✓")
            success += 1
        else:
            print(f"[{idx}/{len(projects)}] {proj['full_name']} – ошибка или пустой")
        time.sleep(DOWNLOAD_DELAY)

    print(f"Скачивание завершено: успешно {success} из {len(projects)}")


if __name__ == "__main__":
    main()