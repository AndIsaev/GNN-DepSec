#!/usr/bin/env python3
"""
Поиск Python-проектов с requirements.txt в корне и скачивание этих файлов.
Оставляет только непустые requirements.txt с реальными зависимостями.
Требуется: pip install requests
"""

import os
import sys
import time
import json
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict
import functools

# --- Настройки ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BASE_URL = "https://api.github.com"
PER_PAGE = 100
MAX_RESULTS = 1000
OUTPUT_FILE = "python_projects_with_requirements.json"
DOWNLOAD_DIR = "requirements_files"
DOWNLOAD_DELAY = 0.2

# Поиск только файлов requirements.txt в корне
SEARCH_QUERY = "filename:requirements.txt path:/"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def retry(max_retries=5, initial_delay=1, backoff=2, exceptions=(requests.exceptions.RequestException,)):
    """
    Декоратор для повторных попыток выполнения запроса.

    Параметры:
        max_retries   – максимальное количество попыток
        initial_delay – начальная задержка в секундах
        backoff       – множитель задержки при каждой следующей попытке
        exceptions    – кортеж исключений, которые нужно перехватывать
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Успешный ответ от GitHub API не бывает пустым словарём
                    if result == {}:
                        print(f"[РЕТРАЙ] Попытка {attempt}/{max_retries}: пустой ответ, "
                              f"ждём {delay:.1f}с...")
                        time.sleep(delay)
                        delay *= backoff
                        continue
                    # Если ответ не пустой – возвращаем его
                    return result

                except exceptions as e:
                    print(f"[РЕТРАЙ] Попытка {attempt}/{max_retries}: {e}, "
                          f"ждём {delay:.1f}с...")
                    last_exception = e
                    time.sleep(delay)
                    delay *= backoff

            # Если все попытки исчерпаны
            if last_exception:
                raise last_exception
            # Если ни разу не было исключения, но возвращался только {}
            print("[РЕТРАЙ] Все попытки исчерпаны, возвращаю {}")
            return {}

        return wrapper
    return decorator

@retry()
def search_code(query: str, page: int = 1, per_page: int = PER_PAGE) -> dict:
    params = {"q": query, "per_page": per_page, "page": page}
    try:
        resp = requests.get(f"{BASE_URL}/search/code", headers=HEADERS, params=params, timeout=20)
        print(resp.json())
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ОШИБКА] page={page}: {e}", file=sys.stderr)
        return {}


def check_rate_limit() -> Optional[dict]:
    try:
        resp = requests.get(f"{BASE_URL}/rate_limit", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def is_valid_requirements(file_path: Path) -> bool:
    """
    Проверяет, что файл не пуст и содержит хотя бы одну строку,
    похожую на объявление пакета (не комментарий, не пустая).
    """
    if not file_path.exists():
        return False
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return False

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        # Считаем, что строка с зависимостью содержит хотя бы один символ имени пакета
        if re.search(r'[a-zA-Z0-9]', stripped):
            return True
    return False


def download_requirements_files(projects: List[Dict]) -> None:
    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    print(f"\nСкачивание и фильтрация {len(projects)} requirements.txt в '{DOWNLOAD_DIR}/' ...")
    print("Оставляем только непустые файлы с реальными зависимостями.\n")

    success = 0
    skipped = 0
    failed = 0
    filtered_out = 0

    for idx, proj in enumerate(projects, 1):
        full_name = proj.get("full_name", f"unknown_{idx}")
        project_dir_name = full_name.replace("/", "__")
        project_dir = Path(DOWNLOAD_DIR) / project_dir_name
        file_path = project_dir / "requirements.txt"

        # Если файл уже существует, проверяем его валидность
        if file_path.exists():
            if is_valid_requirements(file_path):
                print(f"[{idx}/{len(projects)}] {full_name} – уже есть, OK")
                skipped += 1
            else:
                print(f"[{idx}/{len(projects)}] {full_name} – пустой или невалидный, удаляю")
                file_path.unlink()
                try:
                    project_dir.rmdir()  # удалить папку, если пуста
                except OSError:
                    pass
                filtered_out += 1
            continue

        html_url = proj.get("file_url", "")
        if not html_url:
            print(f"[{idx}/{len(projects)}] {full_name} – нет URL")
            failed += 1
            continue

        raw_url = html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        try:
            resp = requests.get(raw_url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                project_dir.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(resp.content)

                if is_valid_requirements(file_path):
                    print(f"[{idx}/{len(projects)}] {full_name} – скачан ✓")
                    success += 1
                else:
                    print(f"[{idx}/{len(projects)}] {full_name} – пустой файл, удалён")
                    file_path.unlink()
                    try:
                        project_dir.rmdir()
                    except OSError:
                        pass
                    filtered_out += 1
            elif resp.status_code == 404:
                print(f"[{idx}/{len(projects)}] {full_name} – 404")
                skipped += 1
            elif resp.status_code in (403, 429):
                print(f"[{idx}/{len(projects)}] {full_name} – {resp.status_code}")
                failed += 1
            else:
                print(f"[{idx}/{len(projects)}] {full_name} – ошибка {resp.status_code}")
                failed += 1
        except requests.RequestException as e:
            print(f"[{idx}/{len(projects)}] {full_name} – сеть: {e}")
            failed += 1

        time.sleep(DOWNLOAD_DELAY)

    print(f"\nИтог: успешно {success}, пропущено (уже было) {skipped}, ошибок {failed}, отфильтровано пустых {filtered_out}")


def main() -> None:
    print(f"Поиск: {SEARCH_QUERY}")
    print("=" * 60)

    projects: List[Dict] = []
    total_count = None
    page = 1

    while len(projects) < MAX_RESULTS:
        print(f"Страница {page}...", end=" ")
        data = search_code(SEARCH_QUERY, page=page)

        if not data:
            print("стоп.")
            break

        items = data.get("items", [])
        print(f"{len(items)} элементов.")
        if not items:
            break

        if total_count is None:
            total_count = data.get("total_count", 0)
            print(f"Всего на GitHub: {total_count}")

        for item in items:
            repo = item.get("repository", {})
            projects.append({
                "full_name": repo.get("full_name"),
                "html_url": repo.get("html_url"),
                "description": repo.get("description"),
                "stars": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "requirements_path": item.get("path"),
                "file_url": item.get("html_url"),
            })
            if len(projects) >= MAX_RESULTS:
                break

        if total_count and page * PER_PAGE >= total_count:
            break

        page += 1
        time.sleep(2)

    projects = projects[:MAX_RESULTS]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Найдено проектов с requirements.txt: {len(projects)} (сохранено в {OUTPUT_FILE})")

    rl = check_rate_limit()
    if rl:
        sr = rl.get("resources", {}).get("search", {})
        print(f"Лимит запросов: {sr.get('remaining')}/{sr.get('limit')} (сброс {time.ctime(sr.get('reset', 0))})")

    download_requirements_files(projects)


if __name__ == "__main__":
    main()