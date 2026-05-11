#!/bin/bash
# Пакетный прогон Python-проектов через Data Collector → Graph Builder → Vulnerability Enricher
# с использованием uv для управления окружениями.

set -euo pipefail
export GITHUB_TOKEN="token"

# Проверяем наличие uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv не найден. Установите его: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# === ОПЦИОНАЛЬНАЯ ГЕНЕРАЦИЯ ФЕЙКОВЫХ ПРОЕКТОВ ===
GENERATE_FAKE="${GENERATE_FAKE:-0}"
if [ "$GENERATE_FAKE" = "1" ]; then
    # Сначала проверяем варианты с префиксом N_, потом без
    N_CRITICAL="${N_CRITICAL:-${CRITICAL:-10}}"
    N_MIXED="${N_MIXED:-${MIXED:-20}}"
    N_SAFE="${N_SAFE:-${SAFE:-70}}"
    MIN_DEPS="${MIN_DEPS:-5}"
    FAKE_SEED="${FAKE_SEED:-${SEED:-}}"
    FAKE_OUT_DIR="${FAKE_OUT_DIR:-./datasets/fake_projects}"

    echo "🏗️  Генерирую синтетические проекты..."
    uv run python app/tools/generate_fake_projects.py \
        --critical "$N_CRITICAL" \
        --mixed "$N_MIXED" \
        --safe "$N_SAFE" \
        --min-deps "$MIN_DEPS" \
        --output-dir "$FAKE_OUT_DIR" \
        $( [ -n "$FAKE_SEED" ] && echo "--seed $FAKE_SEED" )
fi

# === НАСТРОЙКИ ===
PROJECTS_DIR="${PROJECTS_DIR:-${FAKE_OUT_DIR:-./datasets/fake_projects}}"
OUTPUT_DIR="${OUTPUT_DIR:-./datasets/graphs}"
FORCE_RECREATE=0                           # установите в 1, чтобы пересоздавать .venv
echo "📂 Проекты будут взяты из: $PROJECTS_DIR"
echo "📁 Результаты сохранятся в: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Перебираем все поддиректории
for PROJECT in "$PROJECTS_DIR"/*; do
    [ -d "$PROJECT" ] || continue

    PROJECT_NAME=$(basename "$PROJECT")
    echo "========================================"
    echo "📦 Обрабатываю проект: $PROJECT_NAME"
    echo "========================================"

    # 1. Рекурсивный поиск pyproject.toml или requirements.txt (исключая .venv)
    PROJECT_FILE=""
    PROJECT_TYPE=""
    PROJECT_DIR_FOR_ENV="$PROJECT"   # по умолчанию корень

    # Ищем pyproject.toml в любых подпапках
    FOUND_PYPROJECT=$(find "$PROJECT" -not -path '*/.venv/*' -name pyproject.toml -print -quit 2>/dev/null)
    if [ -n "$FOUND_PYPROJECT" ]; then
        PROJECT_FILE="$FOUND_PYPROJECT"
        PROJECT_TYPE="pyproject"
        PROJECT_DIR_FOR_ENV=$(dirname "$FOUND_PYPROJECT")
    else
        # Ищем requirements.txt
        FOUND_REQ=$(find "$PROJECT" -not -path '*/.venv/*' -name requirements.txt -print -quit 2>/dev/null)
        if [ -n "$FOUND_REQ" ]; then
            PROJECT_FILE="$FOUND_REQ"
            PROJECT_TYPE="requirements"
            PROJECT_DIR_FOR_ENV=$(dirname "$FOUND_REQ")
        else
            echo "⚠️  Нет pyproject.toml или requirements.txt в $PROJECT_NAME, пропускаю."
            continue
        fi
    fi
    echo "📄 Найден файл: $PROJECT_FILE (тип: $PROJECT_TYPE)"

    # 2. Виртуальное окружение (создаётся в папке с найденным файлом зависимостей)
    VENV_DIR="$PROJECT_DIR_FOR_ENV/.venv"
    if [ -d "$VENV_DIR" ] && [ $FORCE_RECREATE -eq 0 ]; then
        echo "✅ Виртуальное окружение уже существует: $VENV_DIR"
    else
        echo "🔧 Создаю виртуальное окружение в $VENV_DIR ..."
        if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
        uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
    fi

    # 3. Активация и установка зависимостей
    CURRENT_DIR=$(pwd)
    echo "Текущая директория: $CURRENT_DIR"
    # Активируем venv проекта
    source "$VENV_DIR/bin/activate"

    echo "📥 Устанавливаю зависимости..."
    if [ "$PROJECT_TYPE" = "pyproject" ]; then
        uv sync 2>/dev/null || {
            echo "⚠️  uv sync не удался, пробую резервный метод..."
            uv pip install -e . 2>/dev/null || uv pip install -r "$PROJECT_FILE" 2>/dev/null || true
        }
    else
        # Используем полный путь к файлу зависимостей
        uv pip install -r "$PROJECT_FILE" 2>/dev/null || {
            echo "⚠️  Не удалось установить зависимости из $PROJECT_FILE"
        }
    fi
    deactivate   # Выходим из venv проекта

    # 4. Запуск модулей (из основного виртуального окружения)
    # Активируем основное окружение с предустановленными инструментами
    source "$CURRENT_DIR/.venv/bin/activate"

    PROJECT_OUT="$OUTPUT_DIR/$PROJECT_NAME"
    mkdir -p "$PROJECT_OUT"

    echo "🐍 Data Collector..."
    python "$CURRENT_DIR/app/data_collector/main.py" \
        --project-path "$VENV_DIR/bin/python" \
        --output "$PROJECT_OUT/dependencies.json" || {
        echo "❌ Ошибка Data Collector для $PROJECT_NAME"
        deactivate
        continue
    }

     echo "🔗 Graph Builder..."
     python "$CURRENT_DIR/app/graph_builder/main.py" \
         --input "$PROJECT_OUT/dependencies.json" \
         --output-dir "$PROJECT_OUT" || {
         echo "❌ Ошибка Graph Builder для $PROJECT_NAME"; deactivate; continue
     }

     echo "🛡️ Vulnerability Enricher..."
     python "$CURRENT_DIR/app/vulnerability_enricher/main.py" \
         --nx-input "$PROJECT_OUT/nx_graph.json" \
         --het-input "$PROJECT_OUT/hetero_data.pt" \
         --output-dir "$PROJECT_OUT" || {
         echo "❌ Ошибка Vulnerability Enricher для $PROJECT_NAME"; deactivate; continue
     }

    deactivate   # Выходим из основного venv
    echo "✅ Проект $PROJECT_NAME успешно обработан!"
done

echo "🎉 Все проекты обработаны!"