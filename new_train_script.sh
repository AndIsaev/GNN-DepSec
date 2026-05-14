#!/bin/bash
# Пакетный прогон Python-проектов через Data Collector → Graph Builder → Vulnerability Enricher
# с использованием uv для управления окружениями.
# После обработки каждого проекта виртуальное окружение удаляется.

set -euo pipefail
export GITHUB_TOKEN="${GITHUB_TOKEN:-token}"

# Проверяем наличие uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv не найден. Установите его: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# === ОПЦИОНАЛЬНАЯ ГЕНЕРАЦИЯ ФЕЙКОВЫХ ПРОЕКТОВ ===
GENERATE_FAKE="${GENERATE_FAKE:-0}"
if [ "$GENERATE_FAKE" = "1" ]; then
    N_CRITICAL="${N_CRITICAL:-${CRITICAL:-10}}"
    N_MIXED="${N_MIXED:-${MIXED:-20}}"
    N_SAFE="${N_SAFE:-${SAFE:-70}}"
    MIN_DEPS="${MIN_DEPS:-5}"
    FAKE_SEED="${FAKE_SEED:-${SEED:-}}"
    FAKE_OUT_DIR="${FAKE_OUT_DIR:-./datasets/fake_projects}"

    echo "🏗️  Генерирую синтетические проекты..."
    python app/tools/generate_fake_projects.py \
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
FORCE_RECREATE="${FORCE_RECREATE:-0}"   # 1 – пересоздавать .venv, даже если существует
CLEAN_VENV="${CLEAN_VENV:-1}"           # 1 – удалять .venv после обработки (по умолчанию)
FAILED_LOG="${FAILED_LOG:-./datasets/failed_projects.log}"

echo "📂 Проекты будут взяты из: $PROJECTS_DIR"
echo "📁 Результаты сохранятся в: $OUTPUT_DIR"
echo "🧹 Удаление venv после обработки: $( [ "$CLEAN_VENV" = "1" ] && echo 'ДА' || echo 'НЕТ' )"
mkdir -p "$OUTPUT_DIR"

# Счётчики
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL=0

# Вспомогательная функция удаления venv проекта
cleanup_project_venv() {
    local venv_dir="$1"
    if [ "$CLEAN_VENV" = "1" ] && [ -d "$venv_dir" ]; then
        echo "🧹 Удаляю виртуальное окружение проекта: $venv_dir"
        rm -rf "$venv_dir"
    fi
}

# Основной цикл
for PROJECT in "$PROJECTS_DIR"/*; do
    [ -d "$PROJECT" ] || continue
    TOTAL=$((TOTAL + 1))
    PROJECT_NAME=$(basename "$PROJECT")
    echo "========================================"
    echo "📦 Обрабатываю проект: $PROJECT_NAME"
    echo "========================================"

    # 1. Поиск файла зависимостей
    PROJECT_FILE=""
    PROJECT_TYPE=""
    PROJECT_DIR_FOR_ENV="$PROJECT"

    FOUND_PYPROJECT=$(find "$PROJECT" -not -path '*/.venv/*' -name pyproject.toml -print -quit 2>/dev/null)
    if [ -n "$FOUND_PYPROJECT" ]; then
        PROJECT_FILE="$FOUND_PYPROJECT"
        PROJECT_TYPE="pyproject"
        PROJECT_DIR_FOR_ENV=$(dirname "$FOUND_PYPROJECT")
    else
        FOUND_REQ=$(find "$PROJECT" -not -path '*/.venv/*' -name requirements.txt -print -quit 2>/dev/null)
        if [ -n "$FOUND_REQ" ]; then
            PROJECT_FILE="$FOUND_REQ"
            PROJECT_TYPE="requirements"
            PROJECT_DIR_FOR_ENV=$(dirname "$FOUND_REQ")
        else
            echo "⚠️  Нет pyproject.toml или requirements.txt в $PROJECT_NAME, пропускаю."
            FAIL_COUNT=$((FAIL_COUNT + 1))
            continue
        fi
    fi
    echo "📄 Найден файл: $PROJECT_FILE (тип: $PROJECT_TYPE)"

    # 2. Виртуальное окружение
    VENV_DIR="$PROJECT_DIR_FOR_ENV/.venv"
    if [ -d "$VENV_DIR" ] && [ "$FORCE_RECREATE" -eq 0 ]; then
        echo "✅ Виртуальное окружение уже существует: $VENV_DIR"
    else
        echo "🔧 Создаю виртуальное окружение в $VENV_DIR ..."
        [ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
        uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
    fi

    # 3. Активация и установка зависимостей
    CURRENT_DIR=$(pwd)
    source "$VENV_DIR/bin/activate"

    echo "📥 Устанавливаю зависимости..."
    INSTALL_OK=0
    if [ "$PROJECT_TYPE" = "pyproject" ]; then
        if uv sync 2>/dev/null; then
            INSTALL_OK=1
        else
            echo "⚠️  uv sync не удался, пробую резервный метод..."
            if uv pip install -e . 2>/dev/null; then
                INSTALL_OK=1
            elif uv pip install -r "$PROJECT_FILE" 2>/dev/null; then
                INSTALL_OK=1
            fi
        fi
    else
        if uv pip install -r "$PROJECT_FILE" 2>/dev/null; then
            INSTALL_OK=1
        fi
    fi
    deactivate

    if [ "$INSTALL_OK" -eq 0 ]; then
        echo "❌ Не удалось установить зависимости для $PROJECT_NAME, пропускаю."
        echo "$PROJECT_NAME" >> "$FAILED_LOG"
        cleanup_project_venv "$VENV_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # 4. Запуск модулей из основного окружения
    source "$CURRENT_DIR/.venv/bin/activate"

    PROJECT_OUT="$OUTPUT_DIR/$PROJECT_NAME"
    mkdir -p "$PROJECT_OUT"

    echo "🐍 Data Collector..."
    python "$CURRENT_DIR/app/data_collector/main.py" \
        --project-path "$VENV_DIR/bin/python" \
        --output "$PROJECT_OUT/dependencies.json" || {
        echo "❌ Ошибка Data Collector для $PROJECT_NAME"
        deactivate
        cleanup_project_venv "$VENV_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    }

    echo "🔗 Graph Builder..."
    python "$CURRENT_DIR/app/graph_builder/main.py" \
        --input "$PROJECT_OUT/dependencies.json" \
        --output-dir "$PROJECT_OUT" || {
        echo "❌ Ошибка Graph Builder для $PROJECT_NAME"
        deactivate
        cleanup_project_venv "$VENV_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    }

    echo "🛡️ Vulnerability Enricher..."
    python "$CURRENT_DIR/app/vulnerability_enricher/main.py" \
        --nx-input "$PROJECT_OUT/nx_graph.json" \
        --het-input "$PROJECT_OUT/hetero_data.pt" \
        --output-dir "$PROJECT_OUT" || {
        echo "❌ Ошибка Vulnerability Enricher для $PROJECT_NAME"
        deactivate
        cleanup_project_venv "$VENV_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    }

    deactivate   # Выходим из основного venv
    cleanup_project_venv "$VENV_DIR"
    echo "✅ Проект $PROJECT_NAME успешно обработан!"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
done

echo "========================================"
echo "🎉 Обработка завершена!"
echo "   Всего проектов: $TOTAL"
echo "   Успешно: $SUCCESS_COUNT"
echo "   Провалено: $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "   Список проваленных проектов сохранён в $FAILED_LOG"
fi