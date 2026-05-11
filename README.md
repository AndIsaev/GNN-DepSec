# GNN-DepSec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.3%2B-green.svg)](https://pyg.org)


**GNN-DepSec** — система мониторинга и анализа уязвимостей в зависимостях программного обеспечения на основе графовых нейронных сетей.  
Проект создан в рамках магистерской диссертации (НИУ ВШЭ, «Кибербезопасность», 2026) и предназначен для выявления скрытых транзитивных угроз в экосистеме Python (PyPI), которые пропускают классические SCA‑инструменты.

## Содержание
- [Почему это важно?](#почему-это-важно)
- [Основные возможности](#основные-возможности)
- [Структура репозитория](#структура-репозитория)
- [Архитектура системы](#архитектура-системы)
- [Установка и зависимости](#установка-и-зависимости)
- [Быстрый старт](#быстрый-старт)
- [Подробное руководство](#подробное-руководство)
  - [Генерация синтетических проектов](#генерация-синтетических-проектов)
  - [Пакетная обработка проектов (train_script.sh)](#пакетная-обработка-проектов-train_scriptsh)
  - [Обучение модели на множестве проектов](#обучение-модели-на-множестве-проектов)
  - [Инференс на целевом проекте](#инференс-на-целевом-проекте)

## Почему это важно?
Современные приложения на 98% состоят из открытых зависимостей, при этом 93% кодовых баз содержат компоненты без активной поддержки. Традиционные SCA‑решения (Software Composition Analysis) работают по принципу «версия → CVE» и не учитывают структурный контекст использования библиотеки. Как следствие — высокий процент ложных срабатываний и пропуск реальных угроз, приходящих транзитивно.

**GNN-DepSec** представляет зависимости проекта в виде гетерогенного графа и применяет гибридную графовую нейронную сеть (RGCN + GraphSAGE + GAT) для оценки уровня риска каждого компонента с учётом его окружения.

## Основные возможности
- 🔍 **Сбор зависимостей** любого Python‑проекта прямо из виртуального окружения.
- 🧠 **Построение гетерогенного графа** с узлами `Package`, `Version`, `Vulnerability` и типизированными связями (`DEPENDS_ON`, `VULNERABLE_TO`, `HAS_VERSION`).
- 📡 **Обогащение уязвимостями** через OSV API (автоматическое разрешение алиасов и расчёт CVSS).
- 🤖 **Гибридная GNN‑модель** – трансдуктивный и индуктивный режимы (на базе PyTorch Geometric).
- 📊 **Генерация наглядного HTML‑отчёта** с разделением на критические и подозрительные зависимости, путями достижимости и рекомендациями.
- 🗺️ **Интерактивная визуализация графа** с цветовой индикацией уровня риска.


## Структура репозитория
```text
GNN-DepSec/
├── app/                          # исходный код системы
│   ├── data_collector/           # модуль сбора зависимостей
│   ├── graph_builder/            # модуль построения гетерогенного графа
│   ├── vulnerability_enricher/   # модуль обогащения уязвимостями
│   ├── gnn_inference_engine/     # модуль обучения и инференса GNN
│   └── reporter_visualizer/      # модуль отчётов и визуализации
|   └── train_multigraph          # модуль обучения индуктивной модели на графах
├── tools/                        # утилиты для пакетной обработки и генерации
│   ├── generate_fake_projects.py # генератор синтетических проектов
├── pyproject.toml                # зависимости окружения разработки
├── train_script.sh               # пакетный прогон через первые три модуля
└── README.md
```

## Архитектура системы
Система построена как сквозной конвейер обработки данных из пяти модулей:
```markdown
+----------------+    +---------------+    +------------------------+
| Data Collector | -> | Graph Builder | -> | Vulnerability Enricher |
+----------------+    +---------------+    +------------------------+
                                                        |
                                                        v
                                           +----------------------+
                                           | GNN Inference Engine |
                                           +----------------------+
                                                        |
                                                        v
                                            +----------------------+
                                            | Reporter & Visualizer|
                                            +----------------------+
```
### Модули
1. **Data Collector** – собирает список установленных пакетов и их зависимостей из виртуального окружения с помощью `pipdeptree`.
2. **Graph Builder** – строит гетерогенный граф, запрашивает метаданные пакетов через PyPI JSON API и GitHub API, формирует признаковые тензоры для GNN.
3. **Vulnerability Enricher** – асинхронно опрашивает OSV API, добавляет узлы `Vulnerability` и рёбра `VULNERABLE_TO`, рассчитывает CVSS-оценки.
4. **GNN Inference Engine** – обучает/загружает гибридную GNN-модель и вычисляет `risk_score` для каждой версии пакета.
5. **Reporter & Visualizer** – генерирует HTML-отчёт с таблицами критических и подозрительных уязвимостей, строит интерактивный граф зависимостей.

## Установка и зависимости
- Python 3.10+
- Менеджер пакетов `uv` (рекомендуется для быстрого создания окружений)
- Основные библиотеки (устанавливаются в основное виртуальное окружение):
```bash
uv sync
```
Для сборки графа используется утилита pipdeptree, которая устанавливается автоматически в целевое окружение.

## Быстрый старт
```shell
git clone git@github.com:AndIsaev/GNN-DepSec.git && cd GNN-DepSec
```
Чтобы проанализировать один проект, потребуется выполнить четыре шага (подставьте свои пути):

#### 1. Сбор данных
```shell
uv run python app/data_collector/main.py --project-path /path/to/venv/bin/python --output data.json
```
#### 2. Построение графа
```shell
uv run python python app/graph_builder/main.py --input data.json --output-dir ./graph_out
```
#### 3. Обогащение данными об уязвимостях
```shell
uv run python app/vulnerability_enricher/main.py --nx-input ./graph_out/nx_graph.json --het-input ./graph_out/hetero_data.pt --output-dir ./enriched
```
#### 4. Инференс (индуктивный, с предобученной моделью)
```shell
uv run python app/gnn_inference_engine/main.py --nx-input ./enriched/nx_graph.json --het-input ./enriched/hetero_data.pt --output-dir ./scored --mode inductive --model-path pretrained/gnn_model.pt --norm-stats pretrained/norm_stats.pt
```
После этого в ./scored появятся обновлённые графы с risk_score, которые можно визуализировать с помощью модуля Reporter & Visualizer.

## Подробное руководство
### Генерация синтетических проектов
Скрипт `tools/generate_fake_projects.py` создаёт зависимости с заданным набором для Python-проектов. Зависимости берутся из двух пулов: уязвимые версии (с известными CVE) и безопасные (современные стабильные релизы).
```shell
# Стандартный набор: 10 critical, 20 mixed, 70 safe, мин. 5 зависимостей
uv run python tools/generate_fake_projects.py

# Настроить количество проектов и минимальное число зависимостей
uv run python tools/generate_fake_projects.py --critical 30 --mixed 60 --safe 120 --min-deps 7

# Указать выходную директорию и зафиксировать seed для воспроизводимости
uv run python tools/generate_fake_projects.py --output-dir ./my_projects --seed 42
```
### Пакетная обработка проектов (train_script.sh)
`train_script.sh` автоматизирует всё: от генерации проектов (опционально) до последовательного запуска Data Collector, Graph Builder и Vulnerability Enricher для каждого проекта. Использует uv для управления виртуальными окружениями.
#### Переменные окружения для настройки:
```text
GENERATE_FAKE=1 – включить генерацию синтетических проектов перед обработкой.

N_CRITICAL, N_MIXED, N_SAFE – количество проектов каждой категории (по умолчанию 10/20/70).

MIN_DEPS – минимальное число зависимостей в проекте.

FAKE_OUT_DIR – директория для сгенерированных проектов (по умолчанию ./datasets/fake_projects).

PROJECTS_DIR – директория с проектами для обработки (если не задана, берётся из FAKE_OUT_DIR).

OUTPUT_DIR – куда сохранять готовые графы (по умолчанию ./datasets/graphs).

FORCE_RECREATE – 1 для пересоздания venv каждого проекта.
```
#### Примеры:
Сгенерировать 20 проектов (5 critical, 5 mixed, 10 safe) и сразу обработать
```shell
GENERATE_FAKE=1 N_CRITICAL=5 N_MIXED=5 N_SAFE=10 ./train_script.sh
```
Только обработка существующих проектов из другой папки
```shell
PROJECTS_DIR=./existing_projects OUTPUT_DIR=./graphs ./train_script.sh
```
После выполнения в OUTPUT_DIR появятся подпапки с файлами dependencies.json, nx_graph.json, hetero_data.pt и обогащённые версии.

### Обучение модели на множестве проектов
Скрипт `app/train_multigraph/main.py` обучает индуктивную модель на наборе предварительно обработанных проектов (папки с hetero_data.pt). Он вычисляет глобальные статистики нормализации и сохраняет модель и статистики для последующего использования.

```shell
uv run python app/train_multigraph/main.py --data-dir ./datasets/graphs \
                                 --epochs 200 \
                                 --output-model ./pretrained/gnn_model.pt \
                                 --device cpu
```
#### Основные аргументы:
```text
--data-dir – директория с подпапками проектов (в каждой должен быть hetero_data.pt).

--epochs – число эпох обучения (рекомендуется 200–300).

--lr – скорость обучения (по умолчанию 0.001).

--output-model – путь для сохранения весов модели.

--val-project – (опционально) путь к проекту для валидации.

--device – cpu или cuda.
```
После завершения в указанной директории появятся gnn_model.pt и norm_stats.pt. Именно их нужно передавать в индуктивный режим инференса.
### Инференс на целевом проекте
Модуль GNN Inference Engine (скрипт `app/gnn_inference_engine/main.py`) поддерживает два режима:
#### 1. Трансдуктивный – обучение с нуля на одном проекте (идеально для глубокого аудита).
```shell
uv run python app/gnn_inference_engine/main.py \
    --nx-input ./enriched/nx_graph.json \
    --het-input ./enriched/hetero_data.pt \
    --output-dir ./scored \
    --mode transductive \
    --epochs 200
```
#### 2. Индуктивный – использование предобученной модели (быстрый анализ).
```shell
uv run python app/gnn_inference_engine/main.py \
    --nx-input ./enriched/nx_graph.json \
    --het-input ./enriched/hetero_data.pt \
    --output-dir ./scored \
    --mode inductive \
    --model-path ./pretrained/gnn_model.pt \
    --norm-stats ./pretrained/norm_stats.pt
```
#### Дообучение (fine‑tune) в индуктивном режиме – адаптация общей модели под конкретный проект (указывается число эпох):
```shell
uv run python app/gnn_inference_engine/main.py ... --mode inductive ... --fine-tune 50
```
Если в проекте нет ни одной уязвимости, модель не будет обучаться (или дообучаться), а всем версиям присваивается risk_score = 0.0.

Результаты сохраняются в output-dir: hetero_data.pt с полем risk_score и nx_graph_scored.json для визуализации.


