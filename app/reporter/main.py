#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
from jinja2 import Template
from networkx.readwrite import json_graph
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logger = logging.getLogger("reporter")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
# Пороги для классификации узлов по риску
SUSPICIOUS_THRESHOLD = 0.3   # нижний порог для "подозрительных" пакетов
HIGH_RISK_THRESHOLD = 0.7    # выше этого значения – красный цвет в визуализации

# Цвета для визуализации
COLOR_HIGH_RISK = "#ff0000"       # красный
COLOR_MEDIUM_RISK = "#ffcc00"     # жёлтый
COLOR_LOW_RISK = "#00cc00"        # зелёный
COLOR_PACKAGE = "lightblue"       # голубой для узлов-пакетов
COLOR_VULNERABILITY = "orange"    # оранжевый для узлов-уязвимостей
COLOR_UNKNOWN = "gray"            # серый для неизвестных типов

# ---------------------------------------------------------------------------
# HTML-шаблон отчёта
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Vulnerability Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .risk-high { background-color: #ffcccc; }
        .risk-medium { background-color: #fff3cc; }
        .suspicious { background-color: #fff8e1; }
        caption { font-weight: bold; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Vulnerability Analysis Report</h1>
    <p><strong>Threshold for critical:</strong> {{ threshold }}</p>
    <p>Пакеты с риском ≥ {{ threshold }} считаются уязвимыми, от {{ suspicious_threshold }} до {{ threshold }} – подозрительными (нет прямых CVE, но соседствуют с уязвимыми).</p>

    <h2>Критические уязвимости</h2>
    <table>
        <tr>
            <th>Package</th>
            <th>Version</th>
            <th>CVE</th>
            <th>CVSS Score</th>
            <th>Risk Score</th>
            <th>Vulnerability Path</th>
            <th>Fixed In</th>
        </tr>
        {% for item in vulnerable %}
        <tr class="{% if item.risk_score > 0.7 %}risk-high{% elif item.risk_score > 0.5 %}risk-medium{% endif %}">
            <td>{{ item.name }}</td>
            <td>{{ item.version }}</td>
            <td>{{ item.cve_id }}</td>
            <td>{{ item.cvss_score }}</td>
            <td>{{ item.risk_score }}</td>
            <td>{{ item.path }}</td>
            <td>{{ item.fixed_in }}</td>
        </tr>
        {% endfor %}
    </table>

    {% if suspicious %}
    <h2>Подозрительные зависимости (требуют ручного анализа)</h2>
    <p>Эти пакеты не имеют зарегистрированных CVE, но получили повышенную оценку риска из-за близости к уязвимым компонентам в графе зависимостей.</p>
    <table>
        <tr>
            <th>Package</th>
            <th>Version</th>
            <th>Risk Score</th>
            <th>Connection Path</th>
        </tr>
        {% for item in suspicious %}
        <tr class="suspicious">
            <td>{{ item.name }}</td>
            <td>{{ item.version }}</td>
            <td>{{ item.risk_score }}</td>
            <td>{{ item.path }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_nx_graph(path: Path) -> nx.MultiDiGraph:
    """Загружает NetworkX MultiDiGraph из JSON-файла, сохранённого в формате node-link.

    Args:
        path: Путь к JSON-файлу с данными графа.

    Returns:
        Восстановленный граф NetworkX.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


# ---------------------------------------------------------------------------
# Анализ уязвимостей
# ---------------------------------------------------------------------------
def find_vulnerabilities(
    nx_graph: nx.MultiDiGraph,
    threshold: float,
    suspicious_threshold: float = SUSPICIOUS_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Извлекает из графа уязвимые и подозрительные версии пакетов.

    Для каждого узла типа 'version' проверяется наличие связанных CVE
    и значение risk_score. Узлы с оценкой >= threshold классифицируются
    как уязвимые, а узлы с оценкой от suspicious_threshold до threshold
    без зарегистрированных CVE — как подозрительные.
    Для каждого уязвимого узла указывается путь до корневого проекта.

    Args:
        nx_graph: Граф зависимостей с атрибутами risk_score и VULNERABLE_TO.
        threshold: Порог риска для включения в список уязвимых (например, 0.5).
        suspicious_threshold: Нижний порог для подозрительных пакетов (по умолчанию 0.3).

    Returns:
        Кортеж из двух списков:
        - vulnerable: список словарей с данными уязвимых версий (по одному на каждый CVE);
        - suspicious: список словарей с подозрительными версиями.
    """
    vulnerable: List[Dict[str, Any]] = []
    suspicious: List[Dict[str, Any]] = []

    # Определяем корневые узлы один раз для ускорения
    root_nodes = [n for n, a in nx_graph.nodes(data=True) if a.get("is_root")]

    for node, attrs in nx_graph.nodes(data=True):
        if attrs.get("node_type") != "version":
            continue
        risk = attrs.get("risk_score")
        if risk is None:
            continue

        # Сбор CVE, связанных с данной версией
        cve_entries: List[Dict[str, Any]] = []
        for _, tgt, edge_data in nx_graph.out_edges(node, data=True):
            if edge_data.get("type") == "VULNERABLE_TO":
                fixed_in = edge_data.get("fixed_in")
                cve_id = None
                cvss_score = None
                if tgt in nx_graph.nodes:
                    cve_attrs = nx_graph.nodes[tgt]
                    cve_id = cve_attrs.get("cve_id")
                    cvss_score = cve_attrs.get("cvss_score")
                cve_entries.append({
                    "cve_id": cve_id or "N/A",
                    "cvss_score": cvss_score,
                    "fixed_in": fixed_in or "N/A",
                })

        # Построение читаемого пути от корневого проекта до текущего узла
        paths: List[str] = []
        for root in root_nodes:
            try:
                path_nodes = nx.shortest_path(nx_graph, root, node)
                readable = []
                for nd in path_nodes:
                    nd_attrs = nx_graph.nodes[nd]
                    if nd_attrs.get("node_type") == "version":
                        readable.append(nd_attrs.get("name", nd))
                    else:
                        readable.append(nd)
                paths.append(" → ".join(readable))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        path_str = " ; ".join(paths) if paths else "unknown"

        if risk >= threshold:
            # Уязвимый узел: по одной строке на каждый связанный CVE
            if cve_entries:
                for cve in cve_entries:
                    vulnerable.append({
                        "name": attrs["name"],
                        "version": attrs["version"],
                        "risk_score": risk,
                        "cve_id": cve["cve_id"],
                        "cvss_score": cve["cvss_score"],
                        "fixed_in": cve["fixed_in"],
                        "path": path_str,
                    })
            else:
                # Теоретически не должно случиться, но оставляем для полноты
                vulnerable.append({
                    "name": attrs["name"],
                    "version": attrs["version"],
                    "risk_score": risk,
                    "cve_id": "N/A",
                    "cvss_score": None,
                    "fixed_in": "N/A",
                    "path": path_str,
                })
        elif risk >= suspicious_threshold and not cve_entries:
            # Подозрительный: риск повышен, но нет прямых CVE
            suspicious.append({
                "name": attrs["name"],
                "version": attrs["version"],
                "risk_score": risk,
                "path": path_str,
            })

    # Сортировка по убыванию риска для наглядности
    vulnerable.sort(key=lambda x: x["risk_score"], reverse=True)
    suspicious.sort(key=lambda x: x["risk_score"], reverse=True)
    return vulnerable, suspicious


# ---------------------------------------------------------------------------
# Генерация отчёта
# ---------------------------------------------------------------------------
def generate_report(
    vulnerable: List[Dict[str, Any]],
    suspicious: List[Dict[str, Any]],
    threshold: float,
    output_path: Path,
) -> None:
    """Генерирует HTML-отчёт с таблицами уязвимых и подозрительных пакетов.

    Использует предопределённый Jinja2-шаблон HTML_TEMPLATE.

    Args:
        vulnerable: Список уязвимых записей (результат find_vulnerabilities).
        suspicious: Список подозрительных записей.
        threshold: Порог риска, используется в описании отчёта.
        output_path: Путь для сохранения HTML-файла.
    """
    template = Template(HTML_TEMPLATE)
    html = template.render(
        vulnerable=vulnerable,
        suspicious=suspicious,
        threshold=threshold,
        suspicious_threshold=SUSPICIOUS_THRESHOLD,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Визуализация графа
# ---------------------------------------------------------------------------
def generate_visualization(nx_graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Создаёт интерактивную HTML‑визуализацию графа с помощью pyvis.

    Узлы окрашиваются в зависимости от типа и уровня риска:
    - version: цвет от зелёного (низкий риск) до красного (высокий);
    - package: голубой;
    - vulnerability: оранжевый.
    Рёбра подписываются типом связи.

    Args:
        nx_graph: Граф NetworkX с атрибутами risk_score.
        output_path: Путь для сохранения HTML-файла визуализации.
    """
    net = Network(height="800px", width="100%", directed=True)

    for node, attrs in nx_graph.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        if node_type == "version":
            risk = attrs.get("risk_score", 0.0)
            if risk > HIGH_RISK_THRESHOLD:
                color = COLOR_HIGH_RISK
            elif risk > SUSPICIOUS_THRESHOLD:
                color = COLOR_MEDIUM_RISK
            else:
                color = COLOR_LOW_RISK
            title = (
                f"Name: {attrs.get('name')}<br>"
                f"Version: {attrs.get('version')}<br>"
                f"Risk: {risk:.4f}"
            )
            label = attrs.get("name", node)
        elif node_type == "package":
            color = COLOR_PACKAGE
            # Извлекаем имя пакета из идентификатора "package/..."
            pkg_name = node.split("/", 1)[1] if "/" in node else node
            title = f"Package: {pkg_name}"
            label = pkg_name
        elif node_type == "vulnerability":
            color = COLOR_VULNERABILITY
            cve = attrs.get("cve_id", node.split("/", 1)[1] if "/" in node else node)
            title = f"CVE: {cve}"
            label = cve
        else:
            color = COLOR_UNKNOWN
            title = node
            label = node

        net.add_node(node, label=label, color=color, title=title)

    for src, dst, data in nx_graph.edges(data=True):
        edge_type = data.get("type", "")
        net.add_edge(src, dst, title=edge_type)

    # Показываем кнопки управления физикой для интерактивности
    net.show_buttons(filter_=["physics"])
    net.save_graph(str(output_path))
    logger.info("Visualization saved to %s", output_path)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    """Основная функция – парсинг аргументов и запуск генерации отчётов.

    Загружает граф, находит уязвимые/подозрительные узлы, создаёт
    HTML-отчёт и интерактивную визуализацию.
    """
    parser = argparse.ArgumentParser(
        description="Reporter & Visualizer – анализ и визуализация графа уязвимостей."
    )
    parser.add_argument(
        "--nx-input",
        type=Path,
        required=True,
        help="Путь к NetworkX JSON с risk_score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./report"),
        help="Директория для сохранения отчётов.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог риска для отнесения к критическим уязвимостям.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробное логирование.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.nx_input.is_file():
        logger.error("Input file not found: %s", args.nx_input)
        sys.exit(1)

    nx_graph = load_nx_graph(args.nx_input)
    vulnerable, suspicious = find_vulnerabilities(nx_graph, args.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(vulnerable, suspicious, args.threshold, args.output_dir / "report.html")
    generate_visualization(nx_graph, args.output_dir / "graph.html")


if __name__ == "__main__":
    main()