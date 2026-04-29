#!/usr/bin/env python3
"""
Reporter & Visualizer (FR5, FR6) – генерация отчёта и визуализация графа.

Принимает NetworkX JSON с risk_score и порог, строит HTML-отчёт и визуализацию.

Использование:
    python reporter.py --nx-input scored/nx_graph_scored.json --output-dir ./report --threshold 0.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import networkx as nx
from networkx.readwrite import json_graph
from jinja2 import Template
from pyvis.network import Network

logger = logging.getLogger("reporter")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Vulnerability Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .risk-high { background-color: #ffcccc; }
        .risk-medium { background-color: #fff3cc; }
    </style>
</head>
<body>
    <h1>Vulnerability Analysis Report</h1>
    <h3>Threshold: {{ threshold }}</h3>
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
</body>
</html>
"""


def load_nx_graph(path: Path) -> nx.MultiDiGraph:
    """Загружает NetworkX граф из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


def find_vulnerabilities(nx_graph: nx.MultiDiGraph, threshold: float) -> List[dict]:
    """Находит все уязвимые версии и для каждой уязвимости формирует отдельную запись."""
    vulnerable = []
    for node, attrs in nx_graph.nodes(data=True):
        if attrs.get("node_type") != "version":
            continue
        risk = attrs.get("risk_score")
        if risk is None or risk < threshold:
            continue

        # Все корневые узлы (для поиска пути)
        root_nodes = [n for n, a in nx_graph.nodes(data=True) if a.get("is_root")]
        # Получаем все CVE через исходящие рёбра VULNERABLE_TO
        for _, tgt, data in nx_graph.out_edges(node, data=True):
            if data.get("type") == "VULNERABLE_TO":
                fixed_in = data.get("fixed_in")
                cve_id = None
                cvss_score = None
                cvss_vector = None
                if tgt in nx_graph.nodes:
                    cve_attrs = nx_graph.nodes[tgt]
                    cve_id = cve_attrs.get("cve_id")
                    cvss_score = cve_attrs.get("cvss_score")
                    cvss_vector = cve_attrs.get("cvss_vector")
                # Формируем читаемый путь
                paths = []
                for root in root_nodes:
                    try:
                        path_nodes = nx.shortest_path(nx_graph, root, node)
                        # Заменяем ID на имена пакетов (из атрибутов)
                        readable = []
                        for nd in path_nodes:
                            nd_attrs = nx_graph.nodes[nd]
                            if nd_attrs.get("node_type") == "version":
                                readable.append(nd_attrs.get("name", nd))
                            else:
                                readable.append(nd)   # на практике путь только из version
                        paths.append(" → ".join(readable))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                path_str = " ; ".join(paths) if paths else "unknown"

                vulnerable.append({
                    "name": attrs["name"],
                    "version": attrs["version"],
                    "risk_score": risk,
                    "cve_id": cve_id or "N/A",
                    "cvss_score": cvss_score,
                    "cvss_vector": cvss_vector or "",
                    "fixed_in": fixed_in or "N/A",
                    "path": path_str,
                })

    # Сортируем по убыванию риска
    vulnerable.sort(key=lambda x: x["risk_score"], reverse=True)
    return vulnerable


def generate_report(vulnerable: List[dict], threshold: float, output_path: Path) -> None:
    """Генерирует HTML-отчёт с таблицей уязвимостей."""
    template = Template(HTML_TEMPLATE)
    html = template.render(vulnerable=vulnerable, threshold=threshold)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Report saved to %s", output_path)


def generate_visualization(nx_graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Создаёт интерактивную визуализацию графа."""
    net = Network(height="800px", width="100%", directed=True)

    for node, attrs in nx_graph.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        if node_type == "version":
            risk = attrs.get("risk_score", 0.0)
            if risk > 0.7:
                color = "#ff0000"
            elif risk > 0.3:
                color = "#ffcc00"
            else:
                color = "#00cc00"
            title = (f"Name: {attrs.get('name')}<br>Version: {attrs.get('version')}<br>"
                     f"Risk: {risk:.4f}")
            label = attrs.get("name", node)
        elif node_type == "package":
            color = "lightblue"
            # Имя пакета из id
            pkg_name = node.split("/", 1)[1] if "/" in node else node
            title = f"Package: {pkg_name}"
            label = pkg_name
        elif node_type == "vulnerability":
            color = "orange"
            cve = attrs.get("cve_id", node.split("/", 1)[1] if "/" in node else node)
            title = f"CVE: {cve}"
            label = cve
        else:
            color = "gray"
            title = node
            label = node
        net.add_node(node, label=label, color=color, title=title)

    for src, dst, data in nx_graph.edges(data=True):
        edge_type = data.get("type", "")
        net.add_edge(src, dst, title=edge_type)

    net.show_buttons(filter_=['physics'])
    net.save_graph(str(output_path))
    logger.info("Visualization saved to %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Reporter & Visualizer")
    parser.add_argument("--nx-input", type=Path, required=True,
                        help="Путь к NetworkX JSON с risk_score.")
    parser.add_argument("--output-dir", type=Path, default=Path("./report"),
                        help="Директория для отчётов.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Порог риска для отображения.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.nx_input.is_file():
        logger.error("Input file not found: %s", args.nx_input)
        sys.exit(1)

    nx_graph = load_nx_graph(args.nx_input)
    vulnerable = find_vulnerabilities(nx_graph, args.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(vulnerable, args.threshold, args.output_dir / "report.html")
    generate_visualization(nx_graph, args.output_dir / "graph.html")


if __name__ == "__main__":
    main()