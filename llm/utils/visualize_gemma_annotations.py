#!/usr/bin/env python3
"""Utility Flask app to visualize enhanced Gemma annotations.

Run with:
    python utils/visualize_gemma_annotations.py --data_dir path/to/enhanced_gemma_annotations

The application scans the specified directory for sub-folders produced by
`gemma3_enhance.py` (each containing an `enhanced_expressions.json` and the
corresponding PNG with the red bounding box). It then presents **all** objects
on a single scrollable web page: the image on the left and, on the right, the
original expressions, the LLM-generated variations, the unique description and
unique expressions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Dict

from flask import (
    Flask,
    abort,
    render_template_string,
    send_from_directory,
    url_for,
)

# ---------------------------------------------------------------------------
# HTML template (inline to keep this file self-contained)
# ---------------------------------------------------------------------------
HTML_TEMPLATE: str = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Gemma Annotation Visualizer</title>
    <style>
        body        { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; }
        .container  { width: 92%; max-width: 1400px; margin: 20px auto; }
        h1          { text-align: center; margin-bottom: 40px; }

        .item       { display: flex; gap: 20px; padding: 20px 0; border-bottom: 1px solid #ddd; }
        .item img   { max-width: 400px; height: auto; border: 1px solid #ccc; }

        .details    { flex: 1; }
        .details h3 { margin: 0 0 10px 0; font-size: 1.1rem; }

        .expr-list  { margin: 0 0 10px 20px; }
        .expr-list li { margin-bottom: 4px; }

        .unique-desc { background: #fff; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gemma Annotation Visualizer (<code>{{ data_dir }}</code>)</h1>
        {% if not items %}
          <p>No annotation folders found under <code>{{ data_dir }}</code>.</p>
        {% endif %}
        {% for item in items %}
        <div class="item">
            <div>
                <img src="{{ url_for('serve_image', path=item.image_path) }}" alt="{{ item.image }}" />
            </div>
            <div class="details">
                <h3>{{ item.image }} — object {{ item.object_id }} <em>({{ item.category }})</em></h3>

                <strong>Original expressions</strong>
                <ul class="expr-list">
                    {% for expr in item.original_expressions %}<li>{{ expr }}</li>{% endfor %}
                </ul>

                <strong>Enhanced expressions (variation of originals)</strong>
                <ul class="expr-list">
                    {% for ee in item.enhanced_expressions %}
                      <li><em>original:</em> {{ ee.original_expression }}<br/>
                          <em>variation:</em> {{ ee.variation }}</li>
                    {% endfor %}
                </ul>

                <strong>Unique description</strong>
                <div class="unique-desc">{{ item.unique_description }}</div>

                <strong>Unique expressions</strong>
                <ul class="expr-list">
                    {% for ue in item.unique_expressions %}<li>{{ ue }}</li>{% endfor %}
                </ul>
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_items(data_dir: Path) -> List[Dict[str, Any]]:
    """Walk `data_dir` and collect data for each object folder.

    Each sub-folder is expected to contain an `enhanced_expressions.json` file
    and at least one PNG. The first PNG found is used for display.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    items: List[Dict[str, Any]] = []

    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        json_path = sub / "enhanced_expressions.json"
        if not json_path.is_file():
            # Skip folders that are missing the expected JSON file.
            continue

        try:
            record = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"[WARN] Failed to parse {json_path}: {exc}")
            continue

        # Locate first PNG to show.
        image_file = next((f for f in sub.iterdir() if f.suffix.lower() == ".png"), None)
        if image_file is None:
            print(f"[WARN] No image found in {sub}")
            continue

        items.append(
            {
                "image": record.get("image"),
                "object_id": record.get("object_id"),
                "category": record.get("category"),
                "original_expressions": record.get("original_expressions", []),
                "enhanced_expressions": record.get("enhanced_data", {}).get("enhanced_expressions", []),
                "unique_description": record.get("enhanced_data", {}).get("unique_description", ""),
                "unique_expressions": record.get("enhanced_data", {}).get("unique_expressions", []),
                # Relative path used in the route parameter
                "image_path": image_file.relative_to(data_dir).as_posix(),
            }
        )

    return items

# ---------------------------------------------------------------------------
# Flask factory
# ---------------------------------------------------------------------------

def create_app(data_dir: Path) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    items = load_items(data_dir)

    @app.route("/")
    def index():  # type: ignore[return-value]
        return render_template_string(HTML_TEMPLATE, items=items, data_dir=data_dir)

    @app.route("/images/<path:path>")
    def serve_image(path: str):  # type: ignore[return-value]
        # `path` is relative to `data_dir` (e.g. "img_123_obj_1/img_123_obj_1.png")
        full_dir = data_dir / os.path.dirname(path)
        filename = os.path.basename(path)
        if not (full_dir / filename).is_file():
            abort(404)
        return send_from_directory(full_dir, filename)

    return app

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize enhanced Gemma annotations with a simple Flask app")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="enhanced_gemma_annotations",
        help="Directory containing sub-folders with annotation outputs",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    try:
        app = create_app(data_dir)
    except FileNotFoundError as exc:
        print(exc)
        return

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main() 