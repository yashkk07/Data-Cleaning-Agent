from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
from etl.pipeline import run_pipeline
from werkzeug.utils import secure_filename
from etl.extract.reader import read_csv_safe
from etl.transform import cleaners
import pandas as pd

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Case A: second-form submission (cleaning request) - uses uploaded_filename
        uploaded_filename = request.form.get("uploaded_filename")
        if uploaded_filename:
            filename = secure_filename(uploaded_filename)
            input_path = os.path.join(UPLOAD_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"cleaned_{filename}")

            if not os.path.exists(input_path):
                flash("Uploaded file not found on server; please re-upload")
                return redirect(request.url)

            mode = request.form.get("mode", "full")
            if mode == "full":
                try:
                    result = run_pipeline(input_path, output_path)
                    history = result.get("history", []) if isinstance(result, dict) else []
                    return render_template("result.html", history=history, output_filename=os.path.basename(output_path))
                except Exception as e:
                    flash(str(e))
                    return redirect(request.url)

            # selective mode
            selected_tools = request.form.getlist("tools")
            trim_cols = request.form.get("trim_whitespace_columns", "")
            convert_col = request.form.get("convert_numeric_column", "")
            parse_col = request.form.get("parse_datetime_column", "")

            try:
                history = apply_selective_cleaners(input_path, output_path, selected_tools, {
                    "trim_whitespace_columns": trim_cols,
                    "convert_numeric_column": convert_col,
                    "parse_datetime_column": parse_col,
                })
                return render_template("result.html", history=history, output_filename=os.path.basename(output_path))
            except Exception as e:
                flash(str(e))
                return redirect(request.url)

        # Case B: initial upload form submission
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_DIR, filename)
            file.save(input_path)

            # Build EDA: columns, dtypes, first 5 rows
            try:
                df, _ = read_csv_safe(input_path)
            except Exception as e:
                flash(f"Failed to read uploaded file for preview: {e}")
                return redirect(request.url)

            eda = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head": df.head(5).to_dict(orient="records"),
                "rows": len(df),
            }

            return render_template("index.html", eda=eda, uploaded_filename=filename)

    return render_template("index.html")


@app.route("/download/<filename>")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("File not found")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8501)))


def apply_selective_cleaners(input_path: str, output_path: str, tools_list: list, args: dict) -> list:
    """Apply a list of cleaner names (strings) in order to the input CSV and save output.

    Returns a history list of steps applied.
    """
    df, _ = read_csv_safe(input_path)
    history = []

    for tool in tools_list:
        step = {"tool": tool, "args": {}, "status": "pending"}
        try:
            if tool == "clean_column_names":
                df = cleaners.clean_column_names(df)
            elif tool == "standardize_missing":
                df = cleaners.standardize_missing(df)
            elif tool == "trim_whitespace":
                cols = [c.strip() for c in args.get("trim_whitespace_columns", "").split(",") if c.strip()]
                kwargs = {"columns": cols} if cols else {}
                step["args"] = kwargs
                df = cleaners.trim_whitespace(df, **kwargs)
            elif tool == "remove_duplicates":
                df = cleaners.remove_duplicates(df)
            elif tool == "convert_numeric":
                col = args.get("convert_numeric_column", "")
                if not col:
                    raise ValueError("convert_numeric requires a column name")
                step["args"] = {"column": col}
                df = cleaners.convert_numeric(df, col)
            elif tool == "parse_datetime":
                col = args.get("parse_datetime_column", "")
                if not col:
                    raise ValueError("parse_datetime requires a column name")
                step["args"] = {"column": col}
                df = cleaners.parse_datetime(df, col)
            else:
                raise ValueError(f"Unknown tool: {tool}")

            step["status"] = "success"
        except Exception as e:
            step["status"] = "failed"
            step["error"] = str(e)
            history.append(step)
            # stop on failure
            raise

        history.append(step)

    # write out
    df.to_csv(output_path, index=False)
    return history
