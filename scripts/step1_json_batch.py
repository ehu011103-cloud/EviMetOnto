import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


SEED_CLASSES = [
    "GeologicalObject",
    "Rock",
    "Mineral",
    "Alteration",
    "Structure",
    "DepositType",
    "Process",
]


def load_dotenv_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ModelConfig:
    model: str
    alias: str
    base_url: str
    api_key_env: str

    @property
    def api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise RuntimeError(
                f"Missing API key. Set environment variable {self.api_key_env} "
                f"for model '{self.alias}'."
            )
        return value


def safe_alias(value: str) -> str:
    value = value.strip() or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "model"


def parse_model_spec(spec: str, default_base_url: str, default_api_key_env: str) -> ModelConfig:
    """Parse MODEL[@ALIAS][|BASE_URL][|API_KEY_ENV]."""
    parts = [p.strip() for p in spec.split("|")]
    model_alias = parts[0]
    base_url = parts[1] if len(parts) > 1 and parts[1] else default_base_url
    api_key_env = parts[2] if len(parts) > 2 and parts[2] else default_api_key_env

    if "@" in model_alias:
        model, alias = model_alias.rsplit("@", 1)
    else:
        model = model_alias
        alias = model_alias

    model = model.strip()
    if not model:
        raise ValueError(f"Invalid empty model spec: {spec!r}")

    return ModelConfig(
        model=model,
        alias=safe_alias(alias),
        base_url=base_url,
        api_key_env=api_key_env,
    )


def parse_json_content(filepath: str, max_chars: int) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")

        content_data = data.get("content", [])
        body_text = content_to_text(content_data)

        full_text = f"Title: {title}\nAbstract: {abstract}\nBody:\n{body_text}"
        return full_text[:max_chars]
    except Exception as exc:
        print(f"[WARN] Failed to parse {os.path.basename(filepath)}: {exc}")
        return ""


def content_to_text(content_data) -> str:
    if isinstance(content_data, list):
        return "\n".join(str(p) for p in content_data if len(str(p)) > 20)

    if isinstance(content_data, str):
        return content_data

    if not isinstance(content_data, dict):
        return str(content_data)

    parts = []
    abstract = content_data.get("abstract")
    if abstract:
        parts.append(str(abstract))

    paragraphs = content_data.get("paragraphs")
    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            if isinstance(paragraph, dict):
                text = paragraph.get("raw_text") or paragraph.get("text") or paragraph.get("content")
            else:
                text = str(paragraph)
            if text and len(str(text)) > 20:
                parts.append(str(text))

    if parts:
        return "\n".join(parts)

    chunks = content_data.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("raw_text") or chunk.get("content")
            else:
                text = str(chunk)
            if text and len(str(text)) > 20:
                parts.append(str(text))

    if parts:
        return "\n".join(parts)

    sentences = content_data.get("sentences")
    if isinstance(sentences, list):
        for sentence in sentences:
            if isinstance(sentence, dict):
                text = sentence.get("text") or sentence.get("raw_text")
            else:
                text = str(sentence)
            if text:
                parts.append(str(text))

    return "\n".join(parts) if parts else str(content_data)


def get_codetaxo_prompt(text_chunk: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": """
You are a Geological Ontology Architect.
Task: Extract geological **Concepts (Classes)** from the text and organize them into a **Python Class Hierarchy**.

**SEED CLASSES (Roots):**
class GeologicalObject: pass
class Rock(GeologicalObject): pass
class Mineral(GeologicalObject): pass
class Alteration(GeologicalObject): pass
class Structure(GeologicalObject): pass
class DepositType(GeologicalObject): pass
class Process(GeologicalObject): pass

**CONSTRAINTS:**
1. **NO INSTANCES:** Do NOT create classes for specific mine names (e.g., 'Dexing', 'Bingham') or locations.
2. **STRICT HIERARCHY:** Use `class Child(Parent): pass` syntax.
3. **OUTPUT:** Output ONLY valid Python code. No markdown.
""".strip(),
        },
        {"role": "user", "content": f"Generate Taxonomy Code for:\n{text_chunk}"},
    ]


def strip_code_fence(content: str) -> str:
    return content.replace("```python", "").replace("```", "").strip()


def safe_class_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^A-Za-z0-9_]", "_", ascii_value)
    ascii_value = re.sub(r"_+", "_", ascii_value).strip("_")
    if not ascii_value:
        return ""
    if ascii_value[0].isdigit():
        ascii_value = f"Class_{ascii_value}"
    return ascii_value


def normalize_taxonomy_code(content: str) -> str:
    lines = strip_code_fence(content).splitlines()
    normalized = []
    class_header = re.compile(r"^class\s+([^\(:\s]+)\s*(?:\(([^)]*)\))?\s*:\s*(?:pass)?\s*(?:#.*)?$")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "pass" or stripped.startswith("#"):
            continue

        match = class_header.match(stripped)
        if not match:
            continue

        child = safe_class_name(match.group(1))
        parent_text = match.group(2)
        if not child:
            continue

        if parent_text:
            first_parent = parent_text.split(",", 1)[0].strip()
            parent = safe_class_name(first_parent)
            if parent and parent != child:
                normalized.append(f"class {child}({parent}): pass")
        elif child in SEED_CLASSES:
            normalized.append(f"class {child}: pass")
        else:
            normalized.append(f"class {child}(GeologicalObject): pass")

    return "\n".join(dict.fromkeys(normalized))


def extract_taxonomy(client: OpenAI, model_name: str, text: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=get_codetaxo_prompt(text),
        temperature=0.0,
        stream=False,
    )
    return normalize_taxonomy_code(response.choices[0].message.content or "")


def extract_taxonomy_with_retries(
    client: OpenAI,
    model_name: str,
    text: str,
    attempts: int,
    retry_sleep: float,
) -> str:
    last_error = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return extract_taxonomy(client, model_name, text)
        except Exception as exc:
            last_error = exc
            if is_non_retriable_error(exc):
                raise exc
            if attempt >= attempts:
                break
            sleep_for = retry_sleep * attempt
            print(f"[WARN] Request failed ({attempt}/{attempts}); retrying in {sleep_for:.1f}s: {exc}")
            time.sleep(sleep_for)
    raise last_error


def is_non_retriable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    non_retriable_markers = [
        "insufficient balance",
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
    ]
    return any(marker in message for marker in non_retriable_markers)


def output_path_for_model(args: argparse.Namespace, config: ModelConfig, total_models: int) -> Path:
    if args.output_file and total_models == 1:
        return Path(args.output_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output_file).stem if args.output_file else "raw_taxonomy_100"
    return output_dir / f"{stem}_{config.alias}.py"


def write_header(handle, config: ModelConfig) -> None:
    handle.write(f"# === TAXONOMY EXTRACTED BY {config.alias} ===\n")
    handle.write(f"# model: {config.model}\n")
    handle.write(f"# base_url: {config.base_url}\n\n")
    handle.write("class GeologicalObject: pass\n")
    handle.write("class Rock(GeologicalObject): pass\n")
    handle.write("class Mineral(GeologicalObject): pass\n")
    handle.write("class Alteration(GeologicalObject): pass\n")
    handle.write("class Structure(GeologicalObject): pass\n")
    handle.write("class DepositType(GeologicalObject): pass\n")
    handle.write("class Process(GeologicalObject): pass\n\n")


def processed_source_files(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    processed = set()
    source_pattern = re.compile(r"^# Source:\s+(.+?)\s*$")
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = source_pattern.match(line)
            if match:
                processed.add(match.group(1))
    return processed


def run_for_model(config: ModelConfig, files: list[str], args: argparse.Namespace) -> Path:
    output_path = output_path_for_model(args, config, len(args.model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(".errors.jsonl")
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    already_done = processed_source_files(output_path) if args.resume else set()
    pending_files = [path for path in files if os.path.basename(path) not in already_done]
    output_exists = output_path.exists() and output_path.stat().st_size > 0

    print(f"\n[MODEL] {config.alias} -> {output_path}")
    if args.resume and already_done:
        print(f"[INFO] Resume enabled: skipping {len(already_done)} completed source file(s).")
    print(f"[INFO] Pending files for {config.alias}: {len(pending_files)}")

    output_mode = "a" if args.resume and output_exists else "w"
    error_mode = "a" if args.resume else "w"
    with open(output_path, output_mode, encoding="utf-8") as out, open(error_path, error_mode, encoding="utf-8") as err:
        if output_mode == "w":
            write_header(out, config)

        for filepath in tqdm(pending_files, desc=f"Mining:{config.alias}"):
            text = parse_json_content(filepath, args.max_chars)
            if not text:
                continue

            try:
                code = extract_taxonomy_with_retries(
                    client=client,
                    model_name=config.model,
                    text=text,
                    attempts=args.retry_attempts,
                    retry_sleep=args.retry_sleep,
                )
            except Exception as exc:
                err.write(
                    json.dumps(
                        {
                            "source_file": os.path.basename(filepath),
                            "model": config.model,
                            "alias": config.alias,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err.flush()
                continue

            if code:
                out.write(f"\n# Source: {os.path.basename(filepath)}\n")
                out.write(code + "\n")
                out.flush()

    if error_path.stat().st_size == 0:
        error_path.unlink()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract geological taxonomy hierarchy from corpus_json with one or more OpenAI-compatible models."
    )
    parser.add_argument("--input-dir", default="corpus_json", help="Directory containing processed JSON papers.")
    parser.add_argument("--file-glob", default="*.json", help="Input filename glob within --input-dir.")
    parser.add_argument("--output-file", default="raw_taxonomy_100.py", help="Output file for single-model runs.")
    parser.add_argument("--output-dir", default="taxonomy_runs", help="Output directory for multi-model runs.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters from each JSON paper sent to model.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of files to process for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output file and skip completed sources.")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Request attempts per paper before logging an error.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Base seconds to sleep between retries.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for splitting large corpora.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shard count for splitting large corpora.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec. Repeat for multi-model runs. Format: "
            "MODEL[@ALIAS][|BASE_URL][|API_KEY_ENV]. "
            "Example: deepseek-v4-flash@deepseek|https://api.deepseek.com|DEEPSEEK_API_KEY"
        ),
    )
    return parser


def main() -> int:
    load_dotenv_file()
    args = build_parser().parse_args()
    if not args.model:
        args.model = [os.getenv("MODEL_NAME", "deepseek-v4-flash")]

    files = sorted(glob.glob(os.path.join(args.input_dir, args.file_glob)))
    if args.limit > 0:
        files = files[: args.limit]

    if args.shard_count < 1:
        print("[ERROR] --shard-count must be >= 1.")
        return 1
    if not 0 <= args.shard_index < args.shard_count:
        print("[ERROR] --shard-index must satisfy 0 <= shard-index < shard-count.")
        return 1
    if args.shard_count > 1:
        files = [path for idx, path in enumerate(files) if idx % args.shard_count == args.shard_index]

    if not files:
        print(f"[ERROR] No .json files found in {args.input_dir!r}.")
        return 1

    configs = [parse_model_spec(spec, args.base_url, args.api_key_env) for spec in args.model]

    print(f"[INFO] Found {len(files)} JSON papers after limit/shard filtering.")
    print(f"[INFO] Running {len(configs)} model(s): {', '.join(c.alias for c in configs)}")

    outputs = []
    for config in configs:
        outputs.append(run_for_model(config, files, args))

    print("\n[DONE] Taxonomy extraction finished.")
    for path in outputs:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
