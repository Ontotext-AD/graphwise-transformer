from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import os
import sys
import re
import glob


def _cleanup_generated_protos(out_dir: str) -> None:
    """Remove previously generated proto files before regenerating."""
    patterns = [
        os.path.join(out_dir, "transformer_pb2.py"),
        os.path.join(out_dir, "transformer_pb2_grpc.py"),
        os.path.join(out_dir, "*_pb2.py"),
        os.path.join(out_dir, "*_pb2_grpc.py"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass  # Ignore if already gone


def generate_protos() -> None:
    try:
        from grpc_tools import protoc
    except Exception as exc:
        print("grpcio-tools is required to generate protos:", exc)
        sys.exit(1)

    root_dir = os.path.dirname(__file__)
    proto_dir = os.path.join(root_dir, "protos")
    out_dir = os.path.join(root_dir, "graphwise_transformer", "proto")
    os.makedirs(out_dir, exist_ok=True)

    # Clean up old generated files first
    _cleanup_generated_protos(out_dir)

    proto_file = os.path.join(proto_dir, "transformer.proto")
    if not os.path.exists(proto_file):
        print(f"Proto file not found: {proto_file}")
        sys.exit(1)

    args = [
        "protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        proto_file,
    ]
    if protoc.main(args) != 0:
        print("Error: proto generation failed")
        sys.exit(1)

    _write_proto_init(out_dir)
    _patch_generated_imports(out_dir)


def _write_proto_init(out_dir: str) -> None:
    init_path = os.path.join(out_dir, "__init__.py")
    content = (
        "# Auto-generated package init to ensure local imports work for generated modules\n"
        "# Do not edit by hand; regenerated on build.\n"
    )
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(content)


def _patch_generated_imports(out_dir: str) -> None:
    targets = [
        os.path.join(out_dir, "transformer_pb2_grpc.py"),
        os.path.join(out_dir, "transformer_pb2.py"),
    ]
    pat_simple = re.compile(r"^\s*import\s+transformer_pb2(\s+as\s+transformer__pb2)?\s*$")
    for path in targets:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        changed = False
        for i, line in enumerate(lines):
            if pat_simple.match(line) and "from ." not in line:
                # preserve alias if present
                if " as transformer__pb2" in line:
                    lines[i] = "from . import transformer_pb2 as transformer__pb2\n"
                else:
                    lines[i] = "from . import transformer_pb2\n"
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)


class build_py(_build_py):
    def run(self):
        generate_protos()
        super().run()


if __name__ == "__main__":
    # Explicitly list packages to avoid setuptools warnings about proto subpackage
    packages = find_packages(include=["graphwise_transformer", "graphwise_transformer.*"])
    # Ensure proto package is included if it exists
    if "graphwise_transformer.proto" not in packages:
        proto_pkg_path = os.path.join(os.path.dirname(__file__), "graphwise_transformer", "proto", "__init__.py")
        if os.path.exists(proto_pkg_path):
            packages.append("graphwise_transformer.proto")
    
    setup(
        packages=packages,
        include_package_data=True,
        cmdclass={"build_py": build_py},
    )
