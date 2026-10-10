"""Stdlib-only interpreter entry point, executed INSIDE the Python container."""

import ast
import json
import os
import sys
import threading
import traceback


class Capture:
    """Drain an OS output pipe while retaining only a bounded prefix."""

    def __init__(self, fd, limit):
        self.fd, self.limit = fd, limit
        self.data = bytearray()
        self.truncated = False

    def __enter__(self):
        self.saved = os.dup(self.fd)
        reader, writer = os.pipe()
        os.dup2(writer, self.fd)
        os.close(writer)

        def drain():
            with os.fdopen(reader, "rb", buffering=0) as handle:
                while chunk := handle.read(8192):
                    remaining = self.limit - len(self.data)
                    self.data.extend(chunk[:remaining])
                    self.truncated |= len(chunk) > remaining

        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        os.dup2(self.saved, self.fd)
        os.close(self.saved)
        self.thread.join()


def clipped(text, limit):
    raw = text.encode("utf-8", errors="replace")
    return raw[:limit].decode("utf-8", errors="replace"), len(raw) > limit


def execute(code, namespace, limit):
    result = error = None
    truncated = False
    stdout, stderr, stdin = sys.stdout, sys.stderr, sys.stdin
    sys.stdout.flush()
    sys.stderr.flush()
    # Input belongs to the request protocol; snippets cannot consume the next request.
    saved_input = os.dup(0)
    with open(os.devnull) as empty:
        os.dup2(empty.fileno(), 0)
        try:
            with Capture(1, limit) as out, Capture(2, limit) as err:
                try:
                    tree = ast.parse(code, filename="<python-tool>", mode="exec")
                    expression = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                    exec(compile(tree, "<python-tool>", "exec"), namespace)  # noqa: S102 - Interpreter runs inside container.
                    if expression is not None:
                        value = eval(compile(ast.Expression(expression.value), "<python-tool>", "eval"), namespace)
                        if value is not None:
                            result, truncated = clipped(repr(value), limit)
                except BaseException:  # noqa: BLE001 - Includes snippets raising SystemExit/KeyboardInterrupt.
                    error, cut = clipped(traceback.format_exc(), limit)
                    truncated |= cut
                finally:
                    sys.stdout, sys.stderr, sys.stdin = stdout, stderr, stdin
                    stdout.flush()
                    stderr.flush()
        finally:
            os.dup2(saved_input, 0)
            os.close(saved_input)
    return {
        "stdout": out.data.decode("utf-8", errors="replace"),
        "stderr": err.data.decode("utf-8", errors="replace"),
        "result": result,
        "error": error,
        "truncated": truncated or out.truncated or err.truncated,
    }


def main():
    # Separate handles protect the protocol from ordinary print/os.write/subprocess output.
    with os.fdopen(os.dup(0), encoding="utf-8") as requests, os.fdopen(os.dup(1), "w", encoding="utf-8") as replies:
        replies.write('{"ready":true}\n')
        replies.flush()
        namespace = {"__name__": "__main__", "__builtins__": __builtins__}
        for line in requests:
            request = json.loads(line)
            response = execute(request["code"], namespace, request["max_output_bytes"])
            replies.write(json.dumps({"id": request["id"], **response}, ensure_ascii=True) + "\n")
            replies.flush()


if __name__ == "__main__":
    main()
