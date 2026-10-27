"""Verify that reformatting changed no code: compare the AST of every file against a git revision.

Docstrings are compared after normalising whitespace (formatters may re-indent them). Usage:
    python tools/ast_equal.py [REV] [paths...]      (default REV = HEAD, paths = src/MIR)
    python tools/ast_equal.py --dir REFDIR NEWDIR [FILES...]   (compare two trees, optionally only FILES)
    add --code-only to ignore docstrings and type annotations (for documentation/typing-only edits)
"""

import os
import ast
import subprocess
import sys


CODE_ONLY = False


def _strip(tree):
    """Remove docstrings and type annotations (code-only comparison)."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
    for node in ast.walk(tree):  # x: T = v  ->  x = v ;  bare "x: T" -> pass
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                new = []
                for st in stmts:
                    if isinstance(st, ast.ImportFrom) and st.module == "typing":
                        continue  # imports needed only for annotations
                    if isinstance(st, ast.AnnAssign):
                        if st.value is not None:
                            new.append(ast.Assign(targets=[st.target], value=st.value, type_comment=None))
                    else:
                        new.append(st)
                setattr(node, field, new or [ast.Pass()])
    return tree


def norm(tree):
    if CODE_ONLY:
        return ast.dump(_strip(tree), include_attributes=False)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0].value.value = " ".join(body[0].value.value.split())
    return ast.dump(tree, include_attributes=False)


def compare_dirs(ref, new, only=None):
    bad = 0
    for root, _, files in os.walk(new):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            if only and not any(os.path.abspath(p) == os.path.abspath(o) for o in only):
                continue
            r = os.path.join(ref, os.path.relpath(p, new))
            if not os.path.exists(r):
                print("new file:", p)
                continue
            if norm(ast.parse(open(r).read())) != norm(ast.parse(open(p).read())):
                print("AST differs:", p)
                bad += 1
    print(f"{bad} file(s) with code changes")
    return 1 if bad else 0


def main():
    global CODE_ONLY
    args = sys.argv[1:]
    if "--code-only" in args:
        CODE_ONLY = True
        args.remove("--code-only")
    if args and args[0] == "--dir":
        return compare_dirs(args[1], args[2], args[3:])
    sys.argv = [sys.argv[0]] + args
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    paths = sys.argv[2:] or ["src/MIR"]
    files = subprocess.run(["git", "ls-files", *paths], capture_output=True, text=True).stdout.split()
    bad = 0
    for f in files:
        if not f.endswith(".py"):
            continue
        try:
            old = subprocess.run(["git", "show", f"{rev}:{f}"], capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError:
            continue
        new = open(f).read()
        if norm(ast.parse(old)) != norm(ast.parse(new)):
            print("AST differs:", f)
            bad += 1
    print(f"{bad} file(s) with code changes")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
