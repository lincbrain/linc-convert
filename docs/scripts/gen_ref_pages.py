"""Generate API reference pages and a literate navigation file.

This is used by `mkdocs-gen-files` + `mkdocs-literate-nav` to auto-generate
API documentation pages at build time.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent.parent
src = root / "linc_convert"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        # Bind package docs to the section index page (requires mkdocs-section-index).
        parts = parts[:-1]
        doc_path = Path("api", *module_path.parts[:-1], "index.md")
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = Path("api", *module_path.parts).with_suffix(".md")

    if parts:
        # `mkdocs.yml` already creates the top-level "API" section via `- API: api/`,
        # so don't add another redundant root level here.
        nav[parts] = doc_path.relative_to("api").as_posix()

        # mkdocs-gen-files expects paths relative to the MkDocs `docs_dir`.
        with mkdocs_gen_files.open(doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: linc_convert.{ident}")

        # `edit_uri` points at `docs/`, but our sources live outside `docs/`,
        # so we need to traverse up one level.
        mkdocs_gen_files.set_edit_path(doc_path, Path("..") / path.relative_to(root))

with mkdocs_gen_files.open(root / "docs/summary.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
