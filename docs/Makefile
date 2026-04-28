# Minimal Makefile for Sphinx
SPHINXBUILD  = python -m sphinx
SOURCEDIR    = .
BUILDDIR     = _build
# Rust crate root (one directory up from docs/)
CARGO_ROOT   = ..
CARGO_DOC    = $(CARGO_ROOT)/target/doc
RUST_OUT     = $(BUILDDIR)/html/rust

.PHONY: help html html-rust clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

# Build the Sphinx site *and* embed the Rust crate documentation under
# _build/html/rust/.  The rust_api page links to rust/spectroxide/index.html.
# If `cargo` is not on PATH, the rust step is skipped with a warning so that
# the Python-only build still succeeds.
html: sphinx html-rust

sphinx:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -q

html-rust:
	@if command -v cargo >/dev/null 2>&1; then \
		echo "Building rustdoc into $(RUST_OUT)"; \
		cargo doc --manifest-path $(CARGO_ROOT)/Cargo.toml --lib --no-deps --quiet; \
		mkdir -p "$(RUST_OUT)"; \
		cp -R "$(CARGO_DOC)"/. "$(RUST_OUT)"/; \
	else \
		echo "warning: cargo not found on PATH; skipping rustdoc step"; \
		echo "  install Rust via https://rustup.rs/ to include the Rust API"; \
	fi

clean:
	rm -rf "$(BUILDDIR)"
