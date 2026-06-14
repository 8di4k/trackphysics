# Contributing to trackphysics

Thanks for contributing! A few things keep this project trustworthy and easy to adopt.

## License & CLA

- The project is licensed under **Apache-2.0** (see [LICENSE](LICENSE) / [NOTICE](NOTICE)).
- Contributions require signing a lightweight **[Contributor License Agreement](CLA.md)**.
  The CLA keeps the project's licensing options open (including a possible future
  dual-license of the core) while everything you use today stays permissive Apache-2.0.
  In practice this is enforced automatically on pull requests (e.g. via a CLA bot).
- Please also sign off your commits (DCO): `git commit -s`.

## The one rule that must never break: a domain-agnostic core

The engine contains **zero** domain semantics (no `ball`/`player`/`court`/`sport`/etc.).
Domain knowledge enters only through the documented hooks (presets, the grounding slot, the
skeleton graph). A CI test (`tests/test_no_domain_terms.py`) greps the package for a
denylist and fails the build on any hit — see `BRIEF.md` §6. Never weaken it.

Two more invariants, also from `BRIEF.md`:

- **Provenance-first (§10):** no public function returns a bare physical float — wrap it in
  a `Quantity` with a tier and a *derived* confidence. Never emit `Tier.METRIC` without
  genuinely earned scale; fall back honestly.
- **No feature without a consumer (§16):** every capability is exercised by the benchmark,
  the hero demo, or a test.

## Local checks (all must pass)

```bash
pip install -e ".[dev,bench]"
python -m ruff check src bench tests examples
python -m mypy            # mypy --strict on the package
python -m pytest          # unit + integration + the domain-guardrail test
python -m bench.run       # regenerate the benchmark report when physics/robustness changes
```

New physics/robustness code lands **with a measurement in the benchmark** (`BRIEF.md`
§16): "it looks right" is not acceptance — a number is.

## Scope

This repo is the object-agnostic physics engine only. Domain layers (sports, traffic, …)
are separate projects built on top — please keep domain logic out of `core/`.
