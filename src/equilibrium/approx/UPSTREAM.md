# funcapprox upstream provenance

The `equilibrium.approx` implementation is derived from the local
`funcapprox` project maintained by Daniel Greenwald.

## Baseline

- Source checkout: `~/dev/funcapprox`
- Source package version: `0.1.0`
- Source commit: `4efed5bb24c78c9196f69f99ead7b9744ec63977`
- Commit date: 2026-01-06T15:04:01-05:00
- Commit subject: `adding modified hat basis function`
- Port baseline recorded: 2026-08-28
- Upstream worktree at baseline: clean
- License confirmed by the author: GNU General Public License v3, consistent
  with Equilibrium's GPL-3.0-only license

The upstream `pyproject.toml` declared MIT at this commit, while its `LICENSE`
file contained the GPLv3 text. The author confirmed GPL for this port. This
record intentionally follows that confirmation and the checked-in license
text, rather than the stale package metadata.

## Included runtime files

The following upstream files form the planned namespace-only port. SHA-256
hashes describe the unmodified files at the baseline commit; expected port
changes such as import rewriting and formatting will change them.

| Upstream file | SHA-256 |
|---|---|
| `src/funcapprox/__init__.py` | `6e414f8d23c8a387bf2932aaeb87f1994328a1f5dca0f5917a5151728e0ac4c9` |
| `src/funcapprox/presets.py` | `67c6504ca5afdbbd30a38552b54280627b90ce5c740a90d93d84f4105e91947c` |
| `src/funcapprox/py.typed` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/funcapprox/bases/__init__.py` | `bf7376dd8dd7b4e32a370c518108743cf9f4f08089daaf0a24e630d5aa4ff738` |
| `src/funcapprox/bases/base.py` | `28a4a1c08572c5ce40c1af08db9bd90130b607df5d07f363426f228408786978` |
| `src/funcapprox/bases/chebyshev.py` | `d43b001bfb97a0d6ccbc741e4d908642bea5070bfa15286c3f702e0f4ae6b91a` |
| `src/funcapprox/bases/hat.py` | `34e0f5b7b1ada59225023bb7cdff0842a46ddf32ec431d42c5c761ee834ba7ae` |
| `src/funcapprox/grids/__init__.py` | `e7b8c3b35f53ae1d90e87b6ae099e2715b074e298daf98787cfdec09a7b71b2c` |
| `src/funcapprox/grids/base.py` | `f535cbf96c2d46b4f4c66ce829795f30975c92447259c0614bc7d943e1807c40` |
| `src/funcapprox/grids/chebyshev.py` | `083170ab22e029cfedd4a666458ad06ed17695427ccdbefca439edada7367064` |
| `src/funcapprox/grids/uniform.py` | `e478b47c95669feeec591266d4d5d05371770c8f4133b0ddff59dd7e130e8ac7` |
| `src/funcapprox/levels/__init__.py` | `f40a8f0cdadf90a937143a830fd39674fea6723bc8972fe715e170384d91e120` |
| `src/funcapprox/levels/base.py` | `53abeeb3e12184ea0ff54f46674ad2408570bc95769f124413bffb161103ea07` |
| `src/funcapprox/levels/smolyak.py` | `8e3f743450072777705a44a86c23acab598c98200917e5ed6fbe93d5d6284807` |
| `src/funcapprox/levels/tensor.py` | `653b078c93c736e384ec5589b0cf003c02793d63157612c7208c08ff964b9140` |
| `src/funcapprox/core/__init__.py` | `9501acf462960ab9dfd8697fc547b27dafc89cbdd44b7820f0a29a4c87d894ad` |
| `src/funcapprox/core/index.py` | `c846de4ce623eca3b1fcb1ae67399dac0164c70088a847951db26ade3a465a5d` |
| `src/funcapprox/core/scheme.py` | `7f3adeeaa187bcd38b5c8222982664e4401f53e77a1f2a9d027f410b0142e0fd` |
| `src/funcapprox/core/function.py` | `a32e004a6a0d5d72411782662ddb096db765d926282245e89cc208fb53df10bc` |

## Exclusions

The port excludes `src/funcapprox/benchmark/`, benchmark tests,
`tests/test_functions.py`, examples, generated package metadata, caches, and
repository configuration. These are research and development aids rather than
runtime approximation infrastructure.

## Baseline validation

The seven non-benchmark upstream test modules passed before the port:

```text
Environment: Python 3.13.5, pytest 8.4.1
Result: 146 passed in 0.29s
```

The command used the upstream `src` layout without installing or modifying the
checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -p no:cacheprovider \
  tests/test_bases.py tests/test_function.py tests/test_grids.py \
  tests/test_index.py tests/test_levels.py tests/test_presets.py \
  tests/test_scheme.py
```

The same seven modules were ported to Equilibrium in work package 3 with their
assertions and tolerances unchanged. All 146 behavioral tests passed in the
Equilibrium namespace. Three additional package-level tests cover the public
export set, benchmark exclusion, and external-import isolation, giving 149
passing approximation tests after the port.

## Equilibrium-specific changes

Work package 2 made the following changes from the upstream baseline:

1. Change the package namespace from `funcapprox` to `equilibrium.approx`.
2. Remove the top-level benchmark export.
3. Format the port to Equilibrium's Black and Ruff configuration.
4. Keep grid construction, fitting, and the compatibility API NumPy-based.

5. Add a stateless, Chebyshev-first JAX evaluation layer in `jax_eval.py`.
   Coefficients are explicit traced inputs; immutable scheme arrays and static
   shape metadata are carried by a dedicated JAX PyTree. Hat schemes remain
   available through NumPy and are rejected by the JAX adapter.

## Future synchronization procedure

1. Record the candidate upstream commit and verify its worktree state.
2. Compare its runtime manifest against the baseline listed above.
3. Review upstream changes before applying them; do not overwrite
   Equilibrium-specific JAX code mechanically.
4. Port namespace-neutral fixes into `equilibrium.approx` with focused tests.
5. Update this file with the new commit, hashes, material changes, and upstream
   baseline test result.
