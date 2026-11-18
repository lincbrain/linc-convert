# Release process

Releases of `linc-convert` are automated with [auto](https://intuit.github.io/auto/). Each release:

1. Picks the next version from the labels of the pull requests merged since the last release.
2. Updates `CHANGELOG.md` and tags the commit with the version, without a `v` prefix (e.g. `0.1.0`).
3. Builds the package and publishes it to [PyPI](https://pypi.org/project/linc-convert/).
4. Creates a [GitHub release](https://github.com/lincbrain/linc-convert/releases) and comments on the released pull requests.
5. Zenodo archives the GitHub release and mints a DOI for it, using the title, abstract, authors, keywords and license in `CITATION.cff`.

## Label pull requests

Every pull request needs one of the labels below. The `Check PR Labels` workflow fails until one is added.

| Label           | Changelog section    | Version bump |
|-----------------|----------------------|--------------|
| `major`         | Breaking Change      | major        |
| `minor`         | Enhancement          | minor        |
| `patch`         | Bug Fix              | patch        |
| `performance`   | Performance          | patch        |
| `internal`      | Internal             | none         |
| `documentation` | Documentation        | none         |
| `tests`         | Tests                | none         |
| `dependencies`  | Dependency Updates   | none         |

Two more labels control when a release happens:

- `release`: publish a release as soon as this pull request is merged.
- `skip-release`: never publish a release for this pull request.

## Make a release

Either:

- Add the `release` label to a pull request before merging it. Merging it releases it together with any earlier unreleased pull requests, as long as at least one of them is labeled `major`, `minor`, `patch` or `performance`. If the last pull request merged has none of the labels in the table, it counts as `patch`.
- Or merge pull requests without the `release` label and later run the `Auto-release on PR merge` workflow by hand from the [Actions](https://github.com/lincbrain/linc-convert/actions/workflows/release.yml) tab (`Run workflow`). It releases everything merged since the last release, with or without the `release` label.
