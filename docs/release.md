# Release process

Releases of `linc-convert` are automated with [auto](https://intuit.github.io/auto/). Each release:

1. Picks the next version from the labels of the pull requests merged since the last release.
2. Updates `CHANGELOG.md` and tags the commit with the version, without a `v` prefix (e.g. `0.1.0`).
3. Builds the package and publishes it to [PyPI](https://pypi.org/project/linc-convert/).
4. Creates a [GitHub release](https://github.com/lincbrain/linc-convert/releases) and comments on the released pull requests.
5. Zenodo archives the GitHub release and mints a DOI for it, using the title, abstract, authors, keywords and license in `CITATION.cff`.

`CITATION.cff` also gives the repository's "Cite this repository" button on GitHub. To add or change an author, edit `CITATION.cff` (and `authors` in `pyproject.toml` for PyPI). Leave out `version` and `date-released`: releases don't update them, and Zenodo takes the version from the tag.

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

- Add the `release` label to a pull request before merging it. The `Auto-release on PR merge` workflow then releases it, together with any unreleased pull requests merged before it.
- Or merge pull requests without the `release` label and later run the `Auto-release on PR merge` workflow by hand from the [Actions](https://github.com/lincbrain/linc-convert/actions/workflows/release.yml) tab (`Run workflow`). It releases everything merged since the last release, with or without the `release` label.

A release is only made if at least one unreleased pull request has a label that bumps the version (`major`, `minor`, `patch` or `performance`).

!!! warning "First release"
    Version `0.0.1` of `linc-convert` was uploaded to PyPI before this process was set up, and the repository has no version tags yet. `auto` starts counting from `0.0.0`, so the first release must be labeled `minor` (giving `0.1.0`) or `major`. A `patch` release would be `0.0.1`, which PyPI refuses because that version already exists.

## One-time setup

This step needs a repository administrator.

1. **Protected branch.** The release workflow pushes the `CHANGELOG.md` commit and the tag to `main` with the workflow's own `GITHUB_TOKEN`. If `main` requires pull request reviews, also:
    - Create a personal access token for a bot or maintainer account that has write access to the repository, and add it as the Actions secret `GH_TOKEN`. The `protected-branch` plugin uses it to approve a temporary pull request holding the release commit, which lets the push through.
    - Turn on "Allow GitHub Actions to create and approve pull requests" (Settings → Actions → General).
    - If `main` also requires status checks, list them under the `requiredStatusChecks` option of the `protected-branch` plugin in `.autorc`.

    Without the `GH_TOKEN` secret, the plugin logs a warning and does nothing, which is fine while `main` is not protected.
