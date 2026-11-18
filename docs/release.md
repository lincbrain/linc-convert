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

- Add the `release` label to a pull request before merging it. The `Auto-release on PR merge` workflow then releases it, together with any unreleased pull requests merged before it. A release is only made if at least one unreleased pull request has a label that bumps the version (`major`, `minor`, `patch` or `performance`), or if the most recently merged pull request has none of the labels in the table, which counts as `patch`.
- Or merge pull requests without the `release` label and later run the `Auto-release on PR merge` workflow by hand from the [Actions](https://github.com/lincbrain/linc-convert/actions/workflows/release.yml) tab (`Run workflow`). It releases everything merged since the last release, with or without the `release` label.

## One-time setup

This step needs a repository administrator.

1. **Deploy key.** The ruleset on `main` requires changes to go through pull requests, so the release workflow pushes the `CHANGELOG.md` commit and the tag with a deploy key that is allowed to bypass it:
    - Create an SSH key pair, for example `ssh-keygen -t ed25519 -C "linc-convert release" -N "" -f release_key`.
    - Add the public key (`release_key.pub`) as a deploy key with write access (Settings → Deploy keys → Add deploy key, and tick "Allow write access").
    - Add the private key (`release_key`) as the Actions secret `RELEASE_DEPLOY_KEY` (Settings → Secrets and variables → Actions).
    - In the ruleset for `main` (Settings → Rules → Rulesets), add "Deploy keys" to the bypass list, set to always allow.
    - Delete both key files from your computer.
