# Contributing to Tesseract Core

Tesseract Core is an open-source project and, as such, we welcome contributions
from developers, engineers, scientists, and end-users in general. Contributions
are what make the open source community such an amazing place to learn,
inspire, and create. Any contributions you make are greatly appreciated.


## Code of Conduct

Ensure your contributions adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).


## Feedback

Constructive feedback is very welcome. We are interested in hearing from you!

In the case things aren't working as expected, or the documentation is lacking,
please [file a bug
report](https://github.com/pasteurlabs/tesseract-core/issues/new?template=BUG-REPORT.yml).

In the case you want to suggest a new feature, please file a new [feature
request](https://github.com/pasteurlabs/tesseract-core/issues/new?template=FEATURE-REQUEST.yml).
In particular, we recommend you open an issue before contributing code in a
pull request. This allows all parties to talk things over before jumping into
action, and increase the likelihood of pull requests getting merged.

In case you have general questions or feedback, need support from the
community, or have a cool demo to share, start a thread in our [Discourse
Forum](https://si-tesseract.discourse.group/). We use GitHub Issues for bug
reports and feature requests only.


## Documentation

Tesseract documentation is kept under the `docs/` directory of the repository,
written in Markdown and using Sphinx to generate the final HTMLs. Fixes and
enhancements to the documentation should be submitted as pull requests, we
treat the same as code contributions.

To build the documentation locally, install the documentation dependencies in
addition to the project itself, then run `make html`:

```console
$ . venv/bin/activate
$ pip install -e .[dev]
$ pip install -r docs/requirements.txt
$ cd docs
$ make html
```

The resulting HTMLs are in `docs/build/html/`.

Contributions in the form of tutorials, examples, demos, blog posts (including
those posted elsewhere already) are best highlighted and celebrated in the
[Discourse Forum](https://si-tesseract.discourse.group/).


## Code

Tesseract is developed under the [Apache 2.0](LICENSE) license. By contributing
to the Tesseract project you agree that your code contributions are governed by
this license.


### Local development setup

Make sure you have [Docker installed](https://docs.docker.com/engine/install/)
on your machine and you can run `docker` commands via your user. After that,
clone the repository, install the dependencies, and setup pre-commit hooks:

```console
$ git clone git@github.com:pasteurlabs/tesseract-core.git
$ cd tesseract
$ python -m venv venv
$ . venv/bin/activate
$ pip install -e .[dev]
$ pre-commit install
```

### Tests

This project uses the pytest framework for all tests. New code should be
covered by new or existing tests.

To run the tests simply run `pytest` in the root of the project. This will run
the entire test suite, including the end-to-end tests that take quite a while
to finish. Instead, you can run the tests separately:

```console
$ pytest --skip-endtoend
$ pytest --always-run-endtoend tests/endtoend_tests
```

### GitHub workflow

This project uses Git for version control and follows a GitHub workflow. To
contribute follow these steps:

1. Fork the project via the GitHub UI.
1. Clone your fork to your machine.
1. Add an upstream remote: `git remote add upstream git@github.com:pasteurlabs/tesseract-core.git`.
1. Create a new branch for your code contribution: `git switch --create my_branch`.
1. Implement your changes.
1. Commit and push to your fork: `git push --set-upstream origin my_branch`.
1. [Open a Pull Request](https://github.com/pasteurlabs/tesseract-core/pulls) with
   your changes.

It is a good practice to rebase often on top of `main` to keep your code up to
date with latest development and minimize merge conflicts:

```console
$ git fetch upstream
$ git switch main
$ git merge upstream/main
$ git switch my_branch
$ git rebase main
$ git push --force
```

### Commit and pull request messages guidelines

We follow the [Conventional
Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all
commits that reach the `main` branch. Each commit is crafted from a pull
request that is squash-merged. The commit title and message comes from the pull
request title and message, respectively. As such, they should be structured
following the specfication.

The title consists of a _type_, and optional _scope_, and a short
_description_: `type[(scope)]: description`. The types we use are:
- `chore`: for changes that affect the build system, external dependencies, or
  general housekeeping.
- `ci`: for changes in the CI.
- `doc`: for documentation only changes.
- `feat`: for a new feature.
- `fix`: for fixing a bug.
- `perf`: for a code change that improves performance.
- `refactor`: for a code change that neither adds a feature nor fixes a bug.
- `security`: for a change that fixes a security issue.
- `test`: for adding new tests or fixing existing ones.

The scopes we use are:
- `cli`: for changes that affect `tesseract` CLI.
- `engine`: for changes that affect the CLI engine.
- `sdk`: for changes that affect the Tesseract Python API.
- `example`: for changes in the examples.
- `runtime`: for changes in the Tesseract Runtime.
- `deps`: for changes in the dependencies.

In case there are breaking changes in your code, this should be indicated in
the message either by appending an exclamation mark (`!`) after the type/scope
or by adding a `BREAKING CHANGE:` trailer to the message.


## Versioning

The Tesseract project follows [semantic versioning](https://semver.org).


## Changelog

This project [changelog](CHANGELOG.md) is generated by
[git-cliff](https://git-cliff.org). The changelog is generated by running
`git cliff --output CHANGELOG.md`. To bump the semantic version of the project
when generating the changelog: `git cliff --output CHANGELOG.md --bump`.
