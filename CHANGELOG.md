# Changelog

All notable changes to this project will be documented in this file.

## [1.1.1] - 2025-10-15

### Bug Fixes

- Create tesseract user and group (#369)
- Avoid a potential feedback loop in output redirection (#367)

## [1.1.0] - 2025-09-12

### Features

- Add run_id parameter to Tesseract.apply and friends (#352)
- Add pydantic type making other Tesseracts callable (needed for HOTs) (#343)

### Bug Fixes

- Require name and version to be non-empty in TesseractConfig (#336)
- Strip trailing slash from URL's passed to Tesseract HTTP client (#340)
- Handling of tree_transforms edge cases (#331)
- *(sdk)* Cast sets to tuples in Pytorch template when used as positional args (#347)
- Use host port as api port (#330)
- Allow digits in the pre-release semver tag (#355)

### Refactor

- [**breaking**] Use uv to install python virtual env (#353)

### Testing

- Runtime/file_interactions unit tests (#338)

## [1.0.0] - 2025-08-08

### Features

- [**breaking**] Automatically redirect stdout + stderr to logfile within Tesseract endpoints (#265)
- Add e2e tests for MPA (both file and MLflow backend) (#277)
- [**breaking**] Use `version` from `tesseract_config.yaml` as default Docker image tag (#267)
- Add network argument to `tesseract serve` (#285)
- Improved IDE type hints for array annotations (#291)
- Add --network-alias option (#297)
- Improve healthcheck after serve, add restart policy (#296)
- Add hot example (#288)
- Add network args to from_image (#299)
- Add `--output-path` to serve (#295)
- [**breaking**] Drop msgpack support (#303)
- Introduce job ID to control the location of output logs / artifacts (#314)
- Also print text logs to stderr (#311)
- Add experimental `require_file` function to mark externally mounted files as required at runtime  (#261)

### Bug Fixes

- Move private pip imports into relevant func scope (#292)
- Correct error message on use of T created via from_image (#298)
- Ensure signature consistency between `engine.serve` and `Tesseract.from_image` (#302)
- Exclude broken version of setuptools-scm (#312)
- Use TESSERACT_MLFLOW_TRACKING_URI instead of MLFLOW_TRACKING_URI (#313)
- Make `tesseract serve --output-format` behave as expected (#307)
- Test that LogPipe implementations do not diverge (#316)
- Use `--input-path`/`--output-path` as `base_dir` in `json+binref` encoder (#304)
- Bring back CLI option shorthands (#323)
- Ensure .dockerignore is observed to avoid copying large amounts of data to build contexts (#218) (#321)
- Change location of mlruns when TESSERACT_MLFLOW_TRACKING_URI is a relative path (#325)

### Refactor

- [**breaking**] Remove ability to serve multiple tesseracts / docker compose (#286)
- Overhaul all clis (#301)
- [**breaking**] Remove input-schema and output-schema endpoints (#308)
- Transpose folder structure (#318)

### Documentation

- Add link to pre-commit library (#327)
- Document how to use Tesseract via SLURM on HPC clusters (#320)
- Add Documentation for podman usage via TESSERACT_DOCKER_EXECUTABLE (#324)

## [0.10.2] - 2025-07-21

### Bug Fixes

- Fix + test for missing deps in pip install test (#278)

## [0.10.1] - 2025-07-21

### Features

- Add --input-path to tesseract cli and TESSERACT_INPUT_PATH to runtime (#249)
- Introduce logging metrics, parameters and artifacts to file or MLflow (#229)
- Add top-level description field to OAS (#268)

### Bug Fixes

- Ensure tracebacks are always propagated through Python client (#228)
- Ensure default workdir is writable (#263)
- Tesseract-runtime default io paths (#266)
- Add volume statements to dockerfile to ensure logs / metrics / data are always written to a volume (#270)
- Add tests for MLflow backend using MLflow's capabillity to write to file (#271)
- Bug in volume error handling (#272)

### Refactor

- Use RuntimeConfig for --input-path/--output-path (#264)

### Documentation

- Fix docstring on metrics example (#269)

## [0.10.0] - 2025-07-11

### Features

- *(sdk)* Expose no compose in Python API (#223)
- [**breaking**] Enable remote debugging (#184)
- Add --service-names argument to `tesseract serve` so served Tesseracts can be reached by name (#206)
- Allow skipping checks by passing `--skip-checks` flag to the tesseract build command (#233)
- Add Volume class to docker client and --user flag to cli (#241)
- Pass env variables through `tesseract run` and `tesseract serve` (#250)
- Allow to run T containers as any user, for better volume permission handling (#253)

### Bug Fixes

- Fix teardown command crashing for wrong proj ID (#207)
- Add FileNotFoundError to docker info (#215)
- Gracefully exit when Docker executable not found (#216)
- "docker buildx build requires exactly 1 argument" error when using `tesseract build --forward-ssh-agent` (#231)
- Remove zip(strict=True) for py39 support (#227)
- Allow to set all configs via `tesseract build --config-override` (#239)
- Add environment to no_compose (#257)

### Documentation

- Add in data assimilation tutorial and refactor example gallery (#200)
- Remove reference to Hessian matrices (#221)
- New user usability improvements (#226)
- Fine-tune onboarding experience (#243)

## [0.9.1] - 2025-06-05

### Features

- *(cli)* Add serve --no-compose and other missing cli options (#161)
- *(sdk)* Make docker executable and build args configurable (#162)
- More comprehensive validation of input and output schema during `tesseract-runtime check` (#170)
- Add ability to configure host IP during `tesseract serve` (#185)

### Bug Fixes

- Add new cleanup fixture to track docker assets that need to be cleaned up (#129)
- Some validation errors do not get piped through the python client (#152)
- Podman compatibility and testing (#142)
- Apidoc CLI call used container ID in place of container object to retrieve host port (#172)
- Overhaul docker client for better podman compatibility and better error handling (#178)
- Sanitize all config fields passed as envvars to dockerfile (#187)

### Documentation

- Updated diagram on tesseract interfaces (#150)
- Tesseract Example Gallery (#149)
- Remove how-to guides froms sidebar (#177)

## [0.9.0] - 2025-05-02

### Features

- [**breaking**] Remove docker_py usage in favor of custom client that uses Docker CLI (#33)
- *(sdk)* Allow users to serve Tesseracts using multiple worker processes (#135)

### Documentation

- Update quickstart (#144)

## [0.8.5] - 2025-04-24

### Bug Fixes

- Fixed typos in jax recipe (#134)
- *(sdk)* Various improvements to SDK UX (#136)

## [0.8.4] - 2025-04-17

### Features

- Allow creating tesseract objects from python modules (#122)
- Also allow passing an imported module to Tesseract.from_tesseract_api (#130)

## [0.8.3] - 2025-04-14

### Bug Fixes

- Fix Tesseract SDK decoding and error handling (#123)

## [0.8.2] - 2025-04-11

### Features

- Add requirement provider config to build Tesseracts from conda env specs (#54)
- Introduce debug mode for served Tesseracts to propagate tracebacks to clients (#111)

### Bug Fixes

- Ensure Ubuntu-based base images work as expected; change default to vanilla Debian (#115)
- Enable users to opt-in to allowing extra fields in Tesseract schemas by setting `extra="allow"` (#117)
- Meshstats `abstract_eval` (#120)

## [0.8.1] - 2025-03-28

### Bug Fixes

- Pydantic 2.11.0 compatibility (hotfix) (#106)

## [0.8.0] - 2025-03-27

### Features

- Implement check_gradients runtime command (#72)
- [**breaking**] Validate endpoint argument names before building (#95)

### Bug Fixes

- OpenAPI schema failure for differentiable arrays with unknown shape (#100)
- Prevent silent conversion of float array to int (#96)
- Use fixed uid/gid 5000:5000 for all tesseracts (#102)
- Use uid 1000 instead of 5000 (#104)

### Refactor

- Unpack endpoint payload (#80)

### Documentation

- Dependencies and user privileges (#91)

## [0.7.4] - 2025-03-20

### Features

- Friendlier error messages when input validation fails (#71)
- Pytorch initialize template (#53)
- Add `diffable` field to input/output json schemas (#82)
- Add stdout output for tesseract build (#87)

### Documentation

- Various docs nits + UX fixes (#85)

## [0.7.3] - 2025-03-13

### Features

- Raise proper error (`ValidationError`) for invalid inputs (#67)
- Add `abstract_eval` method to `tesseract_core.Tesseract` (#76)

### Bug Fixes

- Jax template now uses equinox `filter_jit` to allow non-array inputs (#56)
- Added pip as dependency (#58)
- Issue #74 (#75)

### Documentation

- Updated comments in jax recipe and docs on Differentiable flag (#65)

## [0.7.2] - 2025-02-27

### Bug Fixes

- Validate ShapeDType in abstract-eval schemas (#40)
- Resolve paths before passing volumes to docker (#48)
- Dangling Tesseracts in e2e tests (#51)
- Sanitize error output (#52)

### Documentation

- Python API for Julia example (#37)
- Fix links again (#49)

## [0.7.1] - 2025-02-26

### Bug Fixes

- Address issues in installing and first steps with Tesseract (#30)

## [0.7.0] - 2025-02-25

### Refactor

- Remove LocalClient and use HTTPClient for local Tesseracts as well (#27)

### Documentation

- Python API for PyTorch example (#24)
- Fix RBF fitting example (#25)

<!-- generated by git-cliff -->
