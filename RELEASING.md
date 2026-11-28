# Releasing polysmelt

The version in Cargo.toml is the one source of truth. After CI passes on a push to `main`, the
[Auto release workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/auto-release.yml)
reads it: if no `vX.Y.Z` tag has that version yet, it tags the commit and publishes it to
crates.io; otherwise it does nothing. So to release, raise the version and push:

```sh
scripts/bump.sh minor          # or patch, major, or an exact version such as 1.0.0-rc.1
git commit -am "Release 0.24.0"
git push
```

`scripts/bump.sh` sets the version in Cargo.toml and moves polysmelt's entry in the three
lockfiles (the crate's, `bench/` and `demo/`) with it. Add the version's section to CHANGELOG.md
in the same commit. Leave the version alone to let changes collect; they go out with the next
version pushed. Nothing is committed back to `main`, and if CI fails nothing is tagged: the next
push that passes releases the version then. A version behind the newest tag fails the workflow
rather than releasing.

## crates.io

There is one crate, `polysmelt`. The
[Publish to crates.io workflow](https://github.com/Azazel-Labs/polysmelt/actions/workflows/crates.yml)
publishes it with the `CARGO_REGISTRY_TOKEN` repository secret. It checks that the tag matches
Cargo.toml's version and skips a version that is already published, so it can be rerun. To retry,
use **Actions → Publish to crates.io → Run workflow** with an existing tag such as `v0.1.0`.
Published versions can't be overwritten: a fix needs a new version.
