## Rebuild SIMAP

Set the dataset directory:
```bash
simap set-dir dataset <dir>
```
**Requirements**: `<dir>` must exist, be writable, and have at least 280 GB free space.

Rebuild the dataset:
```bash
simap prepare-dataset --also-export
```

See useful options with `--help`:
```bash
simap prepare-dataset --help
```

## Future Dataset Update Steps

For future dev, artifact IDs `1..8058` are done. To extend the dataset:
1. update `START_ARTIFACT_ID` to `8059` and `END_ARTIFACT_ID` to the latest upload on musical artifacts
2. download simap from HF
3. rename simap to simap_dev
4. run `simap prepare-dataset --also-export` (new data will look at old data for cross-dedup)
5. update simap version
6. upload the new version to HF
