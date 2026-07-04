# Local Android Native Dependencies

Place optional local native dependency archives here when Android Studio cannot
download them during CMake configure.

For the latest llama.cpp ARM CPU backend, KleidiAI can be provided offline as:

```text
kleidiai-v1.24.0-src.tar.gz
```

The archive is intentionally ignored by git. You can also keep the archive
elsewhere and set this in `examples/llama.android/local.properties`:

```properties
heterospec.kleidiai.archive=/absolute/path/to/kleidiai-v1.24.0-src.tar.gz
```

If you already extracted the archive, use:

```properties
heterospec.kleidiai.sourceDir=/absolute/path/to/extracted/kleidiai-v1.24.0
```
