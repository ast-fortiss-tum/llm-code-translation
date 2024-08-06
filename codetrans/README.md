# Code Translation library codetrans

`codetrans` is a library for LLM-based source code translation.

To install all software dependencies, please execute the following command:
```
pip install -e codetrans
```


## Instructions to install the requirements for the project dependency python-magic:

This dependency is required because of the `comment-parser` dependency of `codetrans`.

The current stable version of python-magic is available on PyPI and
can be installed by running `pip install python-magic`.

This module is a simple wrapper around the libmagic C library, and
that *must be installed as well*:

### Debian/Ubuntu

```
sudo apt-get install libmagic1
```

### Windows

You'll need DLLs for libmagic.  @julian-r maintains a pypi package with the DLLs, you can fetch it with:

```
pip install python-magic-bin
```

### OSX

- When using Homebrew: `brew install libmagic`
- When using macports: `port install file`
