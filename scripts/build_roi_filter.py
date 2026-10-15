"""Compatibility entry point; use build_foveal_filter.py for new workflows."""

from build_foveal_filter import build_filter, main  # noqa: F401


if __name__ == "__main__":
    main()
