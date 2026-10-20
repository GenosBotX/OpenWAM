"""Compatibility entry point for the unified raw Linglong converter.

Use the same --input/--output-root/--prompt arguments as convert_linglong.py.
The old bucket-to-bucket conversion interface has been replaced by direct
raw recording conversion. Both joint and measured-state EEF columns are
always generated together.
"""

if __package__:
    from .convert_linglong import main
else:
    from convert_linglong import main


if __name__ == "__main__":
    main()
