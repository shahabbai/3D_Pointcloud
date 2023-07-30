from pathlib import Path

from .hornschunck import HornSchunck


def getimgfiles(stem: Path, pat: str) -> list:

    stem = Path(stem).expanduser()

    print('searching', stem / pat)
    flist = sorted(stem.glob(pat))

    if not flist:
        raise FileNotFoundError(f'no files found under {stem} using {pat}')

    return flist
