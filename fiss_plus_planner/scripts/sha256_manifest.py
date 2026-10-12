"""SHA256SUMS of a data set folder, read from disk, and the check of a tar archive against it.

The local PC has a RAM fault that corrupts the page cache, so a checksum of a copy is only trustworthy
if the original was read from disk: each file's cached pages are dropped before it is hashed.

python scripts/sha256_manifest.py make  <folder> [--exclude .gif] [--workers 16]   # writes <folder>/SHA256SUMS
python scripts/sha256_manifest.py check <archive.tar> <folder name in the archive>  # every member vs SHA256SUMS
"""
import argparse
import hashlib
import os
import sys
import tarfile
from multiprocessing import Pool


def sha256_from_disk(path: str) -> str:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        h = hashlib.sha256()
        while chunk := os.read(fd, 1 << 22):
            h.update(chunk)
        return h.hexdigest()
    finally:
        os.close(fd)


def make(folder: str, exclude: list, workers: int) -> None:
    files = sorted(os.path.relpath(os.path.join(d, f), folder) for d, _, fs in os.walk(folder) for f in fs
                   if f != "SHA256SUMS" and not f.endswith(tuple(exclude)))
    with Pool(workers) as p:
        sums = p.map(sha256_from_disk, [os.path.join(folder, f) for f in files], chunksize=256)
    with open(os.path.join(folder, "SHA256SUMS"), "w") as fh:
        fh.writelines(f"{s}  {f}\n" for s, f in zip(sums, files))
    print(f"{len(files)} files -> {os.path.join(folder, 'SHA256SUMS')}")


def check(archive: str, name: str) -> None:
    fd = os.open(archive, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # read the archive from disk, not from the cache
    os.close(fd)
    want, seen, bad = None, {}, []
    with tarfile.open(archive, "r|") as tar:  # streamed: one pass over the archive
        for m in tar:
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            rel = os.path.relpath(m.name, name)
            if rel == "SHA256SUMS":
                want = dict(reversed(line.rstrip("\n").split("  ", 1)) for line in data.decode().splitlines())
            else:
                seen[rel] = hashlib.sha256(data).hexdigest()
    if want is None:
        sys.exit("SHA256SUMS is not in the archive")
    missing = sorted(set(want) - set(seen))
    extra = sorted(set(seen) - set(want))
    bad = sorted(f for f in set(want) & set(seen) if want[f] != seen[f])
    print(f"{len(seen)} files in the archive, {len(want)} in SHA256SUMS: {len(bad)} differ, {len(missing)} missing, "
          f"{len(extra)} not listed")
    for f in (bad + missing + extra)[:20]:
        print("  ", f)
    sys.exit(1 if bad or missing or extra else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("folder")
    m.add_argument("--exclude", nargs="*", default=[".gif"], help="file name endings to leave out")
    m.add_argument("--workers", type=int, default=16)
    c = sub.add_parser("check")
    c.add_argument("archive")
    c.add_argument("name", help="top folder name inside the archive")
    a = p.parse_args()
    make(a.folder, a.exclude, a.workers) if a.cmd == "make" else check(a.archive, a.name)
