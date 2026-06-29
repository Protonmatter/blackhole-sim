"""CLI for public GRMHD dump manifest and verification workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .public_dumps import download_public_dump, verify_public_dump, write_public_manifest, write_verification_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Public GRMHD dump acquisition and verification")
    p.add_argument("--manifest", type=Path, default=Path("data/public/illinois_v3_manifest.json"))
    p.add_argument("--write-manifest", action="store_true")
    p.add_argument("--download-url", help="Concrete HDF5 file URL selected from a public data portal")
    p.add_argument("--download-output", type=Path, default=Path("data/public/selected_dump.h5"))
    p.add_argument("--sha256")
    p.add_argument("--allow-unverified-download", action="store_true", help="Allow a download without SHA-256 evidence for local exploration only")
    p.add_argument("--verify", type=Path, help="Downloaded HDF5 dump to inspect and adapt")
    p.add_argument("--adapter", choices=("harm", "koral", "bhac"), default="harm")
    p.add_argument("--report", type=Path, default=Path("out/public_dump_verification.json"))
    args = p.parse_args(argv)

    if args.write_manifest:
        path = write_public_manifest(args.manifest)
        print(f"wrote manifest {path}")

    if args.download_url:
        if not args.sha256 and not args.allow_unverified_download:
            raise SystemExit("--sha256 is required for reproducible public-dump downloads; use --allow-unverified-download for local exploration")
        path = download_public_dump(args.download_url, args.download_output, expected_sha256=args.sha256)
        print(f"downloaded {path}")

    if args.verify:
        report = verify_public_dump(args.verify, adapter=args.adapter)
        write_verification_report(report, args.report)
        print(f"verified {args.verify}; wrote {args.report}")
        print(report)

    if not (args.write_manifest or args.download_url or args.verify):
        p.print_help()


if __name__ == "__main__":
    main()
