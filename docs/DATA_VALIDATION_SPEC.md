# Data Validation Spec

## Purpose

Public GRMHD dump support must stay provenance-driven. Collection landing pages are useful for discovery, but they are not enough evidence for release validation.

## Descriptor States

- `collection_only`: identifies a public collection or model family. It must not be treated as a verified concrete dump.
- `selected_dump_verified`: identifies one concrete downloadable file and must include reproducibility evidence.

## Selected-Dump Gate

A descriptor with `direct_download_url` must include:

- `sha256`: SHA-256 of the exact downloaded file.
- `expected_field_map`: canonical field mapping for at least density, electron temperature, velocity, and magnetic field inputs.
- `accepted_ranges`: accepted numeric envelope for `rho` and `theta_e` at minimum.
- `validation_status`: `selected_dump_verified`.

The CLI requires `--sha256` for downloads by default. `--allow-unverified-download` is only for local exploration and must not be used as release evidence.

## Current State

The Illinois v3 entry is intentionally `collection_only`: the portal is a curated landing page, not a pinned file URL. The next data milestone should select one dump, record its checksum and expected field map, and commit only the compact manifest/report, not the dump itself.
