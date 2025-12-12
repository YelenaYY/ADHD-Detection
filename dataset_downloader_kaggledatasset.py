#!/usr/bin/env python3
"""
Utility to download the ADHD200 preprocessed anatomical dataset via kagglehub.

Usage examples:
  - Default dataset (downloads under this repo's data/ directory):
      python dataset_downloader.py
  - Specify a different Kaggle dataset slug:
      python dataset_downloader.py --dataset owner/dataset-slug
  - Create a convenient local symlink to the cached dataset path:
      python dataset_downloader.py -s ./data/ADHD200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_DATASET_ID = "purnimakumarrr/adhd200-preprocessed-anatomical-dataset"


def ensure_cache_dir(cache_dir: str | Path) -> Path:
	"""
	Ensure kagglehub will cache into the provided directory by setting KAGGLEHUB_CACHE.
	Returns the absolute cache directory path.
	"""
	cache_path = Path(cache_dir).resolve()
	cache_path.mkdir(parents=True, exist_ok=True)
	os.environ["KAGGLEHUB_CACHE"] = str(cache_path)
	return cache_path


def download_dataset(dataset_id: str) -> str:
	"""
	Download the dataset using kagglehub and return the local path where files reside.
	"""
	try:
		import kagglehub  # imported here to give a clearer error if missing
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"kagglehub is not installed. Install it with: pip install kagglehub"
		) from exc

	print(f"Downloading dataset '{dataset_id}' (this may take a while on first run)...")
	path = kagglehub.dataset_download(dataset_id)
	print(f"Path to dataset files: {path}")
	return path


def create_symlink(source_dir: str | Path, link_path: str | Path) -> None:
	"""
	Create a symlink at link_path pointing to source_dir.
	If link_path exists:
	- If it already points to source_dir, keep it.
	- If it's an empty directory, remove it and replace with symlink.
	- Otherwise, raise an error to avoid accidental data loss.
	"""
	src = Path(source_dir).resolve()
	dst = Path(link_path)
	dst_parent = dst.parent
	dst_parent.mkdir(parents=True, exist_ok=True)

	if dst.exists() or dst.is_symlink():
		# Already the correct link?
		try:
			if dst.is_symlink() and dst.resolve() == src:
				print(f"Symlink already exists: {dst} -> {src}")
				return
		except FileNotFoundError:
			# Broken symlink; we'll replace it
			pass

		if dst.is_dir() and not dst.is_symlink():
			# Only remove if empty to be safe
			if any(dst.iterdir()):
				raise FileExistsError(
					f"Refusing to replace non-empty directory at {dst}. "
					f"Please remove or choose a different --symlink path."
				)
			dst.rmdir()
		else:
			dst.unlink()

	dst.symlink_to(src, target_is_directory=True)
	print(f"Created symlink: {dst} -> {src}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download Kaggle dataset via kagglehub and optionally create a local symlink."
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default=DEFAULT_DATASET_ID,
		help=f"Kaggle dataset identifier (default: {DEFAULT_DATASET_ID})",
	)
	default_cache = str((Path(__file__).resolve().parent / "data").resolve())
	parser.add_argument(
		"--cache-dir",
		type=str,
		default=default_cache,
		help=f"Directory to store kagglehub cache (default: {default_cache})",
	)
	parser.add_argument(
		"-s",
		"--symlink",
		type=str,
		default=None,
		help="Optional path to create a symlink to the downloaded dataset (e.g., ./data/ADHD200)",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	# Force kagglehub to cache under the chosen directory (default: repo data/)
	try:
		cache_dir = ensure_cache_dir(args.cache_dir)
		print(f"Using kagglehub cache dir: {cache_dir}")
	except Exception as exc:
		print(f"Error preparing cache directory: {exc}", file=sys.stderr)
		return 1

	try:
		dataset_path = download_dataset(args.dataset)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	if args.symlink:
		try:
			create_symlink(dataset_path, args.symlink)
		except Exception as exc:
			print(f"Warning: failed to create symlink: {exc}", file=sys.stderr)

	print("Done.")
	print(f"Dataset path: {dataset_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


