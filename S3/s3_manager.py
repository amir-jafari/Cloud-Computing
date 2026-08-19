#!/usr/bin/env python3
"""
S3 helper for s3://dats-dl/ajafari@gwu.edu/

Every path you pass is relative to your personal prefix, so you can never
accidentally touch another student's folder.

    python s3_manager.py ls
    python s3_manager.py ls data/ -r
    python s3_manager.py upload model.pt
    python s3_manager.py upload ./results results/ -r
    python s3_manager.py download data/train.csv
    python s3_manager.py download data/ ./local_data -r
    python s3_manager.py rm old_model.pt
    python s3_manager.py rm scratch/ -r

Credentials are resolved in this order:
    1. --profile / AWS_PROFILE
    2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in the environment
    3. a .env file sitting next to this script
    4. ~/.aws/credentials
    5. the IAM role attached to the EC2 instance (nothing to configure)
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import ClientError, NoCredentialsError

BUCKET = "dats-dl"
PREFIX = "ajafari@gwu.edu/"
REGION = "us-east-1"

ENV_FILE = Path(__file__).with_name(".env")


# --------------------------------------------------------------------------- #
# credentials / client
# --------------------------------------------------------------------------- #
def load_env_file():
    """Read KEY=VALUE lines from .env without needing python-dotenv."""
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:  # a blank value would shadow a real one from the environment
            os.environ.setdefault(key.strip(), value)


def credential_hint(code):
    """Turn an opaque AWS error code into something actionable."""
    hints = {
        "ExpiredToken": (
            "  Your temporary credentials have expired.\n"
            "  Start the lab again and paste the fresh key/secret/token into .env"
        ),
        "InvalidAccessKeyId": (
            "  Usually a missing or stale session token.\n"
            "  Keys starting with ASIA are temporary and REQUIRE AWS_SESSION_TOKEN in .env"
        ),
        "SignatureDoesNotMatch": (
            "  The secret key does not match the access key — check for a truncated\n"
            "  copy/paste or stray whitespace in .env"
        ),
        "AccessDenied": f"  Your credentials may not be allowed outside {PREFIX}",
        "NoSuchBucket": f"  Bucket '{BUCKET}' not found in this region.",
    }
    return hints.get(code, "")


def get_client(profile=None, region=None):
    load_env_file()
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", region_name=region or REGION)


# --------------------------------------------------------------------------- #
# key helpers
# --------------------------------------------------------------------------- #
def to_key(relative_path):
    """'data/train.csv' -> 'ajafari@gwu.edu/data/train.csv'"""
    rel = str(relative_path).replace("\\", "/").lstrip("/")
    return PREFIX + rel


def to_relative(key):
    """'ajafari@gwu.edu/data/train.csv' -> 'data/train.csv'"""
    return key[len(PREFIX):] if key.startswith(PREFIX) else key


def human(size):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,} B"
        size /= 1024


class Progress:
    """Single-line transfer progress, safe across boto3's worker threads."""

    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.seen = 0
        self.lock = threading.Lock()

    def __call__(self, chunk):
        with self.lock:
            self.seen += chunk
            if self.total:
                pct = self.seen / self.total * 100
                sys.stdout.write(
                    f"\r  {self.label}  {human(self.seen)} / {human(self.total)}  ({pct:5.1f}%)"
                )
            else:
                sys.stdout.write(f"\r  {self.label}  {human(self.seen)}")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_ls(s3, args):
    prefix = to_key(args.path) if args.path else PREFIX
    if args.path and not prefix.endswith("/") and not args.recursive:
        prefix += "/"

    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}
    if not args.recursive:
        kwargs["Delimiter"] = "/"

    folders, files, total = [], [], 0
    for page in paginator.paginate(**kwargs):
        for cp in page.get("CommonPrefixes", []):
            folders.append(to_relative(cp["Prefix"]))
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):  # folder placeholder object
                continue
            files.append(obj)
            total += obj["Size"]

    print(f"s3://{BUCKET}/{prefix}")
    for f in folders:
        print(f"  {'<DIR>':>12}  {'':19}  {f}")
    for obj in files:
        stamp = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {human(obj['Size']):>12}  {stamp}  {to_relative(obj['Key'])}")

    if not folders and not files:
        print("  (empty)")
    else:
        print(f"\n{len(folders)} folder(s), {len(files)} file(s), {human(total)}")
    return 0


def cmd_upload(s3, args):
    local = Path(args.local).expanduser()
    if not local.exists():
        print(f"error: {local} does not exist", file=sys.stderr)
        return 1

    if local.is_dir():
        if not args.recursive:
            print(f"error: {local} is a folder — add -r to upload it", file=sys.stderr)
            return 1
        base = args.remote.rstrip("/") + "/" if args.remote else ""
        uploaded = 0
        for path in sorted(local.rglob("*")):
            if path.is_file():
                key = to_key(base + path.relative_to(local).as_posix())
                print(f"{path}  ->  s3://{BUCKET}/{key}")
                s3.upload_file(
                    str(path), BUCKET, key,
                    Callback=Progress(path.name, path.stat().st_size),
                )
                print()
                uploaded += 1
        print(f"uploaded {uploaded} file(s)")
        return 0

    remote = args.remote or local.name
    if remote.endswith("/"):
        remote += local.name
    key = to_key(remote)
    print(f"{local}  ->  s3://{BUCKET}/{key}")
    s3.upload_file(str(local), BUCKET, key, Callback=Progress(local.name, local.stat().st_size))
    print("\ndone")
    return 0


def _iter_keys(s3, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def cmd_download(s3, args):
    if args.recursive:
        prefix = to_key(args.remote)
        if not prefix.endswith("/"):
            prefix += "/"
        dest_root = Path(args.local or Path(args.remote.rstrip("/")).name).expanduser()
        count = 0
        for key, size in _iter_keys(s3, prefix):
            if key.endswith("/"):
                continue
            target = dest_root / to_relative(key)[len(to_relative(prefix)):]
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"s3://{BUCKET}/{key}  ->  {target}")
            s3.download_file(BUCKET, key, str(target), Callback=Progress(target.name, size))
            print()
            count += 1
        if count == 0:
            print(f"nothing found under s3://{BUCKET}/{prefix}", file=sys.stderr)
            return 1
        print(f"downloaded {count} file(s)")
        return 0

    key = to_key(args.remote)
    target = Path(args.local).expanduser() if args.local else Path(Path(args.remote).name)
    if target.is_dir():
        target = target / Path(args.remote).name
    target.parent.mkdir(parents=True, exist_ok=True)

    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    print(f"s3://{BUCKET}/{key}  ->  {target}")
    s3.download_file(BUCKET, key, str(target), Callback=Progress(target.name, size))
    print("\ndone")
    return 0


def cmd_rm(s3, args):
    if args.recursive:
        prefix = to_key(args.remote)
        if not prefix.endswith("/"):
            prefix += "/"
        keys = [k for k, _ in _iter_keys(s3, prefix)]
        if not keys:
            print(f"nothing found under s3://{BUCKET}/{prefix}", file=sys.stderr)
            return 1
        print(f"about to delete {len(keys)} object(s) under s3://{BUCKET}/{prefix}")
        for k in keys[:20]:
            print(f"  {to_relative(k)}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
        if not args.yes and input("type 'yes' to confirm: ").strip().lower() != "yes":
            print("aborted")
            return 1
        for i in range(0, len(keys), 1000):  # DeleteObjects caps at 1000 per call
            batch = [{"Key": k} for k in keys[i:i + 1000]]
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": batch, "Quiet": True})
        print(f"deleted {len(keys)} object(s)")
        return 0

    key = to_key(args.remote)
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError:
        print(f"error: s3://{BUCKET}/{key} not found", file=sys.stderr)
        return 1
    if not args.yes and input(f"delete s3://{BUCKET}/{key}? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 1
    s3.delete_object(Bucket=BUCKET, Key=key)
    print("deleted")
    return 0


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description=f"Manage files in s3://{BUCKET}/{PREFIX}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--profile", help="AWS profile name from ~/.aws/credentials")
    p.add_argument("--region", default=None, help=f"AWS region (default {REGION})")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ls", help="list files")
    s.add_argument("path", nargs="?", default="", help="folder inside your prefix")
    s.add_argument("-r", "--recursive", action="store_true")
    s.set_defaults(func=cmd_ls)

    s = sub.add_parser("upload", help="upload a file or folder")
    s.add_argument("local", help="local file or folder")
    s.add_argument("remote", nargs="?", help="destination inside your prefix")
    s.add_argument("-r", "--recursive", action="store_true", help="required for folders")
    s.set_defaults(func=cmd_upload)

    s = sub.add_parser("download", help="download a file or folder")
    s.add_argument("remote", help="path inside your prefix")
    s.add_argument("local", nargs="?", help="local destination")
    s.add_argument("-r", "--recursive", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("rm", aliases=["delete"], help="delete a file or folder")
    s.add_argument("remote", help="path inside your prefix")
    s.add_argument("-r", "--recursive", action="store_true")
    s.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    s.set_defaults(func=cmd_rm)

    return p


def main():
    args = build_parser().parse_args()
    try:
        s3 = get_client(args.profile, args.region)
        return args.func(s3, args)
    except NoCredentialsError:
        print(
            "error: no AWS credentials found.\n"
            "  A .pem file works for SSH only — S3 needs an access key.\n"
            "  Fix it with any one of:\n"
            "    - copy .env.example to .env and fill in your key/secret\n"
            "    - run 'aws configure'\n"
            "    - run this on an EC2 instance that has an S3 IAM role",
            file=sys.stderr,
        )
        return 1
    except ClientError as e:
        err = e.response["Error"]
        print(f"\nAWS error [{err['Code']}]: {err['Message']}", file=sys.stderr)
        print(credential_hint(err["Code"]), file=sys.stderr)
        return 1
    except Boto3Error as e:
        # upload_file/download_file wrap failures in S3UploadFailedError etc.,
        # which are NOT ClientError, so match on the message text instead.
        message = str(e)
        print(f"\ntransfer failed: {message}", file=sys.stderr)
        for code in ("ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch",
                     "AccessDenied", "NoSuchBucket"):
            if code in message:
                print(credential_hint(code), file=sys.stderr)
                break
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())