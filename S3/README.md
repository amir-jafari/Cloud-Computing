# S3 Manager — `s3://dats-dl/ajafari@gwu.edu/`

A single-file Python tool to **list, upload, download, and delete** files in your
personal S3 folder. Every path you type is relative to `ajafari@gwu.edu/`, so
you cannot reach outside your own prefix by accident.

## Setup

```bash
pip install boto3
```

### Credentials — the `.pem` question

A `.pem` file is an **SSH key**. It authenticates you to an EC2 instance; it
cannot sign S3 API requests. Pick one of these instead:

| Where you run the script | What to use |
| --- | --- |
| Your Windows laptop | Access key + secret in `.env` or `aws configure` |
| An EC2 instance you SSH into with the `.pem` | The instance's IAM role — nothing to configure |

For the laptop case:

```bash
cp .env.example .env      # then paste your key + secret into .env
```

`.env` and `*.pem` are already git-ignored.

## Usage

```bash
# list
python s3_manager.py ls                     # top level of your folder
python s3_manager.py ls data/               # inside data/
python s3_manager.py ls -r                  # everything, recursively

# upload
python s3_manager.py upload model.pt                    # -> ajafari@gwu.edu/model.pt
python s3_manager.py upload model.pt models/best.pt     # rename on the way up
python s3_manager.py upload ./results results/ -r       # whole folder

# download
python s3_manager.py download data/train.csv            # -> ./train.csv
python s3_manager.py download data/train.csv C:\tmp\    # into a folder
python s3_manager.py download data/ ./local_data -r     # whole folder

# delete
python s3_manager.py rm old_model.pt        # asks for confirmation
python s3_manager.py rm scratch/ -r         # asks; requires typing "yes"
python s3_manager.py rm old_model.pt -y     # skip the prompt
```

Extra flags: `--profile NAME` to pick a named AWS profile, `--region` to
override `us-east-1`.

## Notes

- Uploads and downloads print live progress and use boto3's multipart transfer,
  so large model checkpoints are handled efficiently.
- `rm -r` deletes in batches of 1000 (the S3 API limit) and always previews what
  it is about to remove.
- S3 has no real folders. `ls` without `-r` fakes them with a `/` delimiter,
  which is why an "empty folder" may simply not exist.