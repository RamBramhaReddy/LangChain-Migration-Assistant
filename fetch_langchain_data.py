import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

REPO = "langchain-ai/langchain"
API_BASE = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

if GITHUB_TOKEN:
    print("GitHub token found -- using authenticated requests (5000/hour limit).")
else:
    print("No GitHub token found -- using unauthenticated requests (60/hour limit).")
    print("Add GITHUB_TOKEN to your .env file to avoid rate limit errors.\n")

CHECKPOINTS = {
    "v0.1": "v0.1.0",
    "v0.2": "langchain==0.2.0",
    "v0.3": "langchain==0.3.0",
    "v1.0": "langchain==1.0.0",
}

CHECKPOINTS_USING_NEW_DOCS_REPO = {"v1.0"}

MIGRATION_GUIDES_PATH = "docs/docs/versions"

OUTPUT_FOLDER = "data/raw"

NEW_DOCS_REPO = "langchain-ai/docs"
NEW_DOCS_PATH = "src/oss/python/releases"

def safe_get(url, params=None, max_retries=3, wait_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            return requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.exceptions.ConnectionError as e:
            print(f"  connection error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"  waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
            else:
                print("  giving up after max retries.")
                raise

def get_all_real_tags(repo, stop_when_found=None):
    all_tags = []
    still_needed = set(stop_when_found) if stop_when_found else None
    page = 1
    while True:
        url = f"{API_BASE}/repos/{repo}/tags"
        response = safe_get(url, params={"per_page": 100, "page": page})
        if response.status_code != 200:
            print(f"  could not fetch tag list (status {response.status_code})")
            break
        page_tags = [t["name"] for t in response.json()]
        if not page_tags:
            break

        all_tags.extend(page_tags)

        if still_needed is not None:
            still_needed -= set(page_tags)
            if not still_needed:
                print(f"  (found all target tags by page {page}, stopping early)")
                break

        page += 1
        if page > 50:
            print("  reached page limit (50) without finding everything.")
            break
    return all_tags

def verify_tags_exist(checkpoints):
    print("Fetching the real tag list from GitHub...")
    target_tags = set(checkpoints.values())
    real_tags = get_all_real_tags(REPO, stop_when_found=target_tags)
    print(f"  scanned {len(real_tags)} tags total on the repo.\n")

    all_good = True
    for label, tag in checkpoints.items():
        if tag in real_tags:
            print(f"  OK   {label} -> tag '{tag}' exists")
        else:
            print(f"  MISSING   {label} -> tag '{tag}' NOT found in repo's tag list")
            all_good = False

    if not all_good:
        print("\nSome tags were not found. Check the exact tag name on GitHub")
        print("(github.com/langchain-ai/langchain/tags) before continuing.")
    else:
        print("All tags confirmed. Safe to proceed.\n")

    return all_good

def download_one_file(file_path, version_tag, save_to, repo=None):
    repo = repo or REPO
    url = f"{API_BASE}/repos/{repo}/contents/{file_path}"
    response = safe_get(url, params={"ref": version_tag})

    if response.status_code != 200:
        print(f"  could not fetch {file_path} (status {response.status_code})")
        return

    data = response.json()
    download_url = data.get("download_url")

    if download_url is None:
        print(f"  no download link for {file_path}")
        return

    file_response = safe_get(download_url)
    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    with open(save_to, "w", encoding="utf-8") as f:
        f.write(file_response.text)

    print(f"  saved: {save_to}")

def download_folder(folder_path, version_tag, save_folder, repo=None, root_path=None):
    repo = repo or REPO
    if root_path is None:
        root_path = folder_path

    url = f"{API_BASE}/repos/{repo}/contents/{folder_path}"
    response = safe_get(url, params={"ref": version_tag})

    if response.status_code != 200:
        print(f"  could not list folder {folder_path} (status {response.status_code})")
        return

    items = response.json()

    for item in items:
        if item["type"] == "dir":
            download_folder(item["path"], version_tag, save_folder, repo, root_path)

        elif item["type"] == "file":
            if item["name"].endswith((".md", ".mdx")):
                relative_path = os.path.relpath(item["path"], root_path)
                save_path = os.path.join(save_folder, relative_path)
                download_one_file(item["path"], version_tag, save_path, repo)

def download_release_notes(version_tag, save_to):
    url = f"{API_BASE}/repos/{REPO}/releases/tags/{version_tag}"
    response = safe_get(url)

    if response.status_code != 200:
        print(f"  could not fetch release notes for {version_tag}")
        return

    data = response.json()
    notes_text = data.get("body", "")

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(notes_text)

    print(f"  saved: {save_to}")

def main():
    tags_ok = verify_tags_exist(CHECKPOINTS)
    if not tags_ok:
        print("Stopping here. Fix the tag names above, then run again.")
        return

    for label, tag in CHECKPOINTS.items():
        print(f"\n--- Fetching checkpoint: {label} (tag={tag}) ---")

        version_folder = os.path.join(OUTPUT_FOLDER, label)

        if label in CHECKPOINTS_USING_NEW_DOCS_REPO:
            print("Fetching migration guides (from new docs repo)...")
            download_folder(
                NEW_DOCS_PATH,
                "main",
                os.path.join(version_folder, "migration_guides"),
                repo=NEW_DOCS_REPO
            )
        else:
            print("Fetching migration guides...")
            download_folder(MIGRATION_GUIDES_PATH, tag, os.path.join(version_folder, "migration_guides"))

        print("Fetching release notes...")
        download_release_notes(tag, os.path.join(version_folder, "release_notes.md"))

    print(f"\n--- Fetching checkpoint: latest (repo={NEW_DOCS_REPO}, ref=main) ---")
    latest_folder = os.path.join(OUTPUT_FOLDER, "latest")

    print("Fetching docs...")
    download_folder(
        NEW_DOCS_PATH,
        "main",
        os.path.join(latest_folder, "migration_guides"),
        repo=NEW_DOCS_REPO
    )

    print("\nAll checkpoints done. Check the data/raw/ folder.")

if __name__ == "__main__":
    main()