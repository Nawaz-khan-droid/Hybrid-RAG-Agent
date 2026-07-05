"""
Push local repo to GitHub using dulwich (pure-Python git, no libcurl).

Usage: set GITHUB_PAT env var, then: python _push_repo.py
"""

import os
import logging
from pathlib import Path
from dulwich import porcelain
from dulwich.repo import Repo

REPO = Path(__file__).parent.resolve()
REMOTE = "https://github.com/Nawaz-khan-droid/Hybrid-RAG-Agent.git"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("push")


def main():
    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        logger.error("GITHUB_PAT not set")
        return 1

    # Init if needed
    if not (REPO / ".git").exists():
        porcelain.init(str(REPO))
        logger.info("Repo initialized")

    repo = Repo(str(REPO))
    # Point HEAD to main branch (fresh repo or default master)
    repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")

    try:
        head_sha = repo.head()
        logger.info("HEAD at %s", head_sha.decode()[:8])
    except KeyError:
        logger.info("Fresh repo - no commits yet")

    # Stage all files (respects .gitignore automatically)
    porcelain.add(REPO, ".")
    logger.info("Files staged")

    # Commit
    try:
        porcelain.commit(REPO, message="Update Secure Hybrid RAG v2 - production-ready")
        logger.info("Commit created")
    except Exception:
        logger.info("Nothing new to commit")

    # Push with PAT in URL (in-memory, never written to config)
    auth_url = f"https://Nawaz-khan-droid:{pat}@github.com/Nawaz-khan-droid/Hybrid-RAG-Agent.git"
    logger.info("Pushing to %s ...", REMOTE)
    result = porcelain.push(REPO, auth_url, "main", force=True)
    logger.info("Push complete")

    return 0


if __name__ == "__main__":
    exit(main())
