
import requests
import json
from datetime import datetime
from pathlib import Path

import hashlib
from bs4 import BeautifulSoup
from typing import Tuple

URLS = [
	"https://www.grammarly.com/business/learn/barriers-to-effective-communication/",
	"https://www.forbes.com/sites/adriandearnell/2022/12/01/pitfalls-of-cross-cultural-communication-dont-fall-in/",
	"https://en.wikipedia.org/wiki/Effective_Public_Relations",
	"https://www.zestfor.com/resources/thought-leadership/people-development/communicate-complex-data-effectively/",
	"https://en.wikipedia.org/wiki/Active_listening",
	"https://www.communicationtheory.org/shannon-weaver-model-of-communication/",
	"https://www.financialprofessionals.org/training-resources/resources/articles/Details/the-5-key-principles-of-nonverbal-communication",
	"https://professional.dce.harvard.edu/blog/8-ways-you-can-improve-your-communication-skills/#1-Be-clear-and-concise",
	"https://www.dalecarnegie.com/blog/how-effective-communication-skills-drive-success-at-work/",
	"https://thebrieflab.com/blog/the-3-cs-of-communication-clear-concise-consistent/",
	"https://www.robertwalters.com.au/insights/career-advice/blog/mastering-effective-communication-skills.html",
]

OUT_PATH = Path("data/processed/web_docs.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def extract_main_text(html: str) -> Tuple[str, str, str]:
    """
    Extrae el texto principal y el título de una página HTML.
    Devuelve: (text, title, extraction_method)
    """
    soup = BeautifulSoup(html, "html.parser")
    # Heurística simple: usar <main> si existe, si no, el <body>
    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
        method = "bs4_main_tag"
    else:
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else ""
        method = "bs4_body_tag"
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    return text, title, method

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fetch_and_parse_url(url: str):
	meta = {
		"source_url": url,
		"fetched_at": datetime.utcnow().isoformat(),
	}
	HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
	try:
		resp = requests.get(url, headers=HEADERS, timeout=15)
		meta["http_status"] = resp.status_code
		meta["content_type"] = resp.headers.get("Content-Type", "")
		html = resp.text
		if resp.status_code == 200:
			text, title, method = extract_main_text(html)
			meta["title"] = title
			meta["extraction_method"] = method
			meta["fetch_status"] = "ok"
			content = text
		else:
			meta["fetch_status"] = "http_error"
			content = ""
	except Exception as e:
		meta["http_status"] = None
		meta["content_type"] = ""
		meta["fetch_status"] = "needs_headless"
		meta["error"] = str(e)
		content = ""
		html = ""
	meta["content_hash"] = compute_sha256(content or html)
	doc_id = compute_sha256(url)
	return {
		"doc_id": doc_id,
		"source_url": url,
		"fetched_at": meta["fetched_at"],
		"content": content,
		"metadata": meta,
	}

def main():

	with OUT_PATH.open("w", encoding="utf-8") as f:
		for url in URLS:
			doc = fetch_and_parse_url(url)
			f.write(json.dumps(doc, ensure_ascii=False) + "\n")
			meta = doc["metadata"]
			status = meta.get("fetch_status")
			if status == "ok":
				print(f"Fetched: {url} | status: ok | title: {meta.get('title','')}")
			elif status == "http_error":
				print(f"FAILED: {url} | HTTP status: {meta.get('http_status')} | content_type: {meta.get('content_type')}")
			elif status == "needs_headless":
				print(f"FAILED: {url} | needs_headless | error: {meta.get('error')}")
			else:
				print(f"FAILED: {url} | status: {status} | meta: {meta}")

if __name__ == "__main__":
	main()