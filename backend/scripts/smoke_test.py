"""
smoke_test.py

End-to-end API checks for the exact contract the React frontend
(scientific-research-assistant-main/src/App.jsx) relies on. Standard library only.

    python backend/scripts/smoke_test.py                                  # local: http://localhost:8000
    python backend/scripts/smoke_test.py https://your-backend.vercel.app  # deployed
    python backend/scripts/smoke_test.py https://your-backend.vercel.app --origin https://your-frontend.vercel.app
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

NO_RESULTS = "I couldn't find relevant information in the research paper corpus to answer this question."
results = []


def request(base, method, path, body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    if body is not None:
        req.add_header("Content-Type", "application/json")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw, time.time() - start
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read(), time.time() - start


def check(name, ok, info=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def chat(base, question, domain="both", history=None, origin=None):
    headers = {"Origin": origin} if origin else {}
    body = {"question": question, "domain_filter": domain, "chat_history": history or []}
    status, hdrs, raw, secs = request(base, "POST", "/api/chat", body, headers)
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = {"_raw": raw[:300].decode(errors="replace")}
    return status, hdrs, payload, secs


def valid_sources(sources):
    """Every field App.jsx reads from a source, with the types it expects."""
    for s in sources:
        if not isinstance(s.get("title"), str) or not s["title"]:
            return False
        if not isinstance(s.get("page"), (int, str)):
            return False
        for key in ("domain", "subtopic", "source_url"):
            if not isinstance(s.get(key, ""), str):
                return False
        if s.get("source_url") and not s["source_url"].startswith("https://arxiv.org/abs/"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", default="http://localhost:8000")
    parser.add_argument("--origin", default="http://localhost:5173", help="frontend origin to test CORS with")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    print(f"Testing {base} (frontend origin {args.origin})\n")

    # 1. Health
    status, _, raw, secs = request(base, "GET", "/api/health")
    health = json.loads(raw) if status == 200 else {}
    check("GET /api/health returns 200", status == 200, f"{secs:.1f}s")
    check("search index loaded", health.get("index_loaded") is True, f"{health.get('chunks')} chunks")
    check("GROQ_API_KEY configured", health.get("groq_configured") is True, health.get("groq_model", ""))

    # 2. CORS preflight exactly as the browser sends it for fetch(... JSON ...)
    status, hdrs, _, _ = request(base, "OPTIONS", "/api/chat", headers={
        "Origin": args.origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    lower = {k.lower(): v for k, v in hdrs.items()}
    allow_origin = lower.get("access-control-allow-origin", "")
    check("CORS preflight allowed", status == 200 and allow_origin in ("*", args.origin), f"allow-origin={allow_origin!r}")

    # 3. Answers in every domain scope (cold start included in the first call)
    cases = [
        ("both", "What are the latest advances in AI security?"),
        ("cybersecurity", "What is zero trust architecture?"),
        ("ai_ml", "How do label-flipping attacks affect federated learning models?"),
    ]
    for domain, question in cases:
        status, hdrs, payload, secs = chat(base, question, domain, origin=args.origin)
        sources = payload.get("sources", [])
        check(f"[{domain}] 200 with answer + sources", status == 200 and isinstance(payload.get("answer"), str)
              and isinstance(sources, list), f"{secs:.1f}s, {len(sources)} sources" if status == 200 else payload)
        check(f"[{domain}] answer is grounded (not a refusal, non-empty)",
              bool(payload.get("answer")) and payload.get("answer") != NO_RESULTS and len(sources) > 0)
        check(f"[{domain}] source fields match frontend contract", valid_sources(sources))
        if domain != "both":
            check(f"[{domain}] domain filter respected", all(s.get("domain") == domain for s in sources),
                  str(sorted({s.get('domain') for s in sources})))
        check(f"[{domain}] no Markdown/snippet markers in answer",
              not any(m in payload.get("answer", "") for m in ("**", "【", "Snippet #", "## ")))
        lower = {k.lower(): v for k, v in hdrs.items()}
        check(f"[{domain}] CORS header on response",
              lower.get("access-control-allow-origin") in ("*", args.origin))

    # 4. Multi-turn: history in the shape App.jsx sends (role/content only)
    first_q = "What is federated learning?"
    s1, _, p1, _ = chat(base, first_q, "ai_ml")
    history = [{"role": "user", "content": first_q}, {"role": "assistant", "content": p1.get("answer", "")}]
    s2, _, p2, secs = chat(base, "What label-flipping attacks threaten federated learning?", "ai_ml", history)
    ok = s1 == 200 and s2 == 200 and bool(p2.get("answer"))
    info = f"{secs:.1f}s" if ok else f"first={s1} {p1.get('detail', '')[:120]} | follow-up={s2} {p2.get('detail', '')[:120]}"
    check("multi-turn request with chat_history", ok, info)

    # 5. Out-of-corpus question must be refused, not hallucinated
    status, _, payload, _ = chat(base, "Who won the 2022 FIFA World Cup?", "both")
    check("out-of-corpus question refused", status == 200 and payload.get("answer") == NO_RESULTS,
          payload.get("answer", "")[:80])

    # 6. Errors come back as {"detail": "<string>"} because App.jsx shows payload.detail as text
    status, _, payload, _ = chat(base, "test", "physics")
    check("invalid domain -> 422 with string detail", status == 422 and isinstance(payload.get("detail"), str),
          str(payload.get("detail"))[:90])
    status, _, payload, _ = chat(base, "   ", "both")
    check("blank question -> 422 with string detail", status == 422 and isinstance(payload.get("detail"), str))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
