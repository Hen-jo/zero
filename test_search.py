"""Quick verification that retrieval server returns proper content for tool responses."""

import json
import requests
import sys

RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"


def test_raw_api(queries, topk=3):
    """Test the raw retrieval server API."""
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    resp = requests.post(RETRIEVAL_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def passages2string(retrieval_result):
    """Same logic as search_r1_like_utils._passages2string."""
    fmt = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        fmt += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return fmt.strip()


def main():
    test_queries = [
        "who won the first nobel prize in physics",
        "Artane Industrial School founding year",
        "Swan Lake ballet composer",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        try:
            data = test_raw_api([q])
            raw_results = data.get("result", [])

            if not raw_results:
                print("  ❌ result key missing or empty")
                continue

            # raw_results[0] = list of docs for the first (only) query
            docs_for_query = raw_results[0]
            print(f"  Docs returned: {len(docs_for_query)}")

            if not docs_for_query:
                print("  ❌ No documents found for this query")
                continue

            formatted = passages2string(docs_for_query)
            result_text = json.dumps({"result": formatted})

            print(f"  ✅ result_text length: {len(result_text)} chars")
            print(f"  Preview: {formatted[:200]!r}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
