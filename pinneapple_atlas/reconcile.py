"""Conservative cross-source publication matching based on canonical DOI only."""
from .store import canonical_doi


def doi_equivalences(store):
    """Return pairs with identical normalized DOIs, never title-only guesses."""
    groups = {}
    for row in store.connection.execute("SELECT source, external_id, doi FROM records WHERE doi IS NOT NULL"):
        doi = canonical_doi(row["doi"])
        if doi:
            groups.setdefault(doi, set()).add((row["source"], row["external_id"]))
    matches = []
    for doi, identities in sorted(groups.items()):
        ordered = sorted(identities)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if left[0] != right[0]:
                    matches.append({"left": left, "right": right, "doi": doi,
                                    "relation": "same_publication_doi",
                                    "evidence_url": "https://doi.org/" + doi})
    return matches


def reconcile_dois(store):
    matches = doi_equivalences(store)
    for item in matches:
        store.relate(*item["left"], *item["right"], item["relation"], item["evidence_url"])
    return len(matches)
