from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class UMLSClient:
    def __init__(self, api_key: str, api_url: str = "https://uts-ws.nlm.nih.gov/rest"):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self._cache: dict[str, Optional[str]] = {}

    def search_concept(self, canonical_form: str) -> Optional[str]:
        if not canonical_form or canonical_form.strip().lower() == "none":
            return None

        canonical_form = canonical_form.strip()
        if canonical_form in self._cache:
            return self._cache[canonical_form]

        # Try multiple search strategies
        search_terms = [
            canonical_form,  # Original canonical form
            canonical_form.lower(),  # Lowercase
            canonical_form.replace(" neoplasms", " cancer"),  # Try "cancer" instead of "neoplasms"
            canonical_form.replace(" cancer", " neoplasms"),  # Try "neoplasms" instead of "cancer"
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term and term not in seen:
                seen.add(term)
                unique_terms.append(term)

        url = f"{self.api_url}/search/current"
        
        for search_term in unique_terms:
            # Check cache first
            if search_term in self._cache:
                cached_result = self._cache[search_term]
                if cached_result:
                    # Cache the result for the original canonical form too
                    self._cache[canonical_form] = cached_result
                    return cached_result
                continue
            
            params = {
                "apiKey": self.api_key,
                "string": search_term,
                "pageSize": 1,
                "returnIdType": "concept",
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                results = data.get("result", {}).get("results", [])
                if results:
                    cui = results[0].get("ui")
                    # Cache for both the search term and original canonical form
                    self._cache[search_term] = cui
                    self._cache[canonical_form] = cui
                    return cui
                # Cache None result for this search term
                self._cache[search_term] = None
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 401:
                    logger.error("UMLS API authentication failed. Check API key validity.")
                    self._cache[canonical_form] = None
                    return None
                else:
                    logger.debug("UMLS search failed for '%s': %s", search_term, exc)
                self._cache[search_term] = None
            except Exception as exc:
                logger.debug("UMLS search failed for '%s': %s", search_term, exc)
                self._cache[search_term] = None
        
        # All searches failed
        self._cache[canonical_form] = None
        return None

