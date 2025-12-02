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

        url = f"{self.api_url}/search/current"
        params = {
            "apiKey": self.api_key,
            "string": canonical_form,
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
                self._cache[canonical_form] = cui
                return cui
            self._cache[canonical_form] = None
            return None
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 401:
                logger.error("UMLS API authentication failed. Check API key validity.")
            else:
                logger.warning("UMLS search failed for '%s': %s", canonical_form, exc)
            self._cache[canonical_form] = None
            return None
        except Exception as exc:
            logger.warning("UMLS search failed for '%s': %s", canonical_form, exc)
            self._cache[canonical_form] = None
            return None

