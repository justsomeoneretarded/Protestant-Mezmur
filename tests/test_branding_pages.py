"""Keep the public OAuth branding pages navigable and explicit about current data handling."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest


DOCS = Path("docs")


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.links, self.images, self.tags, self.text = [], [], [], []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        values = dict(attrs)
        if tag in ("a", "link") and values.get("href"):
            self.links.append(values["href"])
        if tag == "img":
            self.images.append(values)

    def handle_data(self, data):
        self.text.append(data)


@pytest.mark.parametrize("filename", ["index.html", "privacy.html", "terms.html"])
def test_pages_have_brand_identity_no_credential_collection_and_resolvable_local_links(filename):
    page = Page(DOCS / filename)
    assert "Rehoboth Mezmur" in " ".join(page.text)
    assert "form" not in page.tags and "script" not in page.tags
    assert "privacy.html" in page.links and "terms.html" in page.links
    for link in page.links:
        parsed = urlsplit(link)
        if not parsed.scheme and not parsed.netloc:
            target = DOCS / (parsed.path or "index.html")
            assert target.exists(), link


def test_homepage_describes_actual_publishing_and_shows_a_local_renderer_asset():
    page = Page(DOCS / "index.html")
    text = " ".join(page.text)
    assert "YouTube API Services" in text and "four hours" in text
    assert page.images
    for item in page.images:
        assert item.get("alt") and (DOCS / item["src"]).is_file()


def test_privacy_discloses_google_access_public_outputs_revocation_and_contact():
    page = Page(DOCS / "privacy.html")
    text = " ".join(page.text).lower()
    for phrase in ("refresh tokens", "public", "artifacts", "retention", "delete",
                   "support email", "limited use", "general-purpose ai"):
        assert phrase in text
    assert "https://policies.google.com/privacy" in page.links
    assert "https://myaccount.google.com/permissions" in page.links


def test_terms_link_youtube_terms_and_privacy():
    page = Page(DOCS / "terms.html")
    assert "https://www.youtube.com/t/terms" in page.links
    assert "privacy.html" in page.links
    assert (DOCS / ".nojekyll").is_file()


def test_branding_discloses_automatic_timed_amharic_captions():
    home = " ".join(Page(DOCS / "index.html").text).lower()
    privacy = " ".join(Page(DOCS / "privacy.html").text).lower()
    assert "automatically uploads" in home and "amharic lyric subtitle" in home
    assert "timed amharic lyric captions" in privacy and "caption track ids" in privacy
    assert "subtitle hashes" in privacy
    assert "does not upload separate subtitle tracks" not in home
