import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import localize_images as migration
import recover_wayback_slow as slow


class ImageMigrationTests(unittest.TestCase):
    def test_rendered_article_images_and_full_size_links(self):
        parser = migration.ArticleImages()
        parser.feed('<img src="https://site/header.png"><div class="article-body prose">'
                    '<a href="https://site/full"><img src="https://site/thumb.png"></a>'
                    '<picture><source srcset="https://site/a.webp 1x, https://site/b.webp 2x"></picture>'
                    '<a href="https://site/chart.png">Chart</a></div>'
                    '<img src="https://site/footer.png">')
        self.assertEqual(parser.images, {"https://site/full", "https://site/thumb.png",
                                        "https://site/a.webp", "https://site/b.webp", "https://site/chart.png"})

    def test_preserve_longer_urls_and_map_encoded_attributes(self):
        source = '![x](https://site/a.png) [citation](https://site/a.png?extra=1) <img src="https://site/b?a=1&amp;b=2">'
        spans = migration.source_spans(source, ["https://site/a.png", "https://site/b?a=1&b=2"])
        self.assertEqual(len(spans), 2)
        self.assertEqual(source[spans[1][0]:spans[1][1]], "https://site/b?a=1&amp;b=2")
        self.assertEqual(len(migration.source_spans("https://site/a.png\u00a0", ["https://site/a.png"])), 1)

    def test_timezone_and_strict_cutoff(self):
        self.assertEqual(migration.cutoff("2020-01-02T01:00:00+03:00"), "20200101215959")

    def test_reject_html_even_when_download_succeeds(self):
        with self.assertRaises(Exception):
            migration.store_image(b"<html>Access denied</html>")

    def test_identical_bytes_deduplicate(self):
        body = io.BytesIO()
        Image.new("RGB", (3, 3)).save(body, "PNG")
        with tempfile.TemporaryDirectory() as temporary, patch.object(migration, "ASSETS", Path(temporary)):
            first = migration.store_image(body.getvalue())
            second = migration.store_image(body.getvalue())
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)

    def test_archive_replay_must_not_redirect_past_cutoff(self):
        responses = [(b'[["timestamp","original","mimetype","statuscode"],["20191231000000","https://site/a.png","image/png","200"]]', {}),
                     (b"not important", {"url_effective": "https://web.archive.org/web/20210101000000id_/https://site/a.png"})]
        with patch.object(migration, "request", side_effect=responses), patch.object(migration, "store_image") as store:
            result = migration.fetch_archive("https://site/a.png", "2020-01-01T00:00:00Z", 5)
        self.assertEqual(result["status"], "no_usable_capture")
        store.assert_not_called()

    def test_archive_search_preserves_image_identity_parameters(self):
        with patch.object(migration, "request", return_value=(b"[]", {})) as request:
            result = migration.fetch_archive("https://site/cover?doc=book-one&width=200", "2020-01-01T00:00:00Z", 5)
        self.assertEqual(result["status"], "no_usable_capture")
        self.assertEqual(request.call_count, 1)
        self.assertIn("doc%3Dbook-one", request.call_args.args[0])

    def test_slow_replay_uses_the_full_timeout(self):
        body = io.BytesIO()
        Image.new("RGB", (3, 3)).save(body, "PNG")
        responses = [(b'[["timestamp","original","mimetype","statuscode"],["20191231000000","https://site/a.png","image/png","200"]]', {}),
                     (body.getvalue(), {"url_effective": "https://web.archive.org/web/20191231000000id_/https://site/a.png"})]
        with tempfile.TemporaryDirectory() as temporary, patch.object(migration, "ASSETS", Path(temporary)), patch.object(migration, "request", side_effect=responses) as request:
            restored = migration.fetch_archive("https://site/a.png", "2020-01-01T00:00:00Z", 180, replay_timeout=180)
        self.assertEqual(restored["source"], "archive")
        self.assertEqual(request.call_args.args[1], 180)

    def test_slow_attempt_limit_is_preserved_on_restart(self):
        article = {"date": "2020-01-01T00:00:00Z", "urls": ["https://site/a.png", "https://site/b.png", "https://site/c.png"]}
        results = {"https://site/a.png": {"slow_attempts": {article["date"]: [{}] * 10}},
                   "https://site/b.png": {"slow_attempts": {article["date"]: [{}] * 9}},
                   "https://site/c.png": {"live": {"local_url": "/images/c.png"}}}
        self.assertEqual(slow.pending({"articles": [article], "results": results}, 10), [("https://site/b.png", article["date"])])

    def test_slow_worker_spaces_http_requests(self):
        state = {}
        with patch.object(slow, "write_state"), patch.object(slow.time, "time", return_value=100), patch.object(slow.STOP, "wait", return_value=False) as wait, patch.object(slow.images, "request", return_value=(b"ok", {})):
            fetcher = slow.PoliteRequests(60, state)
            fetcher("https://archive.test/one", 180)
            fetcher("https://archive.test/two", 180)
        self.assertEqual([call.args[0] for call in wait.call_args_list], [0, 60])

    def test_apply_can_finish_unresolved_images_after_a_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = '![one](https://site/a.png) ![two](https://site/b.png)'
            path = root / "article.md"
            path.write_text(original)
            image = root / "static/images/one.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"already validated image bytes")
            resolved = {"local_url": "/images/one.png", "sha256": migration.sha(image.read_bytes())}
            article = {"path": "article.md", "title": "Article", "date": "2020-01-01T00:00:00Z", "urls": ["https://site/a.png", "https://site/b.png"],
                       "unmatched": [], "source_sha256": migration.sha(original.encode())}
            data = {"articles": [article], "results": {"https://site/a.png": {"live": resolved}}}
            with patch.object(migration, "ROOT", root), patch.object(migration, "save"), patch.object(migration, "report"):
                migration.apply(data)
                self.assertIn("https://site/b.png", path.read_text())
                data["results"]["https://site/b.png"] = {"live": resolved}
                migration.apply(data)
                self.assertNotIn("https://", path.read_text())
                migration.apply(data)
                self.assertEqual(path.read_text().count("/images/one.png"), 2)

    def test_local_references_are_collected_from_attributes(self):
        parser = migration.ArticleImages()
        parser.feed('<div class="article-body prose"><u>/images/articles/bare.png</u>'
                    '<a href="/images/articles/linked.png">graph</a>'
                    '<img src="/images/articles/embedded.png"></div>')
        self.assertEqual(parser.local_references, {"/images/articles/linked.png", "/images/articles/embedded.png"})

    def test_scoped_application_preserves_unrelated_user_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.md"
            good.write_text("![image](https://site/a.png)")
            edited = root / "edited.md"
            edited.write_text("User changed this article")
            image = root / "static/images/one.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"validated image")
            articles = [{"path": path, "title": path, "date": "2020-01-01T00:00:00Z", "urls": ["https://site/a.png"], "unmatched": [], "source_sha256": original}
                        for path, original in [("good.md", migration.sha(good.read_bytes())), ("edited.md", migration.sha(b"Old article"))]]
            data = {"articles": articles, "results": {"https://site/a.png": {"live": {"local_url": "/images/one.png", "sha256": migration.sha(image.read_bytes())}}}}
            with patch.object(migration, "ROOT", root), patch.object(migration, "save"), patch.object(migration, "report"):
                migration.apply(data, only_paths={"good.md"})
            self.assertIn("/images/one.png", good.read_text())
            self.assertEqual(edited.read_text(), "User changed this article")


if __name__ == "__main__":
    unittest.main()
