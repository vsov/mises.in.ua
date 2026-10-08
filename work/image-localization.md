# Article image localization

## Confirmed decisions

- Download externally hosted article images and replace their source URLs with local URLs.
- Prefer a separate Git branch for this migration.
- For an unavailable image, search archive.org for a capture strictly before the article's Hugo `date`.
- If no suitable capture is available, preserve the original URL and include the image in an unresolved-images report for a later decision.
- Include displayed Markdown/HTML images and linked full-size versions. Preserve original image formats and quality.
- Commit the current site sources, theme, and static assets on `master`, then migrate on `codex/localize-article-images` in an isolated worktree.
- Store images under `static/images/articles/` so every Hugo build includes them.

## Current evidence

- At least 845 Markdown articles contain inline external image URLs.
- The current Git branch is `master`; the checkout contains substantial uncommitted changes.
- The theme's `lead-image.html` reads the first image URL from rendered article content.
- Hugo serves files from `static/` under the corresponding site-root URLs.

## Migration workflow

The baseline site snapshot is commit `3c7f3a58d` on `master`. Migration work lives on
`codex/localize-article-images` in an isolated worktree.

Requirements: Hugo, Python 3 with Pillow, and curl. The inventory uses Hugo's
rendered article bodies and publication dates, including drafts and future articles.
Images are stored without conversion, named by SHA-256 of their contents, and
deduplicated by those bytes.

```sh
hugo --buildDrafts --buildFuture --destination /private/tmp/mises-images-inventory
python3 scripts/localize_images.py inventory --render-root /private/tmp/mises-images-inventory
python3 scripts/localize_images.py download
python3 scripts/localize_images.py apply
hugo --buildDrafts --buildFuture --destination /private/tmp/mises-images-verified
python3 scripts/localize_images.py verify --render-root /private/tmp/mises-images-verified
```

Keep `work/image-localization-manifest.json`: it records original URLs, dates,
download sources, archive capture timestamps, local filenames, and checksums.
Do not rerun inventory over the migrated sources: it deliberately refuses to
overwrite existing provenance. Download/recovery saves progress for resuming.

Retry unresolved archive requests later, then update references and rebuild:

```sh
python3 scripts/localize_images.py recover
python3 scripts/localize_images.py apply
```

`work/unresolved-images.md` contains articles, original URLs, live errors,
and archive outcomes. An archive service error is distinct from a successful
search that found no usable capture. Linked HTML pages (for example, Amazon
product pages and Wikimedia file-description pages) remain ordinary external links.

The migration was merged into `master` and published to `mises@prx:html/` on 2026-10-08.

## Slow Wayback recovery

After publication, run `scripts/recover_wayback_slow.py` on the main checkout.
It makes serial requests, with a 180-second timeout (also for image replay),
a 60-second gap between HTTP requests, and at most 10 slow attempts per URL/Hugo-date pair.
Retries proceed in passes so every unresolved image gets a chance before the next pass.
Attempt history is persisted in the manifest and is retained on restart.

```sh
python3 -u scripts/recover_wayback_slow.py --attempts 10 --timeout 180 --pause 60
```

The worker uses an exclusive process lock. Runtime status and logs live in
`.image-localization/` (ignored by Git). A `SIGTERM` requests a graceful stop;
an active HTTP request may need to finish its timeout first.
Successfully recovered images are saved and their article references are updated locally.
If an article was independently edited since the inventory, the worker preserves it
and reports `downloaded_not_applied` for manual reconciliation.
Subsequent recovery results are not automatically published to production.

## Completed run

- Inventory: 852 articles, 1210 candidate URLs, including 3 linked HTML pages preserved as ordinary links.
- Successful live image URLs: 874; successful archive recoveries before Hugo publication dates: 6.
- Stored unique image files: 878, without re-encoding or resizing.
- Modified articles: 585; localized article/URL pairs: 922.
- Unresolved article/URL pairs: 332. Archive outcomes include 327 service errors and 4 searches with no usable replay; these are not claims that all corresponding captures are absent.
- Hugo build: 2513 pages, 880 static files (878 images and the existing 2 PDFs).
- Validation: all 878 file checksums and generated-file copies match; rendered local references are clickable/displayed; remaining external image URLs match the unresolved inventory.
- An independent comparison with the baseline verified that all 852 article texts are unchanged outside image references and link markup required for bare URLs.
- Nine focused migration checks pass, including strict archive cutoff, redirect validation, preservation of image-identity query parameters, deduplication, and resumable application.
