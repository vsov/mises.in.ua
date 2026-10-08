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
