# Storage and identity

SQLite is the source of truth. Schema version **5** is tracked by `PRAGMA user_version`; migration runs transactionally when the version changes. Existing unified rows are not backfilled again on every request. Connections enable WAL and a busy timeout.

## Tables

| Domain | Main tables |
| --- | --- |
| Subscriptions | `feeds`, `feed_groups`, `group_memberships` |
| Library | `entries`, `entry_contents`, `entry_states`, `entry_sources`, `entry_search` |
| Fetch and discovery | `fetch_logs`, `discovery_cache`, `icons`, refresh scheduling tables |
| Notifications | settings, signatures, events, digests and audit state |
| Sync | `sync_events`, `sync_receipts`, `sync_entity_versions` |
| Configuration | `app_settings` |

Legacy `items` tables remain for migration compatibility. Obsolete push endpoint schema also remains harmlessly in old migrations; there is no active remote push implementation. Removing historical tables needs a separate, tested data migration.

## Identity and content

New parsed article IDs are SHA256-derived from the feed identity and external GUID, falling back to URL/content identity. Manual entries use UUIDs. Empty GUIDs do not count as identities. Refresh resolves existing same-source GUID/URL rows before inserting, preserving pre-upgrade IDs and read state. Cross-source ID collisions fail instead of overwriting another feed.

`entry_contents.content_origin` distinguishes feed text from explicitly captured page text. A feed update cannot silently replace a saved full-text body. Feed/body updates and FTS maintenance share the storage path. FTS5 covers normal token queries; CJK queries also use a literal substring fallback. That fallback can scan matching content and is not a benchmarked large-library search engine.

Local user mutations and their sync journal writes are atomic. Replay receipt/version writes share a transaction with the applied change; see [sync](sync.md).

## Data directories and backups

An explicit `INFOMATRIX_DB_PATH` selects a separate database. Otherwise Rust uses the platform application-data location and preserves an existing legacy location. Apple mobile uses Foundation Application Support; Flutter Android resolves Application Support before opening native storage. Tests use temporary directories.

SQLite WAL is live data. Use SQLite's backup API for a consistent running-app backup, or close the app before copying the database and its sidecars together. Migration is forward-only; keep a backup before opening a valuable library with a new development build. OPML exports subscriptions, not notes, read states or captured content.
