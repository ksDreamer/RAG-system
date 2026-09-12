# Architecture

InfoMatrix has one persistent library and two presentation shells. A hosted service is not required.

```text
SwiftUI → NativeReaderService actor → detached C calls ┐
Flutter → FfiReaderBackend → DB and network isolates   ├→ ffi_bridge
Optional app_server (HTTP) ───────────────────────────┘      ↓
                                                app_core + shared_api
                                                         ↓
                         discovery / fetcher / parser / icon / opml
                                                         ↓
                                  storage (SQLite, FTS5, sync replay)
```

## Responsibilities

`app_core` owns common subscription, entry and state operations. `shared_api` owns refresh orchestration, notification mapping, URL normalization and content extraction shared by FFI and HTTP. `storage` owns SQL transactions, migrations, search and sync replay. `notifications` provides scheduling, policy, digest coordination and audit helpers; delivery belongs to a platform adapter.

The former empty `search` and `sync` crates and no-op push/delivery scaffolds were removed. Search and replay now live next to the SQL they depend on. The HTTP and C entry points use the same refresh implementation so failed requests produce the same persisted backoff.

## Concurrency

Native calls are blocking at the C boundary. Swift's actor dispatches each call to a detached task, avoiding main-actor work while allowing other calls during network waits. Flutter has one persistent database worker and one persistent network worker; the UI isolate only sends JSON messages and renders replies. SQLite uses WAL, a busy timeout and short transactions. Multiple connections can still contend on writes; the app does not promise unlimited parallel imports.

Shell state uses request generations to prevent stale search/detail replies from replacing a newer selection. Busy state is counted across overlapping requests. A successful operation has a status message rather than masquerading as an error.

## Ingestion and reading

A direct subscription validates the feed and saves its initial items. Website discovery returns ranked candidates and diagnostics, with cached parsed snapshots for reuse. Feed identity includes the source; two feeds can reuse a GUID without overwriting each other. Existing stored IDs are preserved when their source and external identity match.

Feed refresh uses HTTP validators, stores attempts and failure schedules, then applies notification policy. A failed feed does not prevent the remaining due feeds from running. Captured full text has its own origin marker and survives a routine feed refresh.

Apple uses a restricted local HTML view; remote images need explicit opt-in. Flutter presents plain text. Notes are editable and deletable through the same Rust storage contracts. The OS controls background lifetime, so refresh is not guaranteed after the app is suspended or closed.

## Boundaries still being developed

CloudKit is an optional Apple adapter, with local replay tested independently. Native notifications exist on macOS; Flutter notification delivery and a non-Apple remote sync adapter are not implemented. Large shell view files remain a maintenance target, but business logic is kept out of them. See [validation and remaining debt](modernization.md).
