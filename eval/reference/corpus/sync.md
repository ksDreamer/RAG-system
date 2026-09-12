# Sync semantics and limits

InfoMatrix works without sync. SQLite maintains the local library and a mutation journal; Apple CloudKit is an optional adapter. Windows, Linux and Android remain local-only.

## Local journal

Entry creation/edit/delete, item states, subscriptions, groups, membership and supported settings write events with stable IDs and timestamps. A mutation and its local event are committed together. Pending events have no `processed_at`; consumers acknowledge only after successful upload.

Incoming events are ordered by timestamp and event ID. `sync_receipts` makes duplicate deliveries harmless. `sync_entity_versions` records the latest accepted event for each entity/domain, providing deterministic last-write-wins behavior. Each event's mutation, receipt and version update are transactional. Unknown or unresolved events fail for retry rather than being acknowledged and lost. Repeated deletions are safe. Item state resolution includes the feed URL so identical GUIDs from different feeds stay separate.

This is coarse last-write-wins, not collaborative editing or a CRDT. Concurrent note edits can replace each other. Local device clock skew can affect conflict outcomes. The journal/receipt tables currently have no compaction; dependency failures can require retry after related events arrive.

## Apple adapter

CloudKit uses `InfoMatrixSyncEvent` records in the configured private database. Uploads use stable record IDs, bounded batches, and inspect each save result before acknowledging. Download pages the remote event set and lets SQLite deduplicate it, avoiding a client-time cursor that could lose delayed offline updates. Full scans grow with history and need a server-change-token/compaction design for large accounts.

The Swift bridge encodes events with `JSONEncoder` before passing JSON through C; native integration tests replay events between two isolated databases. Matching iCloud containers and provisioning must be configured in the owner's Apple account. Simulator and ad-hoc builds do not establish live CloudKit validation. This revision does not claim tested multi-device cloud synchronization.

## Inspection

The optional HTTP API exposes pending events, acknowledgement and replay. The C ABI exposes equivalent JSON operations. The transport must retain failed events and retry; a successful local replay test does not prove cloud permissions, account behavior or mobile background delivery.
