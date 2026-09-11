# Security and data boundaries

Report sensitive vulnerabilities privately through the repository owner's GitHub
profile. Do not include credentials or private document contents in public issues.

This is a single-user application bound to loopback. It has no authentication,
tenant isolation, job sandbox, or rate-limiting service. Host and Origin validation
restrict browser access, and the UI renders document text without HTML execution.
Do not expose it directly to the public internet. Use trusted documents; PDF parsing
has input limits but is not isolated from malicious decompression or parser exploits.

The default provider performs no network calls. Enabling a remote generation
provider sends queries and retrieved passages to that endpoint. API keys stay in
the server environment. Responses from providers have time and size limits, and
errors returned to clients omit upstream response bodies.

Retrieved content is treated as untrusted input. Exact citation checks reject
invented sources and quotations, but do not prove semantic support or eliminate
prompt injection. Models have no tools, filesystem access, or side-effect capability.

SQLite contains unencrypted extracted text, embeddings and cached answers. Deletion
removes searchable records and invalidates caches; it is not secure erasure of
SQLite pages, filesystem snapshots or backups. Protect the data directory using
operating-system permissions and disk encryption where appropriate.
