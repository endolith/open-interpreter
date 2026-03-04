# Test fixtures

## Sanitize-secrets test (no real credentials)

`sanitize_secrets_test.txt` contains:

- **MUST STAY VISIBLE:** Long paths and env vars (APPDATA, TEMP, PROJECT_DIR, PATH, USERPROFILE, etc.), including paths with random-looking segments that were over-redacted before we excluded high-entropy detectors. These must *not* appear as `[REDACTED]`.
- **MUST REDACT:** Fake secrets in two groups:
  - Env-style `NAME=VALUE` where NAME ends in `_KEY`, `_SECRET`, `_TOKEN`, `_PASSWORD` (custom EnvSecretDetector).
  - Format-specific lines that match built-in detectors: AWS, GitHub, Slack, Basic auth, Stripe, SendGrid, Twilio, Discord, JWT (and KeywordDetector in code-style snippets).

To test:

1. Start Open Interpreter with a **remote** model (so sanitization is on).
2. Ask the model to run one of:
   - **Windows:** `type tests\fixtures\sanitize_secrets_test.txt`
   - **Unix/macOS:** `cat tests/fixtures/sanitize_secrets_test.txt`
   - Or from Python: `print(open("tests/fixtures/sanitize_secrets_test.txt").read())`
3. On the **next** turn (or in the chat log), check:
   - Paths and non-secret env vars (USERPROFILE, PATH, HOME, TEMP, PROJECT_DIR, etc.) are still visible.
   - All values in the "MUST REDACT" sections appear as `[REDACTED]` (variable names can stay).

All content is fake and for testing only.
