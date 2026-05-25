# Security Notes

## GitLab Token Exposure Window (git push subprocess)

When `CodeExecutionEngine._git_commit_and_push()` pushes to GitLab it injects
the `GITLAB_TOKEN` via the `GIT_CONFIG_*` environment variables (git ≥ 2.31):

```
GIT_CONFIG_COUNT=1
GIT_CONFIG_KEY_0=http.extraHeader
GIT_CONFIG_VALUE_0=Authorization: Bearer <token>
```

**Trade-off acknowledged:** On Linux, `GIT_CONFIG_VALUE_0` (and therefore the
token) is readable from `/proc/<pid>/environ` for the lifetime of the `git push`
subprocess — up to 120 seconds. Any process running under the same UID, or root,
can read it during that window.

**Why this approach was chosen:**
- Subprocess `argv` (`git -c http.extraHeader=…`) is visible to all users via
  `/proc/<pid>/cmdline` and `ps` — a wider exposure than env vars.
- Writing to `.git/config` persists the token indefinitely on disk.
- `GIT_CONFIG_*` env vars are process-scoped and never hit disk.

**Mitigations in place:**
- The push env is filtered to an explicit allowlist (`_PUSH_ENV_ALLOWLIST`);
  other secrets in the parent process (`AWS_*`, other tokens) are not forwarded.
- `GIT_TRACE*` vars are explicitly excluded so the token is not written to trace
  log files.
- The push timeout is capped at 120 s; the exposure window is bounded by this.
- The parent process's `os.environ` is never mutated.

**Residual risk:** On shared CI runners with multiple untrusted jobs, a
compromised sibling job could potentially read `/proc/<pid>/environ` of the push
subprocess during its 120-second window. If this threat model applies to your
deployment, consider a credential helper backed by a temp file (mode 0600,
deleted on exit) or a short-lived OAuth token scoped to a single push.
