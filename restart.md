# Reconnecting to pinn_gan_selfheating

## From Windows laptop

1. Open VS Code.
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → `fc8engele01.gfoundries.com`
3. Enter Windows/AD password + 6-digit MFA code.
4. Wait for green **SSH: fc8engele01.gfoundries.com** pill (bottom-left).
5. **File** → **Open Folder** → `$PROJECT_ROOT`
   (e.g., `/proj/<cluster>/<user>/python_dev/pinn_gan_selfheating` on the shared filesystem)

## In terminal

```bash
export PROJECT_ROOT=/path/to/pinn_gan_selfheating   # set once per session
cd $PROJECT_ROOT
source .venv/bin/activate

# Smoke test — runs the baseline PINN in ~30s
python -m src.train
```

Expected: final line prints `Relative L2 error vs analytical: ~3e-05`.
Plot saved to `outputs/pinn_first_run.png`.

## Enterprise Linux quirk — home dir permissions

GF FC8ENG periodically resets `~` from `drwxr-xr-x` back to `drwxrwxr-x`
(group-writable), which breaks SSH strict mode and disables key auth.

**Fix (already applied):** `~/.bash_profile` runs the correction on every login:

```bash
chmod g-w,o-w ~ 2>/dev/null
```

**Verify anytime with:** `ls -ld ~`
- Should show `drwxr-xr-x`
- If shows `drwxrwxr-x`, run `chmod g-w,o-w ~` and re-test SSH keys

**Note:** `~/.bashrc` is a read-only symlink to GF's shared template.
Never try to edit it. Use `~/.bash_profile` for personal customizations.

## Git behind GF corporate proxy (one-time global config)

```bash
git config --global http.proxy  "http://uswwwp1.gfoundries.com:80"
git config --global https.proxy "http://uswwwp1.gfoundries.com:80"
git config --global credential.helper store
```

If `git push` still tries a VS Code helper socket, unset the injected env vars
before pushing:

```bash
unset GIT_ASKPASS VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_ASKPASS_NODE \
      VSCODE_GIT_ASKPASS_EXTRA_ARGS VSCODE_GIT_IPC_HANDLE
```

Use a GitHub Personal Access Token (`repo` scope) as password on first push;
credentials cache to `~/.git-credentials` afterward. Secure with `chmod 600 ~/.git-credentials`.