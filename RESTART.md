
# Reconnecting tomorrow to pinn_gan_selfheating

## From Windows laptop
1. Open VS Code.
2. Ctrl+Shift+P -> Remote-SSH: Connect to Host -> fc8engele01.gfoundries.com
3. Enter Windows/AD password + 6-digit MFA code.
4. Wait for green SSH: fc8engele01.gfoundries.com pill (bottom-left).
5. File -> Open Folder -> /proj/cmc3/cmc_users/sroy5/python_dev/pinn_gan_selfheating

## In terminal
```bash
cd /proj/cmc3/cmc_users/sroy5/python_dev/pinn_gan_selfheating
source .venv/bin/activate
python src/hello.py

## Enterprise Linux quirk — home dir permissions

GF FC8ENG periodically resets `~` from `drwxr-xr-x` back to `drwxrwxr-x`
(group-writable), which breaks SSH strict mode and disables key auth.

**Fix (already applied):** `~/.bash_profile` runs the correction on every login:

    chmod g-w,o-w ~ 2>/dev/null

**Verify anytime with:** `ls -ld ~`
- Should show `drwxr-xr-x`
- If shows `drwxrwxr-x`, run `chmod g-w ~` and re-test SSH keys

**Note:** `~/.bashrc` is a read-only symlink to GF's shared template.
Never try to edit it. Use `~/.bash_profile` for personal customizations.
