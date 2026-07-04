
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
