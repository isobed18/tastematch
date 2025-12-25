
# 1. Deactivate
conda deactivate

# 2. Remove old env
Write-Host "Removing old environment..."
conda env remove --name tastematch_env -y

# 3. Create new env with Python 3.13
Write-Host "Creating new environment (Python 3.13)..."
conda create --name tastematch_env python=3.13 -y

# 4. Activate logic (tricky in script, better to run commands using run -n)
Write-Host "Installing PyTorch..."
conda run -n tastematch_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "Installing Requirements..."
conda run -n tastematch_env pip install -r backend/requirements.txt

Write-Host "Installing Additional Deps..."
conda run -n tastematch_env pip install tabulate cffi


Write-Host "Done! Please run 'conda activate tastematch_env' manually."
