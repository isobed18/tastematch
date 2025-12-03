# Build the Docker image
echo "Building Docker image 'tastematch-fm'..."
docker build -t tastematch-fm .

# Check if build succeeded
if ($LASTEXITCODE -ne 0) {
    echo "Docker build failed."
    exit $LASTEXITCODE
}

# Run the training script inside the container
# -v: Mounts current directory to /app
# --rm: Removes container after exit
echo "Running Training in Docker..."
# -it: Interactive terminal (helps with coloring/tty)
# python -u: Unbuffered output (forces prints to show immediately)
docker run --rm -it -v "${PWD}:/app" tastematch-fm python -u project/fm/train.py --epochs 1 --output_dir runs/docker_run

echo "Done. Check runs/docker_run for outputs."
