#!/bin/zsh

# Define source directory
export SRC_DIR=$(dirname "${0:A}")

# Find all Python files in the directory and its subdirectories, excluding .venv directories
FILES=($(find "${SRC_DIR}" -type f -name "*.py" -not -path "*.venv*"))

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel is not installed. Processing files sequentially."
    # Iterate through each file and run the commands
    for FILE in "${FILES[@]}"; do
        echo "Processing $FILE"
        autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$FILE"
        ruff format "$FILE"
        isort "$FILE"
    done
else
    # Process files in parallel
    echo "Processing ${#FILES[@]} files in parallel..."
    
    # Create a temporary script for processing
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/zsh
FILE="$1"
echo "Processing $FILE"
autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$FILE"
ruff format "$FILE"
isort "$FILE"
EOF
    chmod +x "$TEMP_SCRIPT"
    
    # Run in parallel
    printf "%s\n" "${FILES[@]}" | parallel "$TEMP_SCRIPT"
    
    # Clean up
    rm "$TEMP_SCRIPT"
fi