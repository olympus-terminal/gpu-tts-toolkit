#!/bin/bash

# Set the input file as the first command-line argument
INPUT_FILE="$1"

# Check if the input file is provided
if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 <input_file>"
  echo "Deletes lines containing '/A' and lines with fewer than 5 characters"
  exit 1
fi

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file '$INPUT_FILE' does not exist"
  exit 1
fi

# Create a temporary file
TEMP_FILE=$(mktemp)

# Delete lines containing STRING and lines with fewer than 5 characters
sed '/STRING/d; /^[[:print:]]\{0,4\}$/d' "$INPUT_FILE" > "$TEMP_FILE"

# Create an output file with a suffix
OUTPUT_FILE="${INPUT_FILE}_cleaned"

# Move the temporary file to the output file
mv "$TEMP_FILE" "$OUTPUT_FILE"

# Report the number of lines deleted
ORIGINAL_LINES=$(wc -l < "$INPUT_FILE")
NEW_LINES=$(wc -l < "$OUTPUT_FILE")
DELETED_LINES=$((ORIGINAL_LINES - NEW_LINES))
echo "Deleted $DELETED_LINES lines from '$INPUT_FILE' to create '$OUTPUT_FILE'"

# Clean up
rm "$TEMP_FILE"
