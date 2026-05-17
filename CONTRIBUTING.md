# Contributing to Olympus Tools

First off, thank you for considering contributing to Olympus Tools! 🎉

The following is a set of guidelines for contributing to our repositories. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Adding New Tools](#adding-new-tools)
  - [Improving Documentation](#improving-documentation)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
```markdown
**Tool Name**: [e.g., neural_tts.py]
**Environment**: [OS, Python/Bash version]
**Description**: [Clear description of the bug]
**Steps to Reproduce**:
1. [First step]
2. [Second step]
3. [See error]
**Expected Behavior**: [What should happen]
**Actual Behavior**: [What actually happens]
**Additional Context**: [Logs, screenshots, etc.]
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How should it work?
- **Alternative solutions**: Other ways to solve it
- **Additional context**: Examples, mockups, etc.

### Adding New Tools

We love new tools! When submitting a new tool:

1. **Ensure it fits** the repository's scope
2. **Follow naming conventions**:
   - Shell scripts: `descriptive-name.sh`
   - Python scripts: `descriptive_name.py`
   - No spaces, clear purpose
3. **Include documentation** in the script header:
   ```bash
   #!/bin/bash
   # Tool: awesome-tool.sh
   # Description: Does awesome things
   # Usage: ./awesome-tool.sh input.txt output.txt
   # Requirements: bash 4.0+, awk
   # Author: Your Name
   # Date: YYYY-MM-DD
   ```
4. **Add usage examples**
5. **Test on Linux and macOS** (if possible)
6. **Update the README** with your tool

### Improving Documentation

Documentation improvements are always welcome! This includes:
- Fixing typos
- Adding examples
- Clarifying usage
- Translating documentation
- Adding diagrams or visualizations

## Development Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-tool`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**:
   ```
   Add neural network motif finder
   
   - Implements LSTM-based motif discovery
   - Supports FASTA input
   - Outputs confidence scores
   - Includes test data
   ```
6. **Push to your fork**: `git push origin feature/amazing-tool`
7. **Submit a Pull Request**

## Style Guidelines

### Shell Scripts

```bash
#!/bin/bash
# Use consistent indentation (2 spaces)
# Check arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

# Use meaningful variable names
input_file="$1"
output_file="${input_file%.txt}_processed.txt"

# Handle errors
set -euo pipefail

# Add progress indicators for long operations
echo "Processing $input_file..."
```

### Python Scripts

```python
#!/usr/bin/env python3
"""
Tool description and usage information.
"""

import argparse
import sys
from typing import List, Optional

def main(args: argparse.Namespace) -> None:
    """Main function with type hints."""
    # Implementation here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument("input", help="Input file")
    parser.add_argument("-o", "--output", help="Output file")
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

### General Guidelines

- **Use descriptive variable names**
- **Comment complex logic**
- **Handle errors gracefully**
- **Provide helpful error messages**
- **Follow existing patterns in the codebase**

## Testing

### Before Submitting

1. **Test your changes**:
   ```bash
   # For shell scripts
   shellcheck your-script.sh
   
   # For Python
   python -m pylint your_script.py
   python -m pytest tests/
   ```

2. **Test on sample data**:
   - Small files
   - Large files
   - Edge cases
   - Invalid input

3. **Check performance**:
   - Time execution on large datasets
   - Monitor memory usage
   - Compare with existing tools

### Test Data

- Keep test files small (< 1MB)
- Use synthetic data when possible
- Never commit sensitive data
- Include both positive and negative test cases

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** if applicable
3. **Ensure all tests pass**
4. **Update the README** if adding new tools
5. **Request review** from maintainers

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Thanked in commit messages

## Questions?

Feel free to:
- Open an issue for discussion
- Contact maintainers
- Join our community discussions

Thank you for contributing! 🙏

---

**Happy coding!** 🚀