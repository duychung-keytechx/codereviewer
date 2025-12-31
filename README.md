# Code Diff Analyzer

A Python tool to analyze code changes between git branches and extract detailed information from infiniloom codebase maps.

## Features

- Compares two git branches to identify changed files
- Generates a codebase map using infiniloom
- Extracts relevant information for changed files from the codebase map
- Provides detailed output of changes and their context

## Prerequisites

- Python 3.6+
- Git
- infiniloom (must be installed and available in PATH)
- A git repository that's already checked out
- python-dotenv package

## Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env and set your REPO_PATH
```

3. Make the script executable:

```bash
chmod +x main.py
```

## Usage

### Basic Usage

The tool reads `REPO_PATH` from the `.env` file automatically:

```bash
# Ensure your .env file contains: REPO_PATH=/path/to/your/repo
./main.py source_branch destination_branch
```

### Alternative: Specify repo path directly

```bash
./main.py source_branch destination_branch --repo-path /path/to/your/repo
```

### Examples

```bash
# Compare main branch with feature branch (using .env)
./main.py main feature/new-feature

# Compare using explicit repo path (bypasses .env)
./main.py develop staging --repo-path /var/projects/myapp

# Compare branches in current directory
./main.py origin/main HEAD --repo-path .
```

## Output

The tool provides three main outputs:

1. **Execution Progress**: Shows steps being executed
2. **Changed Files List**: Numbered list of all files that changed between branches
3. **File Details**: Information extracted from the codebase map for each changed file

## Configuration

The tool uses a `.env` file to store configuration:

- `REPO_PATH`: Path to the git repository (required if not using --repo-path flag)

Example `.env` file:
```
REPO_PATH=/Users/username/projects/myrepo
```

## How It Works

1. Loads configuration from `.env` file using python-dotenv
2. Validates the repository path from .env or --repo-path argument
3. Executes `infiniloom map . --output codebase-map.txt` in the repo directory
4. Runs `git diff --name-only source...destination` to get changed files
5. Parses the generated codebase-map.txt file
6. Extracts and displays information for each changed file

## Error Handling

The tool will exit with an error if:
- REPO_PATH is not set in .env file and --repo-path is not provided
- The repository path doesn't exist
- infiniloom is not installed or not in PATH
- Git commands fail (invalid branches, not a git repo, etc.)
- The codebase map file cannot be generated or read

## Return Value

When used as a module, the `main()` function returns a dictionary with:

```python
{
    'changed_files': ['file1.py', 'file2.js', ...],
    'file_info': {
        'file1.py': ['...context from codebase map...'],
        'file2.js': ['...context from codebase map...'],
    }
}
```

## Notes

- The git diff uses three-dot syntax (`source...destination`) to show changes between the common ancestor and destination
- The codebase map file is generated in the repository root
- File matching in the codebase map includes context lines before and after mentions
