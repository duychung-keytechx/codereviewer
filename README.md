# Code Diff Analyzer

An AI-powered code review tool that analyzes Java code changes between git branches using Claude Code and infiniloom.

## Features

- Compares two git branches to identify changed files
- Automatically fetches remote branches if not available locally
- Uses Claude Code AI to analyze git diffs and identify dependency classes
- Packs related Java classes using infiniloom for LLM context
- Generates comprehensive AI-powered code reviews with Claude Code
- Reviews focus on code quality, potential bugs, performance, security, and improvements
- Outputs diff, dependencies, prompts, and review results to a dedicated output directory

## Prerequisites

Before using this tool, you must install the following:

### Required Tools

1. **Python 3.9.x or above**
   ```bash
   python --version  # Should be 3.9.x or higher
   ```

2. **Git**
   ```bash
   git --version
   ```

3. **infiniloom** - Code packing tool for LLM consumption
   - Installation: https://github.com/Topos-Labs/infiniloom
   - Must be available in PATH
   ```bash
   infiniloom --version  # Verify installation
   ```

4. **Claude Code** - Claude AI CLI tool
   - Must be installed and authenticated
   - Used for AI-powered diff analysis and code reviews
   ```bash
   claude --version  # Verify installation
   ```

### Python Dependencies

- python-dotenv

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

# Specify custom output directory
./main.py main feature/auth --output-dir my-output
```

## Output

The tool generates multiple files in the output directory (default: `output/`):

1. **Console Output**: Shows execution progress and summary
2. **diff.txt**: Raw git diff between the two branches
3. **llm.txt**: Infiniloom-packed file containing identified dependency classes (uses toon format with balanced compression)
4. **prompt.txt**: The final prompt sent to Claude Code for review
5. **code_review_result.md**: AI-generated code review from Claude Code

The infiniloom output uses toon format with balanced compression, optimized for LLM consumption.

## Configuration

The tool uses a `.env` file to store configuration:

- `REPO_PATH`: Path to the git repository (required if not using --repo-path flag)

Example `.env` file:
```
REPO_PATH=/Users/username/projects/myrepo
```

## How It Works

1. **Setup & Validation**
   - Loads configuration from `.env` file using python-dotenv
   - Validates the repository path from .env or --repo-path argument
   - Checks if branches exist locally, fetches from remote if needed

2. **Diff Analysis**
   - Runs `git diff source...destination` to get full diff between branches
   - Saves raw diff to `diff.txt`

3. **AI-Powered Dependency Detection**
   - Sends the diff to Claude Code CLI for analysis
   - Claude identifies relevant dependency classes that provide context for the changes
   - Excludes library classes and focuses on project-specific dependencies

4. **Context Packing**
   - Checks out the destination branch
   - Uses infiniloom to pack identified dependency classes:
     - Format: `toon`
     - Compression: `balanced`
     - Options: `--remove-comments --remove-empty-lines --no-symbols`
     - Max tokens: 16000
   - Saves packed context to `llm.txt`

5. **AI Code Review**
   - Constructs a comprehensive prompt with:
     - The git diff
     - Packed dependency classes as reference context
   - Sends to Claude Code for review focusing on:
     - Code quality and best practices
     - Potential bugs or issues
     - Performance considerations
     - Security concerns
     - Suggestions for improvements
   - Saves review to `code_review_result.md`

6. **Output**
   - All files saved to the `output/` directory (or custom directory via --output-dir)
   - Summary printed to console

## Error Handling

The tool will exit with an error if:
- REPO_PATH is not set in .env file and --repo-path is not provided
- The repository path doesn't exist
- infiniloom is not installed or not in PATH
- Claude Code CLI (`claude` command) is not installed or not in PATH
- Git commands fail (invalid branches, not a git repo, etc.)
- The infiniloom pack command fails

The tool will continue with warnings if:
- No dependency classes are identified by Claude (review will proceed without context)
- Claude Code requests timeout (partial results may be available)
- Infiniloom packing fails (review will proceed without packed context)

## Return Value

When used as a module, the `main()` function returns a dictionary with:

```python
{
    'diff_file': '/path/to/output/diff.txt',
    'classes': ['com/example/Class1', 'com/example/Class2', ...],
    'output_file': '/path/to/output/llm.txt',
    'prompt_file': '/path/to/output/prompt.txt',
    'review_file': '/path/to/output/code_review_result.md',
    'review': '... full review text ...'
}
```

## Notes

- The git diff uses three-dot syntax (`source...destination`) to show changes between the common ancestor and destination
- The tool focuses on Java files but analyzes the full diff for context
- Claude Code AI is used twice in the workflow:
  1. To identify relevant dependency classes from the diff
  2. To perform the final code review
- If branches are not found locally, the tool automatically runs `git fetch --all`
- The tool automatically checks out the destination branch to pack dependency classes
- The output directory is created automatically if it doesn't exist
- Claude Code requests have timeouts: 3 minutes for dependency analysis, 5 minutes for code review
- The infiniloom pack command uses specific options optimized for LLM consumption:
  - Removes comments and empty lines
  - Excludes symbols
  - Limits output to 16000 tokens
