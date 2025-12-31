#!/usr/bin/env python3
"""
Code Diff Analyzer Tool
Analyzes Java code changes between git branches and packs them using infiniloom.
"""

import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv


def get_repo_path() -> str:
    """Get repository path from environment variable or .env file."""
    # Load .env file if it exists
    load_dotenv()

    repo_path = os.getenv('REPO_PATH')
    if not repo_path:
        raise ValueError("REPO_PATH is not set. Please set it in .env file or environment variable")

    if not os.path.isdir(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")

    return repo_path


def git_ref_exists(repo_path: str, ref: str) -> bool:
    """Check if a git reference (branch, tag, commit) exists."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--verify', ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def fetch_all_branches(repo_path: str) -> None:
    """Fetch all branches from remote repositories."""
    try:
        print("Fetching all branches from remote...")
        result = subprocess.run(
            ['git', 'fetch', '--all'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        print("Successfully fetched all branches")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to fetch branches: {e.stderr}")


def ensure_branch_exists(repo_path: str, branch: str) -> str:
    """Ensure a branch exists locally, fetching if necessary. Returns the full ref name."""
    # Try different ref formats
    possible_refs = [
        branch,                    # local branch or commit hash
        f'origin/{branch}',        # remote branch
        f'refs/heads/{branch}',    # full local ref
        f'refs/remotes/origin/{branch}'  # full remote ref
    ]

    # Check if any ref exists
    for ref in possible_refs:
        if git_ref_exists(repo_path, ref):
            print(f"Found ref: {ref}")
            return ref

    # If not found, try fetching
    print(f"Branch '{branch}' not found locally, fetching from remote...")
    fetch_all_branches(repo_path)

    # Try again after fetch
    for ref in possible_refs:
        if git_ref_exists(repo_path, ref):
            print(f"Found ref after fetch: {ref}")
            return ref

    # Still not found, raise error
    raise ValueError(f"Branch '{branch}' not found even after fetching. Please check the branch name.")


def get_current_branch(repo_path: str) -> str:
    """Get the currently checked out branch."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get current branch: {e.stderr}")


def checkout_branch(repo_path: str, branch_ref: str) -> None:
    """Checkout a specific branch."""
    try:
        print(f"Checking out branch: {branch_ref}")
        result = subprocess.run(
            ['git', 'checkout', branch_ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Successfully checked out {branch_ref}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to checkout branch: {e.stderr}")


def execute_infiniloom_pack_with_classes(repo_path: str, class_list: List[str], output_dir: str) -> str:
    """Execute infiniloom pack command with the list of classes."""
    if not class_list:
        raise ValueError("No classes to pack")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build the output path
    output_file = os.path.join(output_dir, 'llm.txt')

    # Build the command
    cmd = ['infiniloom', 'pack', '.', '--compression', 'balanced', '--format', 'toon',
           '--remove-comments', '--remove-empty-lines', '--no-symbols', '--max-tokens', '16000', '--output',
           output_file]

    # Add --include for each class
    for class_path in class_list:
        cmd.extend(['--include', class_path])

    try:
        print(f"\nExecuting infiniloom pack with {len(class_list)} classes...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"✓ Infiniloom pack completed successfully!")
        print(f"Output saved to: {output_file}")

        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute infiniloom pack: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("infiniloom command not found. Please ensure it's installed and in PATH")


def read_file_content_from_git(repo_path: str, branch_ref: str, file_path: str) -> str:
    """Read and return the content of a file from a git branch."""
    try:
        result = subprocess.run(
            ['git', 'show', f'{branch_ref}:{file_path}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"[Error reading file from git: {e.stderr.strip()}]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def analyze_diff_for_dependencies(diff_content: str) -> List[str]:
    """Call Claude Code CLI to analyze git diff and suggest dependency classes for context."""

    prompt = f"""Given the git diff output below, analyze the changes and suggest what dependency classes should be added as additional context for code review.

Focus on:
- Classes that are directly used or modified in the diff
- Related service/repository/utility classes that would help understand the changes
- Domain models that are referenced
- Exclude: library classes (java.*, org.springframework.*, etc.) and classes in com/kerb/persistent/domain

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{"classes": ["com/example/Class1", "com/example/Class2"]}}

Git Diff:
```
{diff_content}
```
"""

    print("Calling Claude Code CLI to analyze diff and suggest dependencies...")

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=180000  # 3 minute timeout
        )

        response = json.loads(result.stdout)

        # Extract classes from response
        if "result" in response:
            result_text = response["result"]
            # Try to find JSON in the result
            json_match = re.search(r'\{.*"classes".*\}', result_text, re.DOTALL)
            if json_match:
                classes_data = json.loads(json_match.group())
                classes = classes_data.get("classes", [])
                print(f"✓ Found {len(classes)} dependency classes")
                return classes
            else:
                print("⚠ Could not parse classes from Claude response")
                return []
        else:
            print("⚠ Unexpected response format")
            return []

    except subprocess.TimeoutExpired:
        print("✗ Claude request timed out")
        return []
    except subprocess.CalledProcessError as e:
        print(f"✗ Error calling Claude CLI: {e.stderr}")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing Claude response: {e}")
        return []
    except FileNotFoundError:
        print("✗ 'claude' command not found. Please ensure Claude Code CLI is installed.")
        return []
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_full_diff(repo_path: str, source_ref: str, dest_ref: str) -> str:
    """Get the full diff between two branches."""
    try:
        result = subprocess.run(
            ['git', 'diff', f'{source_ref}...{dest_ref}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git diff: {e.stderr}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='AI-powered code review tool using git diff and Claude'
    )
    parser.add_argument('source_branch', help='Source branch name')
    parser.add_argument('destination_branch', help='Destination branch name')
    parser.add_argument('--repo-path', help='Path to repository (overrides REPO_PATH env var)', default=None)
    parser.add_argument('--output-dir', help='Output directory', default='output')

    args = parser.parse_args()

    try:
        # Get repository path
        if args.repo_path:
            repo_path = args.repo_path
        else:
            repo_path = get_repo_path()

        # Get script directory for output
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Repository path: {repo_path}")
        print(f"Analyzing changes: {args.source_branch} -> {args.destination_branch}")
        print(f"Output directory: {output_dir}\n")

        # Ensure branches exist
        print("Verifying branches...")
        source_ref = ensure_branch_exists(repo_path, args.source_branch)
        dest_ref = ensure_branch_exists(repo_path, args.destination_branch)
        print(f"✓ Source: {source_ref}")
        print(f"✓ Destination: {dest_ref}\n")

        # Step 1: Get full diff and save to file
        print("="*10)
        print("STEP 1: Getting git diff")
        print("="*10)

        full_diff = get_full_diff(repo_path, source_ref, dest_ref)

        # Save diff to file for debugging
        diff_file = os.path.join(output_dir, 'diff.txt')
        with open(diff_file, 'w', encoding='utf-8') as f:
            f.write(full_diff)

        print(f"✓ Diff retrieved ({len(full_diff)} characters)")
        print(f"✓ Saved to: {diff_file}\n")

        # Step 2: Analyze diff with Claude to get dependency classes
        print("="*10)
        print("STEP 2: Analyzing diff to identify dependency classes")
        print("="*10)

        classes = analyze_diff_for_dependencies(full_diff)

        if classes:
            print(f"\nDependency classes identified:")
            for idx, cls in enumerate(classes, 1):
                print(f"  {idx}. {cls}")
        else:
            print("⚠ No dependency classes identified")
        print()


        # Step 3: Checkout destination branch and pack dependencies with infiniloom
        print("="*10)
        print("STEP 3: Packing dependency classes with infiniloom")
        print("="*10)

        llm_content = ""
        output_file = None

        if classes:
            try:
                current_branch = get_current_branch(repo_path)
                # Checkout destination branch if not already on it
                if current_branch != dest_ref and current_branch != args.destination_branch:
                    checkout_branch(repo_path, dest_ref)
                else:
                    print("✓ Already on destination branch")

                # Execute infiniloom pack with the class list
                output_file = execute_infiniloom_pack_with_classes(repo_path, classes, output_dir)

                # Read llm.txt content
                if output_file and os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        llm_content = f.read()
                    print(f"✓ LLM content read ({len(llm_content)} characters)\n")
            except Exception as e:
                print(f"\n✗ Error packing dependencies: {str(e)}")
                import traceback
                traceback.print_exc()
                llm_content = "[Dependency context not available - infiniloom pack failed]"
                print()
        else:
            print("No dependency classes to pack\n")
            llm_content = "[No dependency classes identified for additional context]"

        # Step 4: Create final prompt and get code review from Claude
        print("="*10)
        print("STEP 4: Getting code review from Claude")
        print("="*10)

        # Build the final prompt
        final_prompt = f"""Given the Git Diff and Reference as additional context, please help review Git Diff and provide suggestions if needed.
----
## Git Diff:
Changes between {args.source_branch} and {args.destination_branch}:
```diff
{full_diff}
```
----
## Reference as Additional Context (Related Classes)
{llm_content}
---
Please provide a comprehensive code review focusing on:
- Code quality and best practices
- Potential bugs or issues
- Performance considerations
- Security concerns
- Suggestions for improvements
"""

        # Save prompt for debugging
        prompt_file = os.path.join(output_dir, 'prompt.txt')
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        print(f"✓ Prompt saved to: {prompt_file}")

        # Send to Claude for review
        review_file = None
        review_text = ""

        try:
            print("Sending prompt to Claude Code CLI...")
            result = subprocess.run(
                ["claude", "-p", final_prompt, "--output-format", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=300000  # 5 minute timeout
            )

            print("✓ Claude review complete")

            # Parse the response
            claude_response = json.loads(result.stdout)

            # Extract the review from the response
            if "result" in claude_response:
                review_text = claude_response["result"]
            else:
                review_text = json.dumps(claude_response, indent=2)

            # Save review to file
            review_file = os.path.join(output_dir, 'code_review.md')
            with open(review_file, 'w', encoding='utf-8') as f:
                f.write(review_text)
            print(f"✓ Review saved to: {review_file}")

            # Print review to console
            print("\n" + "="*80)
            print("CODE REVIEW")
            print("="*80)
            print(review_text)
            print("="*80)

        except subprocess.TimeoutExpired:
            print("✗ Claude request timed out after 5 minutes")
            review_text = "[Timeout - no response received]"
        except subprocess.CalledProcessError as e:
            print(f"✗ Error calling Claude CLI: {e.stderr}")
            review_text = f"[Error: {e.stderr}]"
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing Claude response: {e}")
            review_text = "[Error parsing response]"
        except FileNotFoundError:
            print("✗ 'claude' command not found. Please ensure Claude Code CLI is installed.")
            review_text = "[Claude CLI not found]"
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            review_text = f"[Error: {str(e)}]"

        # Final summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✓ Diff file: {diff_file}")
        print(f"✓ Dependencies identified: {len(classes)}")
        if output_file and os.path.exists(output_file):
            print(f"✓ Dependencies packed: {output_file}")
        print(f"✓ Prompt file: {prompt_file}")
        if review_file:
            print(f"✓ Code review: {review_file}")
        else:
            print(f"✗ Code review: Failed")
        print("="*80)

        # Return data structure
        return {
            'diff_file': diff_file,
            'classes': classes,
            'output_file': output_file,
            'prompt_file': prompt_file,
            'review_file': review_file,
            'review': review_text
        }

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
