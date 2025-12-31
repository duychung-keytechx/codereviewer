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


def filter_java_files(files: List[str]) -> List[str]:
    """Filter only Java files from the list."""
    java_files = [f for f in files if f.endswith('.java')]
    return java_files


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
           '--remove-comments', '--max-token', 16000, '--output',
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


def analyze_with_claude(java_files: List[str], file_contents: Dict[str, str]) -> Dict:
    """Call Claude Code CLI to analyze changed files and identify used Java classes."""

    # Build the prompt with all file contents
    prompt_parts = [
        "Analyze the following Java files and identify all Java classes from project (not library java classes) that are used/referenced in these files.",
        "Return ONLY a JSON object with this structure:",
        '{"classes": ["com/example/ClassName1", "com/example/ClassName2", ...]}',
    ]

    for file_path, content in file_contents.items():
        prompt_parts.append(f"=== File: {file_path} ===")
        prompt_parts.append(content)
        prompt_parts.append("")

    full_prompt = "\n".join(prompt_parts)
    print("\nCalling Claude Code CLI to analyze files...")

    try:
        # Call claude CLI with the prompt
        result = subprocess.run(
            ["claude", "-p", full_prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the JSON response
        response = json.loads(result.stdout)

        print("✓ Analysis complete")

        return response

    except subprocess.CalledProcessError as e:
        print(f"Error calling Claude CLI: {e.stderr}")
        return {"error": "Failed to call Claude CLI", "details": e.stderr}
    except json.JSONDecodeError as e:
        print(f"Error parsing Claude response: {e}")
        return {"error": "Failed to parse Claude response", "details": str(e)}
    except FileNotFoundError:
        print("Error: 'claude' command not found. Please ensure Claude Code CLI is installed.")
        return {"error": "Claude CLI not found"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": "Unexpected error", "details": str(e)}


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


def get_changed_files(repo_path: str, source_branch: str, destination_branch: str) -> List[str]:
    """Get list of changed files between two branches using git diff (excluding new files)."""
    try:
        # Ensure both branches exist, fetching if necessary
        print(f"Checking source branch: {source_branch}")
        source_ref = ensure_branch_exists(repo_path, source_branch)

        print(f"Checking destination branch: {destination_branch}")
        dest_ref = ensure_branch_exists(repo_path, destination_branch)

        # Get the list of changed files (exclude new/added files with --diff-filter)
        # M = modified, D = deleted, R = renamed, C = copied (excludes A = added)
        print(f"\nComparing {source_ref} ... {dest_ref}")
        result = subprocess.run(
            ['git', 'diff', '--name-only', '--diff-filter=d', f'{source_ref}...{dest_ref}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        # Filter out empty lines
        changed_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        print(f"Found {len(changed_files)} changed files (excluding new files)")

        return changed_files
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute git diff: {e.stderr}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze Java code changes between branches (excluding new files)'
    )
    parser.add_argument('source_branch', help='Source branch name')
    parser.add_argument('destination_branch', help='Destination branch name')
    parser.add_argument('--repo-path', help='Path to repository (overrides REPO_PATH env var)', default=None)
    parser.add_argument('--output-dir', help='Output directory for llm.txt', default='output')

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

        print(f"Repository path: {repo_path}")
        print(f"Analyzing changes: {args.source_branch} -> {args.destination_branch}\n")

        # Get destination branch ref for reading files
        dest_ref = ensure_branch_exists(repo_path, args.destination_branch)

        # Step 1: Get changed files (excluding new files)
        print("Step 1: Getting changed files...")
        changed_files = get_changed_files(repo_path, args.source_branch, args.destination_branch)

        # Step 2: Filter Java files only
        print("\nStep 2: Filtering Java files...")
        java_files = filter_java_files(changed_files)

        if not java_files:
            print("No Java files found in the changes.")
            return {
                'changed_files': changed_files,
                'java_files': []
            }

        print(f"Found {len(java_files)} Java file(s):")
        for idx, file in enumerate(java_files, 1):
            print(f"  {idx}. {file}")

        # Output results
        print("RESULTS")
        print("="*5)
        print(f"\nTotal changed files (excluding new): {len(changed_files)}")
        print(f"Java files: {len(java_files)}")

        # Step 3: Read file contents and display
        print("\nStep 3: Reading file contents...")
        print("="*5)
        print("FILE CONTENTS")
        print("="*5)

        file_contents = {}
        for idx, file in enumerate(java_files, 1):
            print(f"\n{'─'*5}")
            print(f"[{idx}/{len(java_files)}] File: {file}")
            print(f"{'─'*5}")
            content = read_file_content_from_git(repo_path, dest_ref, file)
            file_contents[file] = content
            print(content)

        # Step 4: Analyze with Claude to find used classes
        print("\n" + "="*5)
        print("Step 4: Analyzing with Claude Code CLI...")
        print("="*5)

        analysis_result = analyze_with_claude(java_files, file_contents)

        # Display Claude's analysis
        print("\n" + "="*5)
        print("CLAUDE ANALYSIS RESULTS")
        print("="*5)

        classes = []
        if "error" in analysis_result:
            print(f"\nError: {analysis_result.get('error')}")
            if "details" in analysis_result:
                print(f"Details: {analysis_result.get('details')}")
        else:
            # Extract the result from Claude's response
            if "result" in analysis_result:
                # Parse the result which might contain JSON
                result_text = analysis_result["result"]
                try:
                    # Try to find and parse JSON in the result
                    json_match = re.search(r'\{.*"classes".*\}', result_text, re.DOTALL)
                    if json_match:
                        classes_data = json.loads(json_match.group())
                        classes = classes_data.get("classes", [])
                    else:
                        print("Warning: Could not find classes list in Claude's response")
                except:
                    print("Warning: Could not parse Claude's response")

                print(f"\nFound {len(classes)} Java classes used in the changed files:")
                for idx, class_name in enumerate(classes, 1):
                    print(f"  {idx}. {class_name}")

                # Filter out excluded patterns
                excluded_pattern = "com/kerb/persistent/domain"
                filtered_classes = [c for c in classes if excluded_pattern not in c]

                if len(filtered_classes) < len(classes):
                    excluded_count = len(classes) - len(filtered_classes)
                    print(f"\nExcluded {excluded_count} class(es) matching pattern '{excluded_pattern}'")
                    print(f"Remaining classes: {len(filtered_classes)}")

                classes = filtered_classes
            else:
                print("\nClaude's full response:")
                print(json.dumps(analysis_result, indent=2))

        # Step 5: Checkout destination branch and run infiniloom pack
        output_file = None
        if classes:
            print("\n" + "="*5)
            print("Step 5: Preparing to pack classes with infiniloom...")
            print("="*5)

            try:
                # Check current branch
                current_branch = get_current_branch(repo_path)
                print(f"\nCurrent branch: {current_branch}")
                print(f"Destination branch: {dest_ref}")

                # Checkout destination branch if not already on it
                if current_branch != dest_ref and current_branch != args.destination_branch:
                    checkout_branch(repo_path, dest_ref)
                else:
                    print("✓ Already on destination branch")

                # Execute infiniloom pack with the class list
                output_file = execute_infiniloom_pack_with_classes(repo_path, classes, output_dir)
            except Exception as e:
                print(f"\n✗ Error in Step 5: {str(e)}")
                import traceback
                traceback.print_exc()
                print("\nContinuing to Step 6 anyway...")
                # Set default output file path even if pack failed
                output_file = os.path.join(output_dir, 'llm.txt')

        # Step 6: Create final prompt with git diff and llm.txt content
        # This always runs if we got to analysis
        if True:  # Always execute Step 6
            print("\n" + "="*5)
            print("Step 6: Creating final prompt...")
            print("="*5)

            try:
                # Get source ref
                print("Getting source branch reference...")
                source_ref = ensure_branch_exists(repo_path, args.source_branch)
                print(f"✓ Source ref: {source_ref}")

                # Get full diff between branches
                print("\nGetting git diff between branches...")
                full_diff = get_full_diff(repo_path, source_ref, dest_ref)
                print(f"✓ Diff retrieved ({len(full_diff)} characters)")

                # Read llm.txt content if it exists
                print("\nReading llm.txt content...")
                if output_file and os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        llm_content = f.read()
                    print(f"✓ LLM content read ({len(llm_content)} characters)")
                else:
                    llm_content = "[LLM content not available - infiniloom pack did not complete successfully]"
                    print("⚠ LLM content not available")
            except Exception as e:
                print(f"\n✗ Error in Step 6: {str(e)}")
                import traceback
                traceback.print_exc()
                llm_content = f"[Error: {str(e)}]"
                full_diff = "[Error getting diff]"

            # Build the final prompt
            final_prompt = f"""# Code Review Request

## Git Diff
Below is the diff between {args.source_branch} and {args.destination_branch}:

```diff
{full_diff}
```

## Related Classes and Dependencies
The following classes are related to the changes (packed by infiniloom):

{llm_content}

---
Please analyze these changes and provide a comprehensive code review.
"""

            # Print the prompt for debugging
            print("\n" + "="*80)
            print("FINAL PROMPT (for debugging)")
            print("="*80)
            print(final_prompt)
            print("="*80)

            # Save prompt to file for reference
            prompt_file = os.path.join(output_dir, 'prompt.txt')
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(final_prompt)
            print(f"\n✓ Prompt saved to: {prompt_file}")

            # Step 7: Send prompt to Claude for code review
            print("\n" + "="*5)
            print("Step 7: Requesting Claude code review...")
            print("="*5)

            review_file = None
            review_text = ""

            try:
                print("\nSending prompt to Claude Code CLI...")
                result = subprocess.run(
                    ["claude", "-p", final_prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300000  # 5 minute timeout
                )

                print("✓ Claude analysis complete")

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
                print("CLAUDE CODE REVIEW")
                print("="*80)
                print(review_text)
                print("="*80)

            except subprocess.TimeoutExpired:
                print("✗ Claude request timed out after 5 minutes")
                review_file = None
                review_text = "[Timeout - no response received]"
            except subprocess.CalledProcessError as e:
                print(f"✗ Error calling Claude CLI: {e.stderr}")
                review_file = None
                review_text = f"[Error: {e.stderr}]"
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing Claude response: {e}")
                print(f"Raw output: {result.stdout[:500]}...")
                review_file = None
                review_text = "[Error parsing response]"
            except FileNotFoundError:
                print("✗ 'claude' command not found. Please ensure Claude Code CLI is installed.")
                review_file = None
                review_text = "[Claude CLI not found]"
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                review_file = None
                review_text = f"[Error: {str(e)}]"

            print("\n" + "="*5)
            print("FINAL RESULTS")
            print("="*5)
            print(f"\n✓ Total changed files: {len(changed_files)}")
            print(f"✓ Java files analyzed: {len(java_files)}")
            print(f"✓ Classes identified: {len(classes)}")
            if output_file and os.path.exists(output_file):
                print(f"✓ Output file: {output_file}")
            else:
                print(f"✗ Output file: Not created (infiniloom pack failed)")
            print(f"✓ Prompt file: {prompt_file}")
            if review_file:
                print(f"✓ Code review file: {review_file}")
            else:
                print(f"✗ Code review file: Not created (Claude review failed)")

            # Return data structure
            return {
                'changed_files': changed_files,
                'java_files': java_files,
                'file_contents': file_contents,
                'analysis': analysis_result,
                'classes': classes,
                'output_file': output_file,
                'prompt': final_prompt,
                'prompt_file': prompt_file,
                'review': review_text,
                'review_file': review_file
            }

        # Fallback if no classes found
        print("\nNo classes found to pack. Skipping Steps 5 and 6.")
        return {
            'changed_files': changed_files,
            'java_files': java_files,
            'file_contents': file_contents,
            'analysis': analysis_result,
            'classes': []
        }

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
