# Contributing to IoT Malicious Traffic Detection

First off, thank you for considering contributing to this project! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include details about your configuration and environment

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python style guides
* Include thoughtfully-worded, well-structured tests
* Document new code based on the Documentation Styleguide
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\\venv\\Scripts\\activate  # Windows
   pip install -r requirements-dev.txt
   ```

3. Make your changes:
   * Write meaningful commit messages
   * Follow the style guides
   * Write/update tests as needed
   * Update documentation as needed

4. Ensure the test suite passes:
   ```bash
   pytest tests/
   ```

5. Run code style checks:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   isort src/ tests/
   ```

## Style Guides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use [Black](https://github.com/psf/black) for code formatting
* Use [isort](https://pycqa.github.io/isort/) for import sorting
* Use type hints where possible
* Write docstrings for all public methods and classes

### Documentation Style Guide

* Use [Google style](https://google.github.io/styleguide/pyguide.html) for docstrings
* Include examples in docstrings when possible
* Keep line length to 80 characters or less
* Use [Markdown](https://guides.github.com/features/mastering-markdown/) for documentation files

## Project Structure

```
project_root/
├── src/                    # Source code
│   ├── models/            # ML model implementation
│   ├── data/              # Data processing
│   ├── detection/         # Detection system
│   └── utils/             # Utilities
├── tests/                 # Test files
├── docs/                  # Documentation
└── configs/               # Configuration files
```

## Testing

* Write unit tests for all new code
* Ensure tests are deterministic
* Mock external services and APIs
* Include both positive and negative test cases
* Test edge cases and error conditions

## Documentation

* Update README.md with any new features
* Document all new functions and classes
* Update API documentation when endpoints change
* Include examples for new functionality
* Keep the Wiki up to date

## Questions?

Feel free to contact the maintainers if you have any questions:

* Email: renxuyi@grinnell.edu
<!-- * Discord: [Join our server](https://discord.gg/your-invite-link)
* GitHub Discussions: Use our [discussions board](https://github.com/yourusername/project/discussions) -->

Thank you for your contributions! 🎉 