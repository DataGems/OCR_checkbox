#!/bin/bash
# Main run script for OCR Checkbox Detection Pipeline
# Usage: ./run.sh <command> [arguments]

set -e  # Exit on any error

PROJECT_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
cd \"$PROJECT_ROOT\"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Helper functions
log_info() {
    echo -e \"${BLUE}[INFO]${NC} $1\"
}

log_success() {
    echo -e \"${GREEN}[SUCCESS]${NC} $1\"
}

log_warning() {
    echo -e \"${YELLOW}[WARNING]${NC} $1\"
}

log_error() {
    echo -e \"${RED}[ERROR]${NC} $1\"
}

# Check if virtual environment is activated
check_venv() {
    if [[ \"$VIRTUAL_ENV\" != *\".venv\"* ]]; then
        log_warning \"Virtual environment not detected. Activating...\"
        if [ -f \".venv/bin/activate\" ]; then
            source .venv/bin/activate
            log_success \"Virtual environment activated\"
        else
            log_error \"Virtual environment not found. Please run 'uv sync' first.\"
            exit 1
        fi
    fi
}

# Setup function
setup() {
    log_info \"Setting up OCR Checkbox Detection Pipeline...\"
    
    # Install dependencies
    log_info \"Installing dependencies with uv...\"
    uv sync
    
    # Activate environment
    source .venv/bin/activate
    
    # Run installation test
    log_info \"Running installation test...\"
    python scripts/test_installation.py
    
    log_success \"Setup completed successfully!\"
    log_info \"Next steps:\"
    echo \"  1. Download sample data: ./run.sh download-sample\"
    echo \"  2. Run demo: ./run.sh demo\"
    echo \"  3. Process a PDF: ./run.sh infer your_document.pdf\"
}

# Download sample data
download_sample() {
    check_venv
    log_info \"Downloading sample data...\"
    python scripts/download_data.py --sample
    log_success \"Sample data downloaded\"
}

# Download all datasets
download_all() {
    check_venv
    log_info \"Downloading all datasets...\"
    python scripts/download_data.py --all
    log_success \"All datasets downloaded\"
}

# Run demo
demo() {
    check_venv
    log_info \"Running interactive demo...\"
    python scripts/demo.py
}

# Run installation test
test() {
    check_venv
    log_info \"Running installation test...\"
    python scripts/test_installation.py
}

# Run inference
infer() {
    check_venv
    if [ -z \"$1\" ]; then
        log_error \"Usage: ./run.sh infer <pdf_path> [additional_args]\"
        exit 1
    fi
    
    log_info \"Running inference on: $1\"
    shift  # Remove first argument (pdf_path)
    python scripts/infer.py \"$@\"
}

# Clean generated files
clean() {
    log_info \"Cleaning generated files...\"
    
    # Remove Python cache
    find . -name \"__pycache__\" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name \"*.pyc\" -delete 2>/dev/null || true
    find . -name \"*.pyo\" -delete 2>/dev/null || true
    
    # Remove generated data (but keep raw datasets)
    rm -rf data/sample/test_form.pdf* 2>/dev/null || true
    rm -rf data/sample/*_results.json* 2>/dev/null || true
    rm -rf data/sample/*_crops/ 2>/dev/null || true
    
    # Remove log files
    rm -rf *.log 2>/dev/null || true
    rm -rf logs/ 2>/dev/null || true
    
    # Remove temporary model files
    rm -rf runs/ 2>/dev/null || true
    
    log_success \"Cleanup completed\"
}

# Development setup
dev_setup() {
    log_info \"Setting up development environment...\"
    
    # Install with dev dependencies
    uv sync --group dev
    
    # Activate environment
    source .venv/bin/activate
    
    # Install pre-commit hooks if available
    if command -v pre-commit &> /dev/null; then
        log_info \"Installing pre-commit hooks...\"
        pre-commit install
    fi
    
    log_success \"Development environment setup completed\"
}

# Format code
format() {
    check_venv
    log_info \"Formatting code with black...\"
    black src/ scripts/
    log_success \"Code formatting completed\"
}

# Run linting
lint() {
    check_venv
    log_info \"Running linting with flake8...\"
    flake8 src/ scripts/
    log_success \"Linting completed\"
}

# Show help
show_help() {
    echo \"OCR Checkbox Detection Pipeline - Run Script\"
    echo \"\"
    echo \"Usage: ./run.sh <command> [arguments]\"
    echo \"\"
    echo \"Setup Commands:\"
    echo \"  setup              - Initial project setup (install deps, run tests)\"
    echo \"  dev-setup          - Setup development environment with dev dependencies\"
    echo \"\"
    echo \"Data Commands:\"
    echo \"  download-sample    - Download sample test data\"
    echo \"  download-all       - Download all datasets (CheckboxQA, FUNSD, etc.)\"
    echo \"\"
    echo \"Processing Commands:\"
    echo \"  demo               - Run interactive demo\"
    echo \"  infer <pdf_path>   - Run inference on a PDF file\"
    echo \"  test               - Run installation test\"
    echo \"\"
    echo \"Development Commands:\"
    echo \"  format             - Format code with black\"
    echo \"  lint               - Run linting with flake8\"
    echo \"  clean              - Clean generated files and cache\"
    echo \"\"
    echo \"Examples:\"
    echo \"  ./run.sh setup                              # Initial setup\"
    echo \"  ./run.sh download-sample                    # Get sample data\"
    echo \"  ./run.sh demo                               # Run demo\"
    echo \"  ./run.sh infer document.pdf                 # Process PDF\"
    echo \"  ./run.sh infer doc.pdf --save-visualizations # Process with visualization\"
    echo \"\"
    echo \"For more detailed usage, see README.md\"
}

# Main command dispatcher
case \"$1\" in
    \"setup\")
        setup
        ;;
    \"dev-setup\")
        dev_setup
        ;;
    \"download-sample\")
        download_sample
        ;;
    \"download-all\")
        download_all
        ;;
    \"demo\")
        demo
        ;;
    \"test\")
        test
        ;;
    \"infer\")
        shift  # Remove 'infer' from arguments
        infer \"$@\"
        ;;
    \"format\")
        format
        ;;
    \"lint\")
        lint
        ;;
    \"clean\")
        clean
        ;;
    \"help\" | \"-h\" | \"--help\" | \"\")
        show_help
        ;;
    *)
        log_error \"Unknown command: $1\"
        echo \"\"
        show_help
        exit 1
        ;;
esac
