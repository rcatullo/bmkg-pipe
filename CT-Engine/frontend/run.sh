#!/bin/bash
# Run the ReACT Engine Frontend Server

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
HOST="0.0.0.0"
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting ReACT Engine Frontend..."
echo "Server will be available at: http://$HOST:$PORT"
echo ""
echo "To access from your local machine:"
echo "  1. Run on your LOCAL machine: ssh -L $PORT:localhost:$PORT user@this-server"
echo "  2. Open in browser: http://localhost:$PORT"
echo ""

python server.py --host "$HOST" --port "$PORT" $RELOAD




