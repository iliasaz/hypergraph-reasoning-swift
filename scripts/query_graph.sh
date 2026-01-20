#!/bin/bash
#
# GraphRAG Query Script
#
# Runs inference on an existing hypergraph to answer questions.
#
# Usage:
#   ./query_graph.sh <graph_directory> "<question>"
#
# Environment:
#   OPENROUTER_API_KEY - Required: Your OpenRouter API key
#   CHAT_MODEL         - Optional: Model for chat (default: meta-llama/llama-4-maverick)
#
# Examples:
#   export OPENROUTER_API_KEY="sk-or-..."
#   ./query_graph.sh ./output/paper "What are the main findings?"
#
#   # Use a different model
#   CHAT_MODEL="openai/gpt-4.1-mini" ./query_graph.sh ./output/paper "Summarize the key concepts"
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHAT_MODEL="${CHAT_MODEL:-meta-llama/llama-4-maverick}"
TOP_K=5
MAX_PATH_LENGTH=4

# Parse arguments
GRAPH_DIR="$1"
QUESTION="$2"

# Validate inputs
if [[ -z "$GRAPH_DIR" ]] || [[ -z "$QUESTION" ]]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <graph_directory> \"<question>\""
    echo ""
    echo "Examples:"
    echo "  $0 ./output/paper \"What are the main findings?\""
    echo "  $0 ./output/docs \"How does authentication work?\""
    echo ""
    echo "Environment Variables:"
    echo "  OPENROUTER_API_KEY  - Required: Your OpenRouter API key"
    echo "  CHAT_MODEL          - Optional: Model for chat (default: meta-llama/llama-4-maverick)"
    exit 1
fi

if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY environment variable not set${NC}"
    echo ""
    echo "Set it with: export OPENROUTER_API_KEY='your-api-key'"
    exit 1
fi

if [[ ! -d "$GRAPH_DIR" ]]; then
    echo -e "${RED}Error: Graph directory not found: $GRAPH_DIR${NC}"
    exit 1
fi

# Find graph files - prefer simplified if available
SIMPLIFIED_DIR="$GRAPH_DIR/simplified"
if [[ -d "$SIMPLIFIED_DIR" ]]; then
    GRAPH_FILE="$SIMPLIFIED_DIR/simplified_graph.json"
    EMBEDDINGS_FILE="$SIMPLIFIED_DIR/simplified_embeddings.json"
    GRAPH_TYPE="simplified"
else
    # Fall back to raw graph files
    GRAPH_FILE=$(find "$GRAPH_DIR" -maxdepth 1 -name "*_graph.json" | head -1)
    EMBEDDINGS_FILE=$(find "$GRAPH_DIR" -maxdepth 1 -name "*_embeddings.json" | head -1)
    GRAPH_TYPE="raw"
fi

# Find optional metadata and chunks files
METADATA_FILE=$(find "$GRAPH_DIR" -maxdepth 1 -name "*_metadata.json" 2>/dev/null | head -1)
CHUNKS_FILE=$(find "$GRAPH_DIR" -maxdepth 1 -name "*_chunks.json" 2>/dev/null | head -1)

# Validate required files exist
if [[ ! -f "$GRAPH_FILE" ]]; then
    echo -e "${RED}Error: Graph file not found in $GRAPH_DIR${NC}"
    echo "Expected: *_graph.json or simplified/simplified_graph.json"
    exit 1
fi

if [[ ! -f "$EMBEDDINGS_FILE" ]]; then
    echo -e "${RED}Error: Embeddings file not found in $GRAPH_DIR${NC}"
    echo "Expected: *_embeddings.json or simplified/simplified_embeddings.json"
    exit 1
fi

# Display configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GraphRAG Query${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Graph:      ${GREEN}$GRAPH_FILE${NC} (${GRAPH_TYPE})"
echo -e "Embeddings: ${GREEN}$EMBEDDINGS_FILE${NC}"
[[ -n "$METADATA_FILE" ]] && echo -e "Metadata:   ${GREEN}$METADATA_FILE${NC}"
[[ -n "$CHUNKS_FILE" ]] && echo -e "Chunks:     ${GREEN}$CHUNKS_FILE${NC}"
echo -e "Model:      ${GREEN}$CHAT_MODEL${NC}"
echo -e "Top-K:      ${GREEN}$TOP_K${NC}"
echo ""
echo -e "Question:   ${YELLOW}$QUESTION${NC}"
echo ""
echo -e "${BLUE}----------------------------------------${NC}"
echo ""

# Build query command
QUERY_CMD=(
    swift run hypergraph-cli query "$QUESTION"
    --graph "$GRAPH_FILE"
    --embeddings "$EMBEDDINGS_FILE"
    --provider openrouter
    --api-key "$OPENROUTER_API_KEY"
    --chat-model "$CHAT_MODEL"
    --top-k "$TOP_K"
    --max-path-length "$MAX_PATH_LENGTH"
    --show-context
    --verbose
)

# Add metadata if available
if [[ -n "$METADATA_FILE" ]] && [[ -f "$METADATA_FILE" ]]; then
    QUERY_CMD+=(--metadata "$METADATA_FILE")
fi

# Add chunks if available (for citations)
if [[ -n "$CHUNKS_FILE" ]] && [[ -f "$CHUNKS_FILE" ]]; then
    QUERY_CMD+=(--chunks "$CHUNKS_FILE")
fi

# Execute query
"${QUERY_CMD[@]}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Query complete!${NC}"
echo -e "${BLUE}========================================${NC}"
