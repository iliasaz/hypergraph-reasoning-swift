#!/bin/bash
#
# GraphRAG Pipeline Script
#
# Runs the full hypergraph pipeline: markdown → extraction → embeddings → query
#
# Usage:
#   ./run_pipeline.sh <markdown_file> "<question>"
#
# Environment:
#   OPENROUTER_API_KEY - Required: Your OpenRouter API key
#   CHAT_MODEL         - Optional: Model for chat (default: meta-llama/llama-4-maverick)
#
# Example:
#   export OPENROUTER_API_KEY="sk-or-..."
#   ./run_pipeline.sh ./docs/paper.md "What are the main findings?"
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
EMBEDDING_MODEL="nomic-embed-text:v1.5"
CHUNK_SIZE=10000
TOP_K=5
MAX_PATH_LENGTH=4
SIMPLIFY_THRESHOLD=0.9

# Parse arguments
MARKDOWN_FILE="$1"
QUESTION="$2"

# Validate inputs
if [[ -z "$MARKDOWN_FILE" ]]; then
    echo -e "${RED}Error: Missing markdown file argument${NC}"
    echo ""
    echo "Usage: $0 <markdown_file> [question]"
    echo ""
    echo "Examples:"
    echo "  $0 ./paper.md \"What is the main contribution?\""
    echo "  $0 ./docs/  # Process entire directory"
    exit 1
fi

if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY environment variable not set${NC}"
    echo ""
    echo "Set it with: export OPENROUTER_API_KEY='your-api-key'"
    exit 1
fi

if [[ ! -e "$MARKDOWN_FILE" ]]; then
    echo -e "${RED}Error: File or directory not found: $MARKDOWN_FILE${NC}"
    exit 1
fi

# Determine output directory based on input
INPUT_BASENAME=$(basename "$MARKDOWN_FILE" .md)
OUTPUT_DIR="./output/${INPUT_BASENAME}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GraphRAG Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Input:      ${GREEN}$MARKDOWN_FILE${NC}"
echo -e "Output:     ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Model:      ${GREEN}$CHAT_MODEL${NC}"
echo ""

# Step 1: Process markdown → hypergraph + embeddings + metadata
echo -e "${YELLOW}[Step 1/3] Processing markdown...${NC}"
swift run hypergraph-cli process "$MARKDOWN_FILE" \
    --output "$OUTPUT_DIR" \
    --provider openrouter \
    --api-key "$OPENROUTER_API_KEY" \
    --chat-model "$CHAT_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --chunk-size "$CHUNK_SIZE" \
    --verbose

# Find the generated files (pattern: <basename>_graph.json, etc.)
GRAPH_FILE=$(find "$OUTPUT_DIR" -name "*_graph.json" | head -1)
EMBEDDINGS_FILE=$(find "$OUTPUT_DIR" -name "*_embeddings.json" | head -1)
METADATA_FILE=$(find "$OUTPUT_DIR" -name "*_metadata.json" | head -1)
CHUNKS_FILE=$(find "$OUTPUT_DIR" -name "*_chunks.json" | head -1)

if [[ -z "$GRAPH_FILE" ]] || [[ -z "$EMBEDDINGS_FILE" ]]; then
    echo -e "${RED}Error: Could not find generated graph or embeddings files${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Generated files:${NC}"
echo -e "  Graph:      $GRAPH_FILE"
echo -e "  Embeddings: $EMBEDDINGS_FILE"
[[ -n "$METADATA_FILE" ]] && echo -e "  Metadata:   $METADATA_FILE"
[[ -n "$CHUNKS_FILE" ]] && echo -e "  Chunks:     $CHUNKS_FILE"
echo ""

# Step 2: Simplify (optional - uncomment to enable)
# echo -e "${YELLOW}[Step 2/3] Simplifying hypergraph...${NC}"
# SIMPLIFIED_DIR="$OUTPUT_DIR/simplified"
# swift run hypergraph-cli simplify "$GRAPH_FILE" \
#     --embeddings "$EMBEDDINGS_FILE" \
#     --output "$SIMPLIFIED_DIR" \
#     --threshold "$SIMPLIFY_THRESHOLD" \
#     --recompute-embeddings \
#     --verbose
#
# GRAPH_FILE="$SIMPLIFIED_DIR/simplified_graph.json"
# EMBEDDINGS_FILE="$SIMPLIFIED_DIR/simplified_embeddings.json"

echo -e "${YELLOW}[Step 2/3] Hypergraph info:${NC}"
swift run hypergraph-cli info "$GRAPH_FILE"
echo ""

# Step 3: Query (if question provided)
if [[ -n "$QUESTION" ]]; then
    echo -e "${YELLOW}[Step 3/3] Querying...${NC}"
    echo -e "Question: ${GREEN}$QUESTION${NC}"
    echo ""

    # Build query command with optional metadata
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
    if [[ -n "$METADATA_FILE" ]]; then
        QUERY_CMD+=(--metadata "$METADATA_FILE")
    fi

    # Add chunks if available (for citations)
    if [[ -n "$CHUNKS_FILE" ]]; then
        QUERY_CMD+=(--chunks "$CHUNKS_FILE")
    fi

    "${QUERY_CMD[@]}"
else
    echo -e "${YELLOW}[Step 3/3] Skipped (no question provided)${NC}"
    echo ""
    echo -e "To query the graph, run:"
    echo -e "${GREEN}swift run hypergraph-cli query \"Your question here\" \\${NC}"
    echo -e "${GREEN}    --graph \"$GRAPH_FILE\" \\${NC}"
    echo -e "${GREEN}    --embeddings \"$EMBEDDINGS_FILE\" \\${NC}"
    [[ -n "$METADATA_FILE" ]] && echo -e "${GREEN}    --metadata \"$METADATA_FILE\" \\${NC}"
    [[ -n "$CHUNKS_FILE" ]] && echo -e "${GREEN}    --chunks \"$CHUNKS_FILE\" \\${NC}"
    echo -e "${GREEN}    --provider openrouter \\${NC}"
    echo -e "${GREEN}    --api-key \"\$OPENROUTER_API_KEY\" \\${NC}"
    echo -e "${GREEN}    --show-context${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Pipeline complete!${NC}"
echo -e "${BLUE}========================================${NC}"
