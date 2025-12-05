#!/bin/bash
# file: reset_data_structure.sh
# Description: Remove existing data directory and create new organized structure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}     DATA DIRECTORY RESET AND RESTRUCTURE      ${NC}"
echo -e "${YELLOW}================================================${NC}"
echo

# Warning message
echo -e "${RED}⚠️  WARNING: This will DELETE the entire 'data' directory!${NC}"
echo -e "${RED}   All existing data will be permanently removed.${NC}"
echo
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

echo
echo -e "${YELLOW}Removing existing data directory...${NC}"

# Remove existing data directory if it exists
if [ -d "data" ]; then
    rm -rf data
    echo -e "${GREEN}✓ Existing data directory removed${NC}"
else
    echo -e "${GREEN}✓ No existing data directory found${NC}"
fi

echo
echo -e "${YELLOW}Creating new directory structure...${NC}"

# Create main directories
mkdir -p data/raw
mkdir -p data/processed

# Define gesture classes
gestures=("zero" "one" "two" "three" "four" "five")

# Define split types
splits=("train" "val" "test")

# Create directory structure for raw data
echo -e "${GREEN}Creating raw data structure...${NC}"
for split in "${splits[@]}"; do
    for gesture in "${gestures[@]}"; do
        mkdir -p "data/raw/$split/$gesture"
        echo -e "  ✓ Created: data/raw/$split/$gesture"
    done
done

echo

# Create directory structure for processed data
echo -e "${GREEN}Creating processed data structure...${NC}"
for split in "${splits[@]}"; do
    for gesture in "${gestures[@]}"; do
        mkdir -p "data/processed/$split/$gesture"
        echo -e "  ✓ Created: data/processed/$split/$gesture"
    done
done

echo
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ Directory structure created successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo

# Display the structure
echo -e "${YELLOW}New directory structure:${NC}"
echo "data/"
echo "├── raw/"
echo "│   ├── train/"
echo "│   │   ├── zero/"
echo "│   │   ├── one/"
echo "│   │   ├── two/"
echo "│   │   ├── three/"
echo "│   │   ├── four/"
echo "│   │   └── five/"
echo "│   ├── val/"
echo "│   │   ├── zero/"
echo "│   │   ├── one/"
echo "│   │   ├── two/"
echo "│   │   ├── three/"
echo "│   │   ├── four/"
echo "│   │   └── five/"
echo "│   └── test/"
echo "│       ├── zero/"
echo "│       ├── one/"
echo "│       ├── two/"
echo "│       ├── three/"
echo "│       ├── four/"
echo "│       └── five/"
echo "└── processed/"
echo "    ├── train/"
echo "    │   ├── zero/"
echo "    │   ├── one/"
echo "    │   ├── two/"
echo "    │   ├── three/"
echo "    │   ├── four/"
echo "    │   └── five/"
echo "    ├── val/"
echo "    │   ├── zero/"
echo "    │   ├── one/"
echo "    │   ├── two/"
echo "    │   ├── three/"
echo "    │   ├── four/"
echo "    │   └── five/"
echo "    └── test/"
echo "        ├── zero/"
echo "        ├── one/"
echo "        ├── two/"
echo "        ├── three/"
echo "        ├── four/"
echo "        └── five/"

echo
echo -e "${GREEN}Total directories created: 72${NC}"
echo -e "${YELLOW}Ready for data collection!${NC}"