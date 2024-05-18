#!/bin/bash

pkill -f "envision start"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting script...${NC}"
echo

case $1 in
1)
  echo -e "${CYAN}Starting envision and training on sumo/free scenarios...${NC}"
  scl envision start &
  python train.py sumo/free
  ;;
2)
  echo -e "${CYAN}Starting envision and training on sumo/congested scenarios...${NC}"
  scl envision start &
  python train.py sumo/congested
  ;;
3)
  echo -e "${CYAN}Starting envision and training on sumo/normal scenarios...${NC}"
  scl envision start &
  python train.py sumo/normal
  ;;
4)
  echo -e "${CYAN}Starting envision and training on all scenarios...${NC}"
  scl envision start &
  python train.py sumo/loop sumo/free sumo/normal sumo/congested
  ;;
*)
  echo -e "${RED}Invalid argument. Please provide a number between 1 and 4.${NC}"
  ;;
esac

echo
echo -e "${GREEN}Script execution completed${NC}"
