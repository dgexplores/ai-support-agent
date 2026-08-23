#!/bin/bash
# Double-click this file to launch the server
cd "$(dirname "$0")"
echo "Starting Aster & Row Support Agent..."
echo ""
python3 -W ignore -m src.main --web
