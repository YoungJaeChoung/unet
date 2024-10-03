# Set Path: for Jupyter
ipython profile create
mkdir -p ~/.ipython/profile_default/startup
IPYTHON_DIR="$HOME/.ipython/profile_default/startup"  # Set IPython profile directory
SCRIPT_NAME="00-env.py"  # Set script file name
mkdir -p "$PWD"  # Create IPython profile directory

cat << EOF > "$IPYTHON_DIR/$SCRIPT_NAME"  # Create Python script
import sys
sys.path.append('$PYTHONPATH')
EOF

echo ""
echo "IPython startup script created at $IPYTHON_DIR/$SCRIPT_NAME"
echo "PYTHONPATH will be updated with: $PWD"
echo "This path will be added to 'sys.path' in Jupyter environment."
echo "Please restart Jupyter Notebook for changes to take effect."
