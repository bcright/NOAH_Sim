#!/bin/bash

# 定义 Python 包的安装路径
PYTHON_BIN_PATH=$HOME/.local/bin

# 确保 ~/.local/bin 路径存在
mkdir -p $PYTHON_BIN_PATH

# 安装 Python 包
pip install --user flask gym numpy torch psutil locust requests

# 检查 ~/.bashrc 文件中是否已经添加了路径
if ! grep -q "export PATH=\$PATH:$PYTHON_BIN_PATH" $HOME/.bashrc; then
    echo "export PATH=\$PATH:$PYTHON_BIN_PATH" >> $HOME/.bashrc
    echo "PATH updated in .bashrc, changes will be applied in the next terminal session."
fi

# 重新加载 .bashrc 文件以立即应用更改
source $HOME/.bashrc

echo "Installation complete. All packages are added to PATH."