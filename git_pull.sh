#!/bin/bash

# 一键拉取 GitHub 最新代码脚本
# 使用方式：./git_pull.sh

# 确保在 main 分支上
git checkout main

# 拉取远程主分支更新
git reset --hard
git pull origin main

echo "✅ 已成功从远程仓库拉取最新代码（main 分支）"
