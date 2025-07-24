#!/bin/bash

# 自动版本号 Git 提交脚本：每次提交自动生成 v1, v2, v3…
# 使用方式：./git_push.sh "这次做了什么修改"

# 检查是否提供了提交说明
if [ -z "$1" ]; then
  echo "❌ 错误：请提供提交说明，例如：./git_push.sh \"更新训练集\""
  exit 1
fi

# 确保在 main 分支上操作
git checkout main

# 自动读取并更新版本号
VERSION_FILE="version.txt"
if [ ! -f "$VERSION_FILE" ]; then
  echo "v1" > "$VERSION_FILE"
  VERSION="v1"
else
  LAST_VERSION=$(cat $VERSION_FILE)
  NUM=${LAST_VERSION#v}
  NEW_NUM=$((NUM + 1))
  VERSION="v$NEW_NUM"
  echo "$VERSION" > "$VERSION_FILE"
fi

# 加入 version.txt 到提交内容
git add -A

# 拼接完整提交信息
COMMIT_MSG="$VERSION: $1"
git commit -m "$COMMIT_MSG"

# 推送到远程仓库
git push origin main

echo "✅ [$VERSION] 提交并推送成功：$1"
