#!/bin/bash
# PC Auto-sync: 파일 변경 감지 → 자동 commit & push
# 사용법: bash scripts/pc-watch.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH="main"
DEBOUNCE=3  # 변경 감지 후 대기 시간 (초) - 연속 저장 시 중복 push 방지

echo "=============================="
echo " HybridOptimizer PC Watch"
echo " Watching: $REPO_DIR"
echo " Branch:   $BRANCH"
echo " Debounce: ${DEBOUNCE}s"
echo "=============================="

push_changes() {
    cd "$REPO_DIR" || exit 1

    # 변경사항 확인
    if git diff --quiet && git diff --cached --quiet && \
       [ -z "$(git ls-files --others --exclude-standard)" ]; then
        return  # 변경 없음
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${TIMESTAMP}] Changes detected → pushing..."

    git add -A
    git commit -m "sync: ${TIMESTAMP}"

    # pull --rebase로 서버 push와 충돌 방지
    git pull --rebase origin "$BRANCH" --quiet && \
    git push origin "$BRANCH"

    echo "[${TIMESTAMP}] Done."
}

fswatch -o \
    --exclude='\.git' \
    --exclude='__pycache__' \
    --exclude='\.py[oc]$' \
    --exclude='\.venv' \
    --exclude='doc/' \
    "$REPO_DIR" | while read -r; do
        sleep "$DEBOUNCE"
        push_changes
done
