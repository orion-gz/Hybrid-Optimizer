#!/bin/bash
# Server Auto-sync: 주기적으로 GitHub 확인 → 변경 시 자동 pull
# 실험 결과 생성 시 자동 push도 지원
# 사용법: bash scripts/server-sync.sh
# 백그라운드 실행: nohup bash scripts/server-sync.sh > sync.log 2>&1 &

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH="main"
INTERVAL=3600     # GitHub 폴링 간격 (초)
AUTO_PUSH=true    # 서버 변경사항 자동 push 여부 (false로 끄기 가능)

echo "=============================="
echo " HybridOptimizer Server Sync"
echo " Repo:      $REPO_DIR"
echo " Branch:    $BRANCH"
echo " Interval:  ${INTERVAL}s"
echo " Auto-push: $AUTO_PUSH"
echo "=============================="

while true; do
    cd "$REPO_DIR" || exit 1
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # 서버 로컬 변경사항 → 자동 push (실험 결과 등)
    if [ "$AUTO_PUSH" = true ]; then
        if ! git diff --quiet || ! git diff --cached --quiet || \
           [ -n "$(git ls-files --others --exclude-standard)" ]; then
            echo "[${TIMESTAMP}] Local changes detected → pushing..."
            git add -A
            git commit -m "result: ${TIMESTAMP}"
            git push origin "$BRANCH"
            echo "[${TIMESTAMP}] Pushed."
        fi
    fi

    # 리모트 변경사항 확인 → 자동 pull
    git fetch origin "$BRANCH" --quiet
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse "origin/${BRANCH}")

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "[${TIMESTAMP}] New commits on GitHub → pulling..."
        git pull --rebase origin "$BRANCH"
        echo "[${TIMESTAMP}] Pulled."
    fi

    sleep "$INTERVAL"
done
