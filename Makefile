# HybridOptimizer Sync Makefile

.PHONY: watch push pull

# PC에서 파일 감시 시작 (변경 시 자동 push)
watch:
	bash scripts/pc-watch.sh

# 수동 push
push:
	git add -A && git commit -m "sync: $$(date '+%Y-%m-%d %H:%M:%S')" && git push origin main

# 수동 pull
pull:
	git pull --rebase origin main
