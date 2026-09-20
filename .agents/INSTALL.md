# 在另一台机器同步 Lumen Codex skills

源仓库：`https://github.com/DaiXindi-AMD/Lumen.git`

同步分支：`codex/lumen-skills`

技能目录：`.agents/skills/`

包含 `lumen-coding`、`lumen-test`、`lumen-benchmark`、`lumen-sdma`、`lumen-rl`、`lumen-training`、`lumen-aiter`，以及各自的参考文件和 `agents/openai.yaml`。下载完整目录，保留这 7 个技能作为同级目录，以便相互引用。

## 首次安装

推荐保留专门的稀疏 checkout，并链接到用户级 `~/.agents/skills/`，使它们在 Lumen 和 AITER 工作目录中都可使用。下载不需要 GPU，也不需要安装 Lumen/AITER 或初始化子模块。

以下命令用于目标路径尚不存在的首次安装。如果 checkout 或同名技能已经存在，先核对来源；复用正确的 checkout 和软链接。对于不同来源或包含本地改动的旧副本，先备份到技能扫描目录之外，再更新，避免直接覆盖。

```bash
lumen_skill_repo="$HOME/.local/share/codex/lumen-skills"
mkdir -p "$(dirname "$lumen_skill_repo")"

git clone --depth 1 --filter=blob:none --sparse --single-branch \
  --branch codex/lumen-skills \
  https://github.com/DaiXindi-AMD/Lumen.git "$lumen_skill_repo"
git -C "$lumen_skill_repo" sparse-checkout set .agents

mkdir -p "$HOME/.agents/skills"
for lumen_skill_name in lumen-coding lumen-test lumen-benchmark lumen-sdma lumen-rl lumen-training lumen-aiter; do
  ln -s "$lumen_skill_repo/.agents/skills/$lumen_skill_name" \
    "$HOME/.agents/skills/$lumen_skill_name"
done
```

安装时检查 `~/.agents/skills/` 和旧环境的 `${CODEX_HOME:-$HOME/.codex}/skills/` 中是否有这 7 个同名技能，保留一个生效来源。不要改动其他技能或 `.system`。若已经从项目级 `.agents/skills/` 加载同一套技能，也应选择项目级或用户级其中一种发现方式，避免重复列出。

## 验证

- 7 个目标目录均包含可读的 `SKILL.md`，名称与目录一致。
- 各自的参考资料、`agents/openai.yaml` 和跨技能相对链接完整。
- 如果本机有 `skill-creator/scripts/quick_validate.py`，对这 7 个目录逐一运行。
- 在 Codex 的技能列表中确认可发现它们；更新未显示时重启 Codex。可以用 `$lumen-aiter` 显式调用联合技能。
- 报告实际下载的 `git rev-parse HEAD`、安装位置、检查结果，以及有无旧副本备份。

## 后续更新

先确认这个专用 checkout 没有未保存的本地改动，然后执行：

```bash
git -C "$HOME/.local/share/codex/lumen-skills" pull --ff-only
```

软链接自动指向更新后的内容，无须重新复制。若分支分叉，保留本地改动并说明情况，不要自动 `reset --hard`。将来此分支合入主分支后，再明确切换同步来源。

技能中用于训练调试的 `.codex/tmp-training-bugs.md` 是各个实际 Lumen 工作目录自己的运行记录，不存放在这个全局技能包里。稀疏 checkout 只提供技能；执行技能时仍需定位用户真正工作的 Lumen/AITER 仓库。

本目录遵循 [Codex 官方 skills 目录与软链接约定](https://developers.openai.com/codex/skills/)。
