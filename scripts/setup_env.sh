#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Lumen Debug Environment Bootstrap
#
# Usage (on a fresh AMD GPU node with ROCm pre-installed):
#
#   # Step 1: manual installs (need interaction)
#   curl -fsSL https://claude.ai/install.sh | bash -s stable
#   bash <(curl -fsSL https://raw.githubusercontent.com/DaiXindi-AMD/Lumen/main/scripts/setup_env.sh)
#
#   # Or if you already cloned Lumen:
#   bash ~/Lumen/scripts/setup_env.sh
#
# Idempotent — safe to re-run.
# ===========================================================================

API_KEY="${LUMEN_API_KEY:-f6a8446b100b4081bf679c47be0a14c9}"  # override via env var

# ---------------------------------------------------------------------------
echo "=== [1/7] Oh-My-Zsh + Plugins ==="
# ---------------------------------------------------------------------------
if [ ! -d ~/.oh-my-zsh ]; then
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
[ -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]         || git clone https://github.com/zsh-users/zsh-autosuggestions "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
[ -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]     || git clone https://github.com/zsh-users/zsh-syntax-highlighting "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
[ -d "$ZSH_CUSTOM/plugins/zsh-history-substring-search" ] || git clone https://github.com/zsh-users/zsh-history-substring-search "$ZSH_CUSTOM/plugins/zsh-history-substring-search"

# ---------------------------------------------------------------------------
echo "=== [2/7] Shell Config ==="
# ---------------------------------------------------------------------------

# ~/.profile: auto-switch to zsh
grep -q 'ZSH_SWITCHOVER' ~/.profile 2>/dev/null || cat >> ~/.profile << 'PROFILE_EOF'

# Switch to zsh
if [ -x /usr/bin/zsh ] && [ "$ZSH_SWITCHOVER" != "1" ]; then
    export ZSH_SWITCHOVER=1
    exec /usr/bin/zsh -l
fi
PROFILE_EOF

# ~/.zshrc: remove old block, append fresh
sed -i '/^# === Claude Code + AMD LLM Gateway ===/,/^# === END Claude Block ===/d' ~/.zshrc 2>/dev/null || true

cat >> ~/.zshrc << ZSHRC_EOF

# === Claude Code + AMD LLM Gateway ===
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${API_KEY}
user: \$(whoami)"
export DISABLE_PROMPT_CACHING="1"
export ANTHROPIC_MODEL="claude-opus-4.6[1m]"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4.6[1m]"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4.6"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4.5"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export ENABLE_TOOL_SEARCH=true
alias claude='claude --model claude-opus-4.6'
alias cda="claude --permission-mode dontAsk"
alias ca="claude --mode auto-accept"
alias cb="claude --permission-mode bypassPermissions"
# === END Claude Block ===
ZSHRC_EOF

if grep -q '^plugins=' ~/.zshrc; then
    sed -i 's/^plugins=.*/plugins=(git z extract history zsh-autosuggestions zsh-syntax-highlighting zsh-history-substring-search)/' ~/.zshrc
fi
grep -q 'HOME/.local/bin' ~/.zshrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

# ---------------------------------------------------------------------------
echo "=== [3/7] Claude Code Settings ==="
# ---------------------------------------------------------------------------
mkdir -p ~/.claude

cat > ~/.claude/settings.json << 'SETTINGS_EOF'
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/Anthropic",
    "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: API_KEY_PLACEHOLDER",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4.6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4.6[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4.6[1m]",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "ENABLE_TOOL_SEARCH": "true"
  },
  "model": "opus",
  "enabledPlugins": {
    "superpowers@superpowers-marketplace": true
  },
  "extraKnownMarketplaces": {
    "superpowers-marketplace": {
      "source": {
        "source": "github",
        "repo": "obra/superpowers-marketplace"
      }
    }
  },
  "autoUpdatesChannel": "stable",
  "theme": "dark"
}
SETTINGS_EOF

sed -i "s/API_KEY_PLACEHOLDER/${API_KEY}/" ~/.claude/settings.json

# ---------------------------------------------------------------------------
echo "=== [4/7] Clone Lumen ==="
# ---------------------------------------------------------------------------
if [ ! -d ~/Lumen ]; then
    git clone https://github.com/ZhangDanyang-AMD/Lumen.git ~/Lumen
    cd ~/Lumen
    git remote add mine https://github.com/DaiXindi-AMD/Lumen.git 2>/dev/null || true
    git fetch mine
    git submodule update --init --recursive
else
    echo "Lumen already exists, updating..."
    cd ~/Lumen
    git fetch origin
    git remote add mine https://github.com/DaiXindi-AMD/Lumen.git 2>/dev/null || true
    git fetch mine
    git submodule update --init --recursive
fi

# ---------------------------------------------------------------------------
echo "=== [5/7] Clone & Install AITER + TorchAO ==="
# ---------------------------------------------------------------------------
if [ ! -d ~/aiter ]; then
    git clone https://github.com/ROCm/aiter.git ~/aiter
fi
cd ~/aiter && git pull origin main 2>/dev/null || true
pip3 install -e . 2>/dev/null || echo "WARN: AITER pip install failed (may need sudo)"

if [ ! -d ~/ao ]; then
    git clone https://github.com/pytorch/ao.git ~/ao
fi
cd ~/ao && git pull origin main 2>/dev/null || true
pip3 install -e . 2>/dev/null || echo "WARN: TorchAO pip install failed (may need sudo)"

# ---------------------------------------------------------------------------
echo "=== [6/7] Install Lumen ==="
# ---------------------------------------------------------------------------
cd ~/Lumen
pip3 install -e ".[dev]" 2>/dev/null || pip3 install -e . 2>/dev/null || echo "WARN: Lumen pip install failed (may need sudo)"

# ---------------------------------------------------------------------------
echo "=== [7/7] Triton 3.7 Shim ==="
# ---------------------------------------------------------------------------
# Triton 3.7 + AITER fork: inspect.getsourcelines() misses the `def` line
# inside @triton.heuristics decorators → crash. This shim monkey-patches the
# source-location lookup. Only affects debug metadata, not computation.
# /tmp clears on reboot — re-run this script to restore.

mkdir -p /tmp/tritonshim
cat > /tmp/tritonshim/sitecustomize.py << 'SHIM_EOF'
try:
    import re, inspect, textwrap, threading
    import triton.runtime.jit as J

    _orig_col = J.get_def_col_number
    def _safe_col(s):
        try: return _orig_col(s)
        except Exception: return 1
    J.get_def_col_number = _safe_col

    _orig_init = J.JITCallable.__init__
    def _safe_init(self, fn):
        try:
            _orig_init(self, fn)
        except AttributeError:
            self.fn = fn
            self.signature = inspect.signature(fn)
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
            self.file_name = fn.__code__.co_filename
            self._fn_name = J.get_full_name(fn)
            self._hash_lock = threading.RLock()
            self.def_file_line_number = self.starting_line_number
            self.def_file_col_number = 1
            src = textwrap.dedent("".join(self.raw_src))
            m = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
            self._src = src[m.start():] if m else src
            self.hash = None
    J.JITCallable.__init__ = _safe_init
except Exception:
    pass
SHIM_EOF

echo ""
echo "============================================"
echo "  Done! Open a new shell:  exec zsh -l"
echo ""
echo "  Verify:"
echo "    claude --version"
echo "    python3 -c \"import torch; print(torch.__version__)\""
echo "    python3 -c \"import torchao; print(torchao.__version__)\""
echo ""
echo "  Run MXFP4 tests:"
echo "    cd ~/Lumen"
echo "    PYTHONPATH=/tmp/tritonshim:\$PWD/third_party/aiter \\"
echo "      python -m pytest tests/ops/test_quantize.py -v -k mxfp4"
echo "============================================"
