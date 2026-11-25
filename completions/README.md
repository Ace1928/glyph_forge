# Shell completion

Glyph Forge delegates completion generation to Typer so scripts always match
the installed command version.

Install completion for the current shell:

```bash
glyph-forge --install-completion
```

Print a script for manual inspection or installation:

```bash
glyph-forge --show-completion bash
glyph-forge --show-completion zsh
glyph-forge --show-completion fish
glyph-forge --show-completion powershell
```

Restart the shell after installation. The command prints the destination and
does not modify unrelated shell configuration.
