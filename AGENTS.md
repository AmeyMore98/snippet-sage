# AGENTS.md

## Dev environment tips

- Use `source .venv/bin/activate` to start the virtual environment.
- You run in an environment where ast-grep is available; whenever a search requires syntax-aware or structural matching, default to `ast-grep --lang python3 -p '<pattern>'` (or set --lang appropriately) and avoid falling back to text-only tools like rg or grep unless I explicitly request a plain-text search.

## Testing instructions

- Tests are located in `app/tests`.
- Add or update tests for the code you change, even if nobody asked.

## Misc. instructions

- At the end of each task, always thoroughly explain your decisions
