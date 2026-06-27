# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Flask application entry point (API/service).
- `requirements.txt`: Python dependencies.
- `planning.md`: Product/architecture notes (not runtime code).
- `README.md`: Project overview (currently empty; consider expanding).

## Build, Test, and Development Commands
Run everything from the repo root.
- Create/activate a virtualenv (recommended): `python -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run locally (Flask): `python main.py` (or `flask --app main run` if configured).
- Environment variables: use a `.env` file (loaded via `python-dotenv`) for secrets like API keys.

## Coding Style & Naming Conventions
- Python: follow PEP 8 (4-space indentation).
- Prefer type hints for new functions.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep Flask routes and handlers small; move reusable logic into helper functions/modules if the file grows.

## Testing Guidelines
- No test suite is present yet. If adding tests, prefer `pytest` and place them under `tests/`.
- Naming: `tests/test_*.py` and `test_*` functions.
- Add minimal coverage for API routes (happy path + common failures like missing inputs/rate limits).

## Commit & Pull Request Guidelines
- Use imperative, scoped commit messages when possible: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`.
- PRs should include: a clear description, linked issue (if any), and notes on how to run/verify the change.
- For API behavior changes, include example requests/responses or curl snippets.

## Security & Configuration Tips
- Do not commit `.env` or secrets.
- Validate and sanitize user input at the API boundary; keep rate limiting (`flask-limiter`) enabled where applicable.
