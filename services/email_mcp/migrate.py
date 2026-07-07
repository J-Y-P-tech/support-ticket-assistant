"""Standalone migration runner — the entrypoint behind `make migrate`.

Applies every `migrations/*.sql` file to the ticket database using email_mcp's
own credentials (SPEC §6 least-privilege). The user runs this; it is idempotent.
"""

from __future__ import annotations

import db


def main() -> None:
    """Open a connection from the environment and apply all migrations."""
    with db.connect_from_env() as conn:
        db.apply_migrations(conn)
    print("migrations applied.")


if __name__ == "__main__":
    main()
