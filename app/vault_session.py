"""In-memory selection state for the active local vault."""

import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from app.config import get_settings
from app.ingest import list_vault_entries
from app.models import VaultEntry


@dataclass(frozen=True)
class SelectedVault:
    vault_id: UUID
    name: str
    path: Path


class VaultSession:
    """Keep the user-selected vault path out of API requests and responses."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._selected: SelectedVault | None = None

    def select(self, path: Path) -> tuple[SelectedVault, list[VaultEntry]]:
        if path.is_symlink() or path.is_junction():
            raise ValueError("Select a real directory, not a symbolic link or junction.")

        resolved_path = path.resolve(strict=True)
        entries = list_vault_entries(resolved_path)
        selection = SelectedVault(
            vault_id=uuid4(),
            name=resolved_path.name or "Vault",
            path=resolved_path,
        )
        with self._lock:
            self._selected = selection
        return selection, entries

    def current(self) -> SelectedVault | None:
        with self._lock:
            return self._selected

    def resolve(self, vault_id: UUID) -> SelectedVault | None:
        with self._lock:
            selection = self._selected
        if selection is None or selection.vault_id != vault_id:
            return None

        if selection.path.is_symlink() or selection.path.is_junction():
            return None
        try:
            resolved_path = selection.path.resolve(strict=True)
        except OSError:
            return None
        if not resolved_path.is_dir() or resolved_path != selection.path:
            return None
        return selection

    def clear(self) -> None:
        with self._lock:
            self._selected = None


vault_session = VaultSession()


def collection_name_for_vault(vault_path: Path) -> str:
    """Return a stable collection name isolated by vault path and embedding model."""
    settings = get_settings()
    resolved_path = vault_path.resolve(strict=True)
    identity = f"{resolved_path.as_posix().casefold()}|{settings.local_embedding_model}"
    digest = sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{settings.local_embedding_collection_prefix}_{digest}"
