"""Native local folder selection for the browser-based Recall UI."""

from pathlib import Path


def choose_vault_directory() -> Path | None:
    """Open a native directory chooser on the machine running the local API."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("The native folder chooser is unavailable.") from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError("The native folder chooser is unavailable.") from error

    try:
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askdirectory(
            parent=root,
            title="Select an Obsidian vault folder",
            mustexist=True,
        )
    except tk.TclError as error:
        raise RuntimeError("The native folder chooser is unavailable.") from error
    finally:
        root.destroy()

    return Path(selected_path) if selected_path else None
