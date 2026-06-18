"""
BISINDO Gesture Guide Database (Placeholder)
=============================================
Database of BISINDO gesture guides for common words.
Phase 4 implementation - currently placeholder.
"""


class GestureGuideDB:
    """Database of BISINDO gesture descriptions and guides."""

    def __init__(self, db_path=None):
        """Initialize gesture guide database."""
        self.db_path = db_path
        self.guides = {}

    def get_guide(self, word):
        """
        Get gesture guide for a word.

        Args:
            word: Indonesian word

        Returns:
            guide: dict with gesture description, or None
        """
        # Phase 4: Implement gesture guide database
        raise NotImplementedError("Gesture guide database will be implemented in Phase 4")
