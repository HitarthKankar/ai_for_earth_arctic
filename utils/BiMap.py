"""
Bidirectional mapping utility.

Provides a bidirectional dictionary-like structure for mapping band names to indices
and vice versa.
"""


class BiMap:
    """
    Bidirectional map for looking up values by key or key by value.
    
    Useful for mapping between band indices and band names.
    """
    
    def __init__(self, mapping):
        """
        Initialize BiMap.
        
        Args:
            mapping (dict): Dictionary with unique keys and unique values
        """
        self.forward = mapping
        self.backward = {v: k for k, v in mapping.items()}
    
    def get_forward(self, key):
        """Get value from key."""
        return self.forward.get(key)
    
    def get_backward(self, value):
        """Get key from value."""
        return self.backward.get(value)
    
    def __repr__(self):
        return f"BiMap({self.forward})"


# Example: Band index to name mapping
BAND_INDEX_TO_NAME = BiMap({
    0: 'B2',
    1: 'B3',
    2: 'B4',
    3: 'B5',
    4: 'B6',
    5: 'B7',
    6: 'B8',
    7: 'B8a',
    8: 'B11',
    9: 'B12'
})
