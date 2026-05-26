"""
Utility module for parameter loading from YAML files.

This module provides utilities for reading and managing configuration parameters,
following the structure used in similar projects.
"""

import yaml
from pathlib import Path


class YParams:
    """
    YAML parameter loader utility.
    
    Loads configuration from YAML files and provides easy access to parameters.
    """
    
    def __init__(self, yaml_file):
        """
        Initialize YParams with a YAML configuration file.
        
        Args:
            yaml_file (str): Path to YAML configuration file
        """
        self.yaml_file = Path(yaml_file)
        
        if not self.yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")
        
        with open(self.yaml_file, 'r') as f:
            self.params = yaml.safe_load(f)
    
    def __getitem__(self, key):
        """Access parameters using dictionary syntax."""
        return self.params[key]
    
    def __contains__(self, key):
        """Check if parameter exists."""
        return key in self.params
    
    def get(self, key, default=None):
        """Get parameter with optional default value."""
        return self.params.get(key, default)
    
    def to_dict(self):
        """Return all parameters as a dictionary."""
        return self.params
    
    def __repr__(self):
        return f"YParams({self.yaml_file})"
