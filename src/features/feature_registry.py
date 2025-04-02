# src/features/feature_registry.py

import inspect
import functools
import json
import datetime
from typing import Dict, Any, Callable, List, Optional, Union, Set
import hashlib

import polars as pl
import pandas as pd

class FeatureRegistry:
    def __init__(self, registry_name: str = "main"):
        """
        Initialize the feature registry
        
        Args:
            registry_name: Name identifier for this registry
        """
        self.features = {}
        self.feature_groups = {}
        self.targets = {}
        self.registry_name = registry_name
        self.feature_importance = {}
        self.test_results = {}
    
    def feature(self, 
                description: str = "", 
                entity_id: Optional[str] = None,
                time_reference: Optional[str] = "contract_start_date",
                relative_time: bool = False,
                dependencies: List[str] = None,
                version: str = "1.0.0"):
        """
        Decorator to register a feature transformation function
        
        Args:
            description: Feature description
            entity_id: ID column for the entity being aggregated (e.g., 'client_id', 'region')
            time_reference: Column name used for temporal calculations
            relative_time: Whether this feature uses relative time calculations
            dependencies: List of other features this feature depends on
            version: Version string for this feature
        """
        def decorator(func):
            feature_name = func.__name__
            source_code = inspect.getsource(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Register feature
            self.features[feature_name] = {
                "function": wrapper,
                "description": description,
                "entity_id": entity_id,
                "time_reference": time_reference,
                "relative_time": relative_time,
                "dependencies": dependencies or [],
                "source_code": source_code,
                "signature": str(inspect.signature(func)),
                "version": version,
                "is_target": False,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            return wrapper
        
        return decorator
    
    def target(self, 
               description: str = "", 
               time_point: Optional[str] = None,
               version: str = "1.0.0"):
        """
        Decorator to register a target variable
        
        Args:
            description: Target description
            time_point: Time point this target is measured at
            version: Version string for this target
        """
        def decorator(func):
            feature_name = func.__name__
            if not feature_name.startswith("tar_"):
                feature_name = f"tar_{feature_name}"
                
            source_code = inspect.getsource(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Register target
            self.targets[feature_name] = {
                "function": wrapper,
                "description": description,
                "time_point": time_point,
                "source_code": source_code,
                "signature": str(inspect.signature(func)),
                "version": version,
                "is_target": True,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Also register in features dict for unified handling
            self.features[feature_name] = self.targets[feature_name]
            
            return wrapper
        
        return decorator
    
    def feature_group(self, name: str, description: str = ""):
        """
        Decorator to define a group of related features
        
        Args:
            name: Group name
            description: Group description
        """
        def decorator(func):
            source_code = inspect.getsource(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Register group
            self.feature_groups[name] = {
                "function": wrapper,
                "description": description,
                "source_code": source_code,
                "members": []
            }
            
            return wrapper
        
        return decorator
    
    def add_to_group(self, group_name: str, feature_name: str):
        """Add a feature to a group"""
        if group_name not in self.feature_groups:
            raise ValueError(f"Feature group '{group_name}' not found in registry")
        
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        if "members" not in self.feature_groups[group_name]:
            self.feature_groups[group_name]["members"] = []
            
        if feature_name not in self.feature_groups[group_name]["members"]:
            self.feature_groups[group_name]["members"].append(feature_name)
    
    def update_importance(self, feature_name: str, importance: float, model_version: str):
        """Update the importance score for a feature from a model run"""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        if feature_name not in self.feature_importance:
            self.feature_importance[feature_name] = []
            
        self.feature_importance[feature_name].append({
            "model_version": model_version,
            "importance": importance,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def record_test_result(self, feature_name: str, test_name: str, passed: bool, 
                         error_message: str = "", execution_time: float = 0.0):
        """Record the result of a feature test"""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
            
        if feature_name not in self.test_results:
            self.test_results[feature_name] = []
            
        self.test_results[feature_name].append({
            "test_name": test_name,
            "passed": passed,
            "error_message": error_message,
            "execution_time": execution_time,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def list_features(self) -> List[Dict[str, Any]]:
        """List all available features with their descriptions"""
        return [
            {
                "name": name,
                "description": info.get("description", ""),
                "entity_id": info.get("entity_id"),
                "time_reference": info.get("time_reference"),
                "relative_time": info.get("relative_time", False),
                "is_target": info.get("is_target", False),
                "version": info.get("version", "1.0.0")
            }
            for name, info in self.features.items()
        ]
    
    def list_feature_groups(self) -> List[Dict[str, Any]]:
        """List all feature groups"""
        return [
            {
                "name": name,
                "description": info.get("description", ""),
                "members": info.get("members", [])
            }
            for name, info in self.feature_groups.items()
        ]
    
    def get_feature_details(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a feature"""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
            
        feature_info = self.features[feature_name].copy()
        
        # Remove function reference for serialization
        if "function" in feature_info:
            del feature_info["function"]
            
        # Add importance data
        feature_info["importance"] = self.feature_importance.get(feature_name, [])
        
        # Add test results
        feature_info["tests"] = self.test_results.get(feature_name, [])
        
        return feature_info
    
    def get_dependency_graph(self, feature_name: str = None) -> Dict[str, List[str]]:
        """Get the dependency graph for features"""
        graph = {}
        
        if feature_name:
            # Get dependencies for a specific feature and its dependencies
            def add_dependencies(feat, graph):
                if feat not in self.features:
                    return
                
                if feat not in graph:
                    graph[feat] = []
                
                deps = self.features[feat].get("dependencies", [])
                graph[feat].extend(deps)
                
                for dep in deps:
                    add_dependencies(dep, graph)
            
            add_dependencies(feature_name, graph)
        
        else:
            # Get all dependencies
            for feat, info in self.features.items():
                if "dependencies" in info and info["dependencies"]:
                    graph[feat] = info["dependencies"]
        
        return graph
    
    def apply_feature(self, df: Union[pl.DataFrame, pd.DataFrame], feature_name: str) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Apply a single feature transformation, automatically handling dependencies
        
        Args:
            df: Input DataFrame (Polars or pandas)
            feature_name: Name of feature to apply
            
        Returns:
            DataFrame with feature added
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        is_polars = isinstance(df, pl.DataFrame)
        
        # First, check if we need to apply dependencies
        dependencies = self.features[feature_name].get("dependencies", [])
        for dep in dependencies:
            # Check if dependency is already in the dataframe
            if is_polars and dep not in df.columns:
                df = self.apply_feature(df, dep)
            elif not is_polars and dep not in df.columns:
                df = self.apply_feature(df, dep)
        
        # Apply the feature function
        result = self.features[feature_name]["function"](df)
        
        return result
    
    def apply_features(self, df: Union[pl.DataFrame, pd.DataFrame], feature_names: List[str]) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Apply multiple feature transformations
        
        Args:
            df: Input DataFrame
            feature_names: List of feature names to apply
            
        Returns:
            DataFrame with features added
        """
        result = df.copy() if isinstance(df, pd.DataFrame) else df.clone()
        
        # Use a set to track which features have been applied
        applied = set()
        
        # Apply features with dependency handling
        for name in feature_names:
            if name not in applied:
                result = self._apply_feature_with_deps(result, name, applied)
        
        return result
    
    def _apply_feature_with_deps(self, df: Union[pl.DataFrame, pd.DataFrame], 
                                feature_name: str, 
                                applied: Set[str]) -> Union[pl.DataFrame, pd.DataFrame]:
        """Helper method to apply a feature with dependency tracking"""
        if feature_name in applied:
            return df
            
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        # First apply dependencies
        dependencies = self.features[feature_name].get("dependencies", [])
        for dep in dependencies:
            if dep not in applied:
                df = self._apply_feature_with_deps(df, dep, applied)
        
        # Now apply this feature
        result = self.features[feature_name]["function"](df)
        applied.add(feature_name)
        
        return result
    
    def apply_feature_group(self, df: Union[pl.DataFrame, pd.DataFrame], group_name: str) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Apply all features in a feature group
        
        Args:
            df: Input DataFrame
            group_name: Name of the feature group
            
        Returns:
            DataFrame with all features in the group added
        """
        if group_name not in self.feature_groups:
            raise ValueError(f"Feature group '{group_name}' not found in registry")
        
        # If the group has a direct function implementation, use that
        if "function" in self.feature_groups[group_name]:
            return self.feature_groups[group_name]["function"](df)
        
        # Otherwise, apply each feature in the group
        members = self.feature_groups[group_name].get("members", [])
        if not members:
            return df  # Empty group
            
        return self.apply_features(df, members)
    
    def export_documentation(self, format: str = "markdown", output_path: Optional[str] = None) -> str:
        """
        Export feature documentation in the specified format
        
        Args:
            format: Output format ("markdown" or "json")
            output_path: Path to save documentation file
            
        Returns:
            Documentation string
        """
        if format == "markdown":
            docs = f"# Feature Documentation - {self.registry_name}\n\n"
            docs += f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            
            # Document feature groups
            if self.feature_groups:
                docs += "## Feature Groups\n\n"
                for name, group in self.feature_groups.items():
                    docs += f"### {name}\n\n"
                    docs += f"{group.get('description', '')}\n\n"
                    
                    # List features in this group
                    members = group.get("members", [])
                    if members:
                        docs += "**Features in this group:**\n\n"
                        for feature in members:
                            docs += f"- {feature}\n"
                        docs += "\n"
            
            # Document targets
            targets = {name: info for name, info in self.targets.items()}
            if targets:
                docs += "## Target Variables\n\n"
                for name, target in targets.items():
                    docs += f"### {name}\n\n"
                    docs += f"**Description:** {target.get('description', '')}\n\n"
                    
                    if "time_point" in target and target["time_point"]:
                        docs += f"**Time Point:** {target['time_point']}\n\n"
                        
                    docs += f"**Version:** {target.get('version', '1.0.0')}\n\n"
            
            # Document individual features (excluding targets)
            features = {name: info for name, info in self.features.items() 
                      if not info.get("is_target", False)}
            
            if features:
                docs += "## Individual Features\n\n"
                for name, feature in features.items():
                    docs += f"### {name}\n\n"
                    docs += f"**Description:** {feature.get('description', '')}\n\n"
                    
                    if "entity_id" in feature and feature["entity_id"]:
                        docs += f"**Entity ID:** {feature['entity_id']}\n\n"
                        
                    if "time_reference" in feature and feature["time_reference"]:
                        docs += f"**Time Reference:** {feature['time_reference']}\n\n"
                        
                    if feature.get("relative_time", False):
                        docs += "**Uses Relative Time:** Yes\n\n"
                    
                    # List dependencies
                    deps = feature.get("dependencies", [])
                    if deps:
                        docs += "**Dependencies:**\n"
                        for dep in deps:
                            docs += f"- {dep}\n"
                        docs += "\n"
                    
                    # Show importance metrics if available
                    importance = self.feature_importance.get(name, [])
                    if importance:
                        # Sort by timestamp (newest first)
                        sorted_importance = sorted(
                            importance, 
                            key=lambda x: x.get("timestamp", ""), 
                            reverse=True
                        )[:5]  # Show at most 5 recent entries
                        
                        docs += "**Recent Importance Scores:**\n\n"
                        docs += "| Model Version | Importance |\n"
                        docs += "|--------------|------------|\n"
                        for imp in sorted_importance:
                            docs += f"| {imp['model_version']} | {imp['importance']:.4f} |\n"
                        docs += "\n"
                    
                    # Show implementation
                    docs += "**Implementation:**\n\n"
                    docs += "```python\n"
                    docs += feature.get("source_code", "# Source code not available")
                    docs += "```\n\n"
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(docs)
                print(f"Documentation written to {output_path}")
                
            return docs
            
        elif format == "json":
            # Prepare full documentation structure
            docs = {
                "registry_name": self.registry_name,
                "generation_time": datetime.datetime.now().isoformat(),
                "features": {},
                "targets": {},
                "groups": {}
            }
            
            # Add features
            for name, feature in self.features.items():
                if feature.get("is_target", False):
                    continue  # Skip targets here
                    
                feature_copy = feature.copy()
                if "function" in feature_copy:
                    del feature_copy["function"]  # Remove non-serializable function
                    
                feature_copy["importance"] = self.feature_importance.get(name, [])
                feature_copy["tests"] = self.test_results.get(name, [])
                
                docs["features"][name] = feature_copy
            
            # Add targets
            for name, target in self.targets.items():
                target_copy = target.copy()
                if "function" in target_copy:
                    del target_copy["function"]  # Remove non-serializable function
                    
                docs["targets"][name] = target_copy
            
            # Add groups
            for name, group in self.feature_groups.items():
                group_copy = {
                    "description": group.get("description", ""),
                    "members": group.get("members", [])
                }
                
                docs["groups"][name] = group_copy
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(docs, f, indent=2)
                print(f"Documentation written to {output_path}")
                
            return json.dumps(docs, indent=2)
            
        else:
            raise ValueError(f"Unsupported documentation format: {format}")
        
    def get_feature_documentation_for_agent(self) -> Dict[str, Any]:
        """
        Return feature documentation in a format suitable for agent consumption
        
        Returns:
            Dictionary with feature documentation
        """
        return {
            name: {
                "description": info.get("description", ""),
                "importance": self.feature_importance.get(name, []),
                "entity": info.get("entity_id"),
                "dependencies": info.get("dependencies", []),
                "is_target": info.get("is_target", False)
            }
            for name, info in self.features.items()
        }


# Create a global instance of the registry
feature_registry = FeatureRegistry()

# Export convenience functions
feature = feature_registry.feature
target = feature_registry.target
feature_group = feature_registry.feature_group
apply_feature = feature_registry.apply_feature
apply_features = feature_registry.apply_features
apply_feature_group = feature_registry.apply_feature_group
list_features = feature_registry.list_features
list_feature_groups = feature_registry.list_feature_groups
export_documentation = feature_registry.export_documentation