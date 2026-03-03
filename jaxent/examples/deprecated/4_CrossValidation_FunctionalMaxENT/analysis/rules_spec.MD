# Rules-Based Clustering Specification (`--json-rules-spec`)

This document explains how to use a JSON file to define rules for clustering frames based on calculated structural features in the `cluster_LocalFeatures_PUF.py` script. This provides a deterministic alternative to k-means clustering, allowing for classification based on physical criteria with support for complex boolean logic operations and composable transforms.

## Command-Line Argument

To use this functionality, provide the `--json-rules-spec` argument with the path to your rules JSON file. When this argument is used, k-means clustering is skipped, and these rules are used instead.

```bash
python jaxent/examples/2_CrossValidation/analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "path/to/your.pdb,path/to/your.xtc" \
    --names "MyEnsemble" \
    --json_data_path /path/to/key_residues.json \
    --json_feature_spec /path/to/feature_spec.json \
    --json_rules_spec /path/to/my_rules.json \
    --input_dir /path/to/analysis_output \
    --output_dir /path/to/output
```

## Core Concepts

### Two-Stage System (Current Format)
The current format uses a composable two-stage approach:

1. **Transforms**: Named boolean transformations of features that can be reused
2. **Clusters**: Rules using features and/or transforms with boolean logic

### Cluster Priority System
- Clusters are evaluated in numerical order (0, 1, 2, ...)
- **First matching cluster wins** - lower numbers have higher priority
- This allows for hierarchical definitions where specific states (cluster 0) take precedence over general states (cluster 1, 2, etc.)
- Frames not matching any cluster are assigned to cluster `-1` (unclassified)

### Rule Evaluation Logic
- A frame is assigned to a cluster if it satisfies **all** the rules defined for that cluster
- Rules can be simple threshold-based conditions or complex boolean expressions
- Transforms are evaluated first, then clusters can reference both features and transforms
- The system supports backward compatibility with simple rule lists

## JSON Schema (Current Format)

### Top-Level Structure
```json
{
  "transforms": {
    "transform_name": {
      "description": "Human-readable description",
      "rule": { /* rule expression */ }
    }
  },
  "clusters": {
    "0": {
      "description": "Human-readable description", 
      "rule": { /* rule expression */ }
    }
  }
}
```

### Transform Definition
Named boolean transformations that can be reused:

```json
{
  "transforms": {
    "my_transform": {
      "description": "Feature within specific range",
      "rule": {
        "spec_name": "feature_name",
        "min_threshold": -1.0,
        "max_threshold": 1.0,
        "inclusive": true
      }
    }
  }
}
```

### Cluster Definition
Rules that can reference features and transforms:

```json
{
  "clusters": {
    "0": {
      "description": "Cluster using transforms and features",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "my_transform"},     // References transform
          {"spec_name": "direct_feature", "min_threshold": 0.0, "max_threshold": 1.0}
        ]
      }
    }
  }
}
```

### Simple Rule Object
Basic threshold-based rule for a single feature:

```json
{
  "spec_name": "feature_name",        // Required: name from feature_spec.json or transform
  "min_threshold": -1.0,              // Required: minimum value (not used for transforms)
  "max_threshold": 1.0,               // Required: maximum value (not used for transforms)  
  "inclusive": true                   // Optional: true (within) or false (outside range)
}
```

### Boolean Logic Operation
Complex rule with logical operators:

```json
{
  "operator": "AND|OR|NOT",           // Required: logical operator
  "operands": [...]                   // Required for AND/OR: list of rules or operations
  "operand": {...}                    // Required for NOT: single rule or operation
}
```

## Complete Example (Current Format)

```json
{
  "transforms": {
    "beta_sheet_intact": {
      "description": "Both beta strands maintain proper phi/psi angles",
      "rule": {
        "operator": "AND",
        "operands": [
          {
            "spec_name": "dih_b1_integrity_phi",
            "min_threshold": -20.0,
            "max_threshold": 20.0,
            "inclusive": true
          },
          {
            "spec_name": "dih_b2_integrity_phi", 
            "min_threshold": -20.0,
            "max_threshold": 20.0,
            "inclusive": true
          }
        ]
      }
    },
    
    "helix_disrupted": {
      "description": "Alpha helix shows significant deviation",
      "rule": {
        "spec_name": "dih_a1_integrity_phi",
        "min_threshold": -10.0,
        "max_threshold": 10.0,
        "inclusive": false
      }
    }
  },
  
  "clusters": {
    "0": {
      "description": "Native folded state - sheets intact, helix normal",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "beta_sheet_intact"},
          {
            "operator": "NOT",
            "operand": {"spec_name": "helix_disrupted"}
          }
        ]
      }
    },
    
    "1": {
      "description": "Partially unfolded - disrupted helix but sheets OK",
      "rule": {
        "operator": "AND", 
        "operands": [
          {"spec_name": "beta_sheet_intact"},
          {"spec_name": "helix_disrupted"}
        ]
      }
    },
    
    "2": {
      "description": "Significantly unfolded - major structural disruption", 
      "rule": {
        "operator": "NOT",
        "operand": {"spec_name": "beta_sheet_intact"}
      }
    }
  }
}
```

## Rule Parameters

### Transform Parameters
- **`description`** (Optional): Human-readable description of the transform
- **`rule`** (Required): Rule expression that evaluates to boolean

### Cluster Parameters  
- **`description`** (Optional): Human-readable description of the cluster
- **`rule`** (Required): Rule expression for cluster assignment

### Rule Expression Parameters

#### `spec_name` (Required)
- **Type**: String
- **Description**: Feature name OR transform name to apply the rule to
- **Features**: Must match a `spec_name` defined in your `--json-feature-spec` file
- **Transforms**: Must match a transform name defined in the `transforms` section
- **Note**: If the name is not found, the rule will be skipped with a warning

#### `min_threshold` and `max_threshold` (Required for features)
- **Type**: Float
- **Description**: Define the range for the feature value
- **Usage**: Used with `inclusive` parameter to determine rule satisfaction
- **Note**: Not used when referencing transforms (transforms are already boolean)

#### `inclusive` (Optional)
- **Type**: Boolean
- **Default**: `true`
- **Behavior**:
  - `true`: Rule satisfied if `min_threshold ≤ feature_value ≤ max_threshold`
  - `false`: Rule satisfied if `feature_value < min_threshold OR feature_value > max_threshold`

#### `operator` (Required for boolean operations)
- **Type**: String
- **Valid values**: `"AND"`, `"OR"`, `"NOT"` (case-insensitive)
- **Description**: Logical operator for combining multiple conditions

#### `operands` vs `operand`
- **`operands`**: Array of rules/operations (required for AND/OR)
- **`operand`**: Single rule/operation (required for NOT)

## Advanced Transform Examples

### Complex Structural States
```json
{
  "transforms": {
    "core_stable": {
      "description": "Core structural elements are stable",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "core_rmsd", "min_threshold": 0.0, "max_threshold": 2.0},
          {"spec_name": "displacement_core", "min_threshold": -0.5, "max_threshold": 0.5}
        ]
      }
    },
    
    "peripheral_flexible": {
      "description": "Peripheral regions show flexibility but not disruption",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "peripheral_rmsd", "min_threshold": 1.0, "max_threshold": 4.0},
          {"spec_name": "displacement_peripheral", "min_threshold": -2.0, "max_threshold": 2.0}
        ]
      }
    },
    
    "any_severe_disruption": {
      "description": "Any structural element shows severe disruption",
      "rule": {
        "operator": "OR",
        "operands": [
          {"spec_name": "dih_b1_integrity_phi", "min_threshold": 50.0, "max_threshold": 180.0},
          {"spec_name": "dih_b2_integrity_phi", "min_threshold": 50.0, "max_threshold": 180.0},
          {"spec_name": "displacement_b2_vs_core", "min_threshold": 2.0, "max_threshold": 10.0}
        ]
      }
    }
  }
}
```

### Hierarchical State Definitions
```json
{
  "clusters": {
    "0": {
      "description": "Native - core stable with controlled flexibility",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "core_stable"},
          {"spec_name": "peripheral_flexible"},
          {
            "operator": "NOT", 
            "operand": {"spec_name": "any_severe_disruption"}
          }
        ]
      }
    },
    
    "1": {
      "description": "Intermediate - stable core but some disruption",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "core_stable"},
          {
            "operator": "NOT",
            "operand": {"spec_name": "any_severe_disruption"}
          }
        ]
      }
    },
    
    "2": {
      "description": "Disrupted - core may be stable but severe local disruption",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "core_stable"},
          {"spec_name": "any_severe_disruption"}
        ]
      }
    },
    
    "3": {
      "description": "Unfolded - core instability",
      "rule": {
        "operator": "NOT",
        "operand": {"spec_name": "core_stable"}
      }
    }
  }
}
```

## Legacy Format Support

The system maintains backward compatibility with the original simple format:

### Legacy Format (Implicit AND):
```json
{
  "0": [
    {"spec_name": "feature1", "min_threshold": 0.0, "max_threshold": 1.0},
    {"spec_name": "feature2", "min_threshold": -1.0, "max_threshold": 1.0}
  ],
  "1": {
    "operator": "OR",
    "operands": [
      {"spec_name": "feature3", "min_threshold": 2.0, "max_threshold": 5.0},
      {"spec_name": "feature4", "min_threshold": -3.0, "max_threshold": -1.0}
    ]
  }
}
```

**Note**: This format works but lacks the composability and reusability of the transform system.

## Logging and Debugging

The script provides detailed logging of both transform evaluation and cluster assignment:

```
TRANSFORMS:
============================================================
Transform: beta_sheet_intact
  Description: Both beta strands maintain proper phi/psi angles
  Rule:
    AND (
      -20.0 <= dih_b1_integrity_phi <= 20.0
    AND
      -20.0 <= dih_b2_integrity_phi <= 20.0
    )
  Frames satisfying: 1523 (30.5%)

============================================================
CLUSTER RULES:
============================================================
Cluster 0:
  Description: Native folded state - sheets intact, helix normal
  Rule:
    AND (
      beta_sheet_intact
    AND
      NOT (
        helix_disrupted
      )
    )
  Frames satisfying: 1200 (24.0%)
  Frames assigned: 1200
  Remaining unassigned: 3800
```

### Key Log Information:
- **Transform evaluation**: Shows which transforms are satisfied by how many frames
- **Cluster rules**: Human-readable format of cluster definitions  
- **Cluster priority**: Confirmation of evaluation order
- **Per-cluster processing**: Frames satisfying rules vs. frames actually assigned
- **Final results**: Complete breakdown of cluster assignments

## Best Practices

### 1. **Design Reusable Transforms**
Create transforms for common structural concepts that can be reused:

```json
{
  "transforms": {
    "secondary_structure_intact": {"rule": ...},
    "core_displacement_minimal": {"rule": ...},
    "peripheral_flexibility_normal": {"rule": ...}
  },
  
  "clusters": {
    "0": {
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "secondary_structure_intact"},
          {"spec_name": "core_displacement_minimal"}
        ]
      }
    }
  }
}
```

### 2. **Hierarchical Design with Transforms**
Design transforms from specific to general, clusters from restrictive to permissive:

```json
{
  "transforms": {
    "perfectly_folded": {"rule": ...},      // Most restrictive
    "mostly_folded": {"rule": ...},         // Moderate
    "partially_structured": {"rule": ...}   // Least restrictive
  },
  
  "clusters": {
    "0": {"rule": {"spec_name": "perfectly_folded"}},
    "1": {"rule": {"spec_name": "mostly_folded"}}, 
    "2": {"rule": {"spec_name": "partially_structured"}}
  }
}
```

### 3. **Descriptive Naming and Documentation**
Use clear names and descriptions:

```json
{
  "transforms": {
    "b2_helix_interface_disrupted": {
      "description": "β2-α1 interface shows disruption (key unfolding signature)",
      "rule": {...}
    }
  }
}
```

### 4. **Test Incrementally**
- Start with simple transforms
- Test each transform individually by examining logging output
- Combine into increasingly complex cluster definitions
- Validate with visualization

## Migration Guide

### From Legacy to Current Format

**Before (Legacy)**:
```json
{
  "0": [
    {"spec_name": "feature1", "min_threshold": 0.0, "max_threshold": 1.0},
    {"spec_name": "feature2", "min_threshold": -1.0, "max_threshold": 1.0}
  ]
}
```

**After (Current)**:
```json
{
  "transforms": {
    "state_criteria": {
      "description": "Combined criteria for state definition",
      "rule": {
        "operator": "AND",
        "operands": [
          {"spec_name": "feature1", "min_threshold": 0.0, "max_threshold": 1.0},
          {"spec_name": "feature2", "min_threshold": -1.0, "max_threshold": 1.0}
        ]
      }
    }
  },
  
  "clusters": {
    "0": {
      "description": "Target state",
      "rule": {"spec_name": "state_criteria"}
    }
  }
}
```

## Error Handling

### Common Issues and Solutions:

**Issue**: `Transform 'xyz' not found`
**Solution**: Verify transform names in cluster rules match those defined in transforms section

**Issue**: `Feature 'xyz' not found in calculated features`  
**Solution**: Verify `spec_name` matches entries in your feature specification file

**Issue**: Transform never evaluates to True
**Solution**: Check transform logging output; may need to adjust thresholds

**Issue**: All frames assigned to Cluster -1
**Solution**: Rules may be too restrictive; start with simpler transforms and build complexity

## Output and Integration

### Generated Files:
- **Transform results**: Saved as `.npz` file with boolean arrays for each transform
- **PDB files**: Representative structures for each cluster (if `--save_pdbs` used)
- **Feature CSVs**: Detailed feature values for all frames and cluster assignments  
- **Mapping files**: Frame-to-cluster assignments with ensemble information
- **Plots**: Feature histograms colored by cluster, PCA projections
- **Logs**: Detailed transform evaluation and cluster assignment process

### Integration with Analysis Pipeline:
The enhanced rules-based clustering integrates seamlessly:
1. **Features calculated** by `analyse_LocalFeatures_PUFs.py`
2. **Transforms evaluated** from feature values  
3. **Clusters assigned** using features and transforms
4. **Results visualized** through automatic plot generation
5. **Structures saved** for further analysis

## Summary

The enhanced rules-based clustering system with composable transforms provides:
- **Modular design** through reusable transform definitions
- **Complex logic** with boolean operations on transforms and features
- **Hierarchical classification** through cluster priority and transform composition
- **Detailed logging** for both transforms and cluster assignments
- **Backward compatibility** with legacy rule formats
- **Comprehensive output** integrated with visualization and structure saving

This system enables researchers to build sophisticated conformational state definitions using modular, reusable components based on domain knowledge, providing both flexibility and maintainability for complex analysis workflows.