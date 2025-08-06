# Topology Interface: Summary of Classes and Methods


usage:

import jaxent.src.interfaces.topology as pt

topology = pt.Partial_Topology(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )


---

## `/core.py`

### **Class:** `Partial_Topology`
Represents a fragment of a biophysical topology (e.g., protein segment).

**Attributes:**
- `chain`
- `residues`
- `fragment_sequence`
- `fragment_name`
- `fragment_index`
- `peptide`
- `peptide_trim`
- **Computed properties:** `residue_start`, `residue_end`, `length`, `is_contiguous`, `peptide_residues`

**Methods:**
- `__post_init__`: Computes derived properties after initialization.
- `_set_peptide(val, trim)`: Updates peptide settings and recomputes peptide residues.
- `_get_active_residues(check_trim)`: Returns active residues, optionally trimmed for peptides.
- `get_residue_ranges()`: Returns contiguous ranges within the residue list.
- `_to_dict()`: Serializes the object to a dictionary.
- `_from_dict(data)`: Creates an object from a dictionary.
- `__str__`: Pretty string representation.
- `__repr__`: Detailed representation.
- `__lt__(other)`: Comparison for sorting.
- `__hash__`: Hash for set operations.
- `_get_chain_score()`: Generates a score for chain ID for sorting.
- `rank_order(check_trim)`: Generates a sort key for ranking topologies.
- `contains_all_residues(residues, check_trim)`: Checks if all specified residues are present.
- `contains_any_residues(residues, check_trim)`: Checks if any specified residues are present.
- `contains_which_residue(residues, check_trim)`: Checks which specified residues are present.

---

## `/factory.py`

### **Class:** `TopologyFactory`
Provides static methods for creating and manipulating `Partial_Topology` objects.

**Methods:**
- `from_range(chain, start, end, ...)`: Creates topology from a contiguous range of residues.
- `from_residues(chain, residues, ...)`: Creates topology from a list of residues.
- `from_single(chain, residue, ...)`: Creates topology from a single residue.
- `merge(topologies, ...)`: Merges multiple topologies into one (union or intersection).
- `merge_contiguous(topologies, ...)`: Merges topologies that are contiguous or nearly contiguous.
- `merge_overlapping(topologies, ...)`: Merges topologies with overlapping residues.
- `extract_residues(topology, use_peptide_trim)`: Extracts individual residue topologies.
- `remove_residues_by_topologies(topology, topologies_to_remove, ...)`: Removes residues from a topology based on others.
- `union(top, other, ...)`: Combines two topologies into one containing all residues.

---

## `/pairwise.py`

### **Class:** `PairwiseTopologyComparisons`
Provides static methods for pairwise comparison of `Partial_Topology` objects.

**Methods:**
- `intersects(top, other, check_trim)`: Checks if two topologies overlap.
- `contains_topology(top, other, check_trim)`: Checks if one topology contains another.
- `is_subset_of(top, other, check_trim)`: Checks if one topology is a subset of another.
- `is_superset_of(top, other, check_trim)`: Checks if one topology is a superset of another.
- `get_overlap(top, other, check_trim)`: Returns overlapping residues.
- `get_difference(top, other, check_trim)`: Returns residues in one topology not in the other.
- `is_adjacent_to(top, other, check_trim)`: Checks if two topologies are adjacent.
- `get_gap_to(top, other, check_trim)`: Returns the gap size between two topologies.

---

## `/mda_adapter.py`

### **Class:** `mda_TopologyAdapter`
Adapter for converting between MDAnalysis Universe objects and `Partial_Topology` objects.

**Refactoring Summary:**
This class has been extensively refactored to eliminate code duplication and improve maintainability.

**Key Improvements:**
1.  **New Shared Utilities** (eliminated ~300+ lines of duplicated code):
    *   `_get_chain_id()`: Standardized chain ID extraction from atoms/residues
    *   `_extract_sequence()`: Centralized amino acid sequence extraction
    *   `_mda_group_to_topology()`: Unified MDA group to Partial_Topology conversion
    *   `_create_mda_group_lookup_key()`: Standardized lookup key generation
    *   `_build_residue_selection_string()`: Centralized selection string building
    *   `_normalize_parameters()`: Unified parameter normalization for ensembles
    *   `_apply_selection_pipeline()`: Consolidated selection and chain grouping logic
    *   `_process_chain_residues()`: Centralized chain processing with terminal exclusion
    *   `_create_topology_from_residues()`: Unified topology creation from residues

2.  **Refactored Methods**:
    *   `get_atomgroup_reordering_indices()`: Reduced from ~150 lines to ~40 lines
    *   `from_mda_universe()`: Reduced by ~50% using shared utilities
    *   `to_mda_group()`: Simplified with shared pipeline
    *   `find_common_residues()`: Now uses parameter normalization
    *   All validation and sorting methods now use shared utilities

3.  **Benefits Achieved**:
    *   Eliminated ~300+ lines of duplicated code
    *   Consistent behavior across all methods
    *   Better testability with isolated utility functions
    *   Easier maintenance - changes only need to be made once
    *   Clearer separation of concerns
    *   Full backward compatibility maintained

**Methods:**
- `get_mda_group_sort_key(group)`: Public method to generate a sort key for an MDAnalysis group that matches Partial_Topology ranking.
- `from_mda_universe(universe, ...)`: Extracts `Partial_Topology` objects from an MDAnalysis Universe.
- `to_mda_group(topologies, universe, ...)`: Converts `Partial_Topology` objects to MDAnalysis groups.
- `find_common_residues(ensemble, ...)`: Finds common residues across an ensemble of MDAnalysis Universe objects.
- `get_residuegroup_ranking_indices(residue_group)`: Get indices to reorder individual residues in a ResidueGroup/AtomGroup by topology ranking.
- `get_atomgroup_reordering_indices(mda_groups, universe, ...)`: Get indices to reorder a list of MDAnalysis groups to match topology order.
- `partial_topology_pairwise_distances(topologies, universe, ...)`: Compute trajectory-averaged pairwise center-of-mass distances between Partial_Topology objects.
- `to_mda_residue_dict(topologies, universe, ...)`: Extract residue information as a dictionary of chain:[residue_indices].

---

## `/serialise.py`

### **Class:** `PTSerialiser`
Handles serialization and deserialization of `Partial_Topology` objects.

**Methods:**
- `to_json(top)`: Serializes a topology to a JSON string.
- `from_json(json_str)`: Deserializes a topology from a JSON string.
- `save_list_to_json(topologies, filepath)`: Saves a list of topologies to a JSON file.
- `load_list_from_json(filepath)`: Loads a list of topologies from a JSON file.

---

## `/utils.py`

### **Functions:**
- `rank_and_index(topologies, check_trim)`: Sorts and assigns `fragment_index` to topologies.
- `calculate_fragment_redundancy(topologies, mode, check_trim)`: Calculates overlap redundancy between fragments.
- `group_set_by_chain(topologies)`: Groups a set of topologies by chain.

---

**Note:**  
All classes and methods are designed to work together for flexible, robust handling of protein topology fragments, including creation, comparison, merging, serialization, and conversion to/from MDAnalysis objects.