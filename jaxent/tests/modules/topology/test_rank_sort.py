# Import the class being tested
# from your_module import Partial_Topology
# For this example, assume the class is in the same file or properly imported
from jaxent.src.interfaces.topology import (
    TopologyFactory,
    rank_and_index,
)


class TestRankingAndSorting:
    """Test the ranking and sorting functionality"""

    def test_rank_order_chain_priority(self):
        """Test sorting priority by chain ID (length then value)"""
        topo_A = TopologyFactory.from_single("A", 1)
        topo_B = TopologyFactory.from_single("B", 1)
        topo_1 = TopologyFactory.from_single("1", 1)
        topo_10 = TopologyFactory.from_single("10", 1)

        # Shorter chains come first
        assert topo_A.rank_order() < topo_10.rank_order()
        assert topo_1.rank_order() < topo_10.rank_order()

        # For same length, alphanumeric sorting applies
        assert topo_1.rank_order() < topo_A.rank_order()
        assert topo_A.rank_order() < topo_B.rank_order()

    def test_rank_order_residue_position_priority(self):
        """Test sorting priority by average residue position"""
        topo_early = TopologyFactory.from_range("A", 1, 5)  # avg = 3
        topo_late = TopologyFactory.from_range("A", 10, 14)  # avg = 12

        assert topo_early.rank_order() < topo_late.rank_order()

    def test_rank_order_length_priority(self):
        """Test sorting priority by fragment length (longer first)"""
        # Same chain, same average residue position (5)
        topo_short = TopologyFactory.from_range("A", 4, 6)  # length 3
        topo_long = TopologyFactory.from_range("A", 3, 7)  # length 5

        # Longer fragment should come first (smaller rank order value)
        assert topo_long.rank_order() < topo_short.rank_order()

    def test_rank_order_with_peptide_trim(self):
        """Test rank_order respects peptide trimming"""
        # Peptide where trimming changes the average position and length
        peptide = TopologyFactory.from_range("A", 1, 10, peptide=True, peptide_trim=5)
        # Full: [1-10], avg=5.5, len=10
        # Trimmed: [6-10], avg=8, len=5

        rank_full = peptide.rank_order(check_trim=False)
        rank_trimmed = peptide.rank_order(check_trim=True)

        assert rank_full != rank_trimmed
        # Check avg residue position part of the key
        assert rank_full[2] == 5.5
        assert rank_trimmed[2] == 8.0
        # Check length part of the key
        assert rank_full[3] == -10
        assert rank_trimmed[3] == -5

    def test_direct_sorting_with_lt(self):
        """Test direct sorting of topologies using the __lt__ method"""
        topo1 = TopologyFactory.from_range("A", 10, 15)  # avg 12.5, len 6
        topo2 = TopologyFactory.from_range("A", 1, 5)  # avg 3, len 5
        topo3 = TopologyFactory.from_range("B", 1, 5)  # chain B
        topo4 = TopologyFactory.from_range("A", 1, 10)  # avg 5.5, len 10

        topologies = [topo1, topo2, topo3, topo4]
        sorted_topos = sorted(topologies)

        # Expected order based on rank_order():
        # 1. topo2 (A, avg 3, len 5)
        # 2. topo4 (A, avg 5.5, len 10)
        # 3. topo1 (A, avg 12.5, len 6)
        # 4. topo3 (B, chain B is after A)
        expected_order = [topo2, topo4, topo1, topo3]
        assert sorted_topos == expected_order

    def test_rank_and_index_method(self):
        """Test the rank_and_index class method"""
        topo_B1 = TopologyFactory.from_range("B", 1, 5, fragment_name="B1")
        topo_A2 = TopologyFactory.from_range("A", 10, 15, fragment_name="A2")
        topo_A1 = TopologyFactory.from_range("A", 1, 5, fragment_name="A1")

        topologies = [topo_B1, topo_A2, topo_A1]

        # Before ranking, indices are None
        assert all(t.fragment_index is None for t in topologies)

        ranked_topologies = rank_and_index(topologies)

        # Check order
        assert ranked_topologies[0].fragment_name == "A1"
        assert ranked_topologies[1].fragment_name == "A2"
        assert ranked_topologies[2].fragment_name == "B1"

        # Check indices
        assert ranked_topologies[0].fragment_index == 0
        assert ranked_topologies[1].fragment_index == 1
        assert ranked_topologies[2].fragment_index == 2

    def test_rank_and_index_stability(self):
        """Test that sorting is stable for equally ranked items"""
        # Create two identical topologies, but they are different objects
        topo1 = TopologyFactory.from_range("A", 1, 5, fragment_name="first")
        topo2 = TopologyFactory.from_range("B", 1, 5, fragment_name="third")
        topo3 = TopologyFactory.from_range("A", 1, 5, fragment_name="second")

        # topo1 and topo3 have identical ranking keys
        assert topo1.rank_order() == topo3.rank_order()

        topologies = [topo1, topo2, topo3]

        # Since sort is stable, topo1 should appear before topo3 in the sorted list
        ranked = rank_and_index(topologies)

        # Expected order: topo1, topo3, topo2
        assert ranked[0].fragment_name == "first"
        assert ranked[1].fragment_name == "second"
        assert ranked[2].fragment_name == "third"

        # Check indices
        assert ranked[0].fragment_index == 0
        assert ranked[1].fragment_index == 1
        assert ranked[2].fragment_index == 2
