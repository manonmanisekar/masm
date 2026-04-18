import unittest
from masm.core.clock import VectorClock

class TestVectorClock(unittest.TestCase):
    def test_initialization(self):
        vc = VectorClock()
        self.assertEqual(vc.clock, {}, "Clock should be empty on initialization.")

    def test_increment(self):
        vc = VectorClock()
        vc.increment("agent1")
        self.assertEqual(vc.clock, {"agent1": 1}, "Clock should increment agent1's timestamp.")
        vc.increment("agent1")
        self.assertEqual(vc.clock, {"agent1": 2}, "Clock should increment agent1's timestamp again.")
        vc.increment("agent2")
        self.assertEqual(vc.clock, {"agent1": 2, "agent2": 1}, "Clock should increment agent2's timestamp.")

    def test_merge(self):
        vc = VectorClock()
        vc.increment("agent1")
        vc.increment("agent2")
        other_clock = {"agent1": 3, "agent3": 2}
        vc.merge(other_clock)
        self.assertEqual(vc.clock, {"agent1": 3, "agent2": 1, "agent3": 2}, "Merge should take the max of each agent's timestamp.")

    def test_happens_before(self):
        vc = VectorClock()
        a = {"agent1": 1, "agent2": 2}
        b = {"agent1": 2, "agent2": 3}
        self.assertTrue(vc.happens_before(a, b), "Clock A should causally precede Clock B.")
        self.assertFalse(vc.happens_before(b, a), "Clock B should not causally precede Clock A.")
        self.assertFalse(vc.happens_before(a, a), "Clock A should not causally precede itself.")

    def test_concurrent(self):
        vc = VectorClock()
        a = {"agent1": 1, "agent2": 3}
        b = {"agent1": 2, "agent2": 2}
        self.assertTrue(vc.concurrent(a, b), "Clocks A and B should be concurrent.")
        self.assertFalse(vc.concurrent(a, a), "A clock should not be concurrent with itself.")

    def test_reset(self):
        vc = VectorClock()
        vc.increment("agent1")
        vc.reset()
        self.assertEqual(vc.clock, {}, "Clock should be empty after reset.")

    def test_equals(self):
        vc = VectorClock()
        a = {"agent1": 1, "agent2": 2}
        b = {"agent1": 1, "agent2": 2}
        c = {"agent1": 2, "agent2": 1}
        self.assertTrue(vc.equals(a, b), "Clocks A and B should be equal.")
        self.assertFalse(vc.equals(a, c), "Clocks A and C should not be equal.")
    def test_difference(self):
        vc = VectorClock()
        a = {"agent1": 3, "agent2": 2}
        b = {"agent1": 1, "agent2": 3}
        result = vc.difference(a, b)
        self.assertEqual(result, {"agent1": 2, "agent2": -1}, "Difference calculation is incorrect.")

    def test_max_clock(self):
        vc = VectorClock()
        vc.increment("agent1")
        vc.increment("agent2")
        vc.increment("agent2")
        self.assertEqual(vc.max_clock(), "agent2", "Max clock calculation is incorrect.")

    def test_from_dict(self):
        vc = VectorClock()
        vc.from_dict({"agent1": 3, "agent2": 2})
        self.assertEqual(vc.clock, {"agent1": 3, "agent2": 2}, "from_dict method is incorrect.")

    def test_to_dict(self):
        vc = VectorClock()
        vc.increment("agent1")
        self.assertEqual(vc.to_dict(), {"agent1": 1}, "to_dict method is incorrect.")

    def test_compare(self):
        vc = VectorClock()
        a = {"agent1": 1, "agent2": 2}
        b = {"agent1": 2, "agent2": 3}
        c = {"agent1": 1, "agent2": 2}
        self.assertEqual(vc.compare(a, b), "happens-before", "Comparison is incorrect.")
        self.assertEqual(vc.compare(b, a), "happens-after", "Comparison is incorrect.")
        self.assertEqual(vc.compare(a, c), "equal", "Comparison is incorrect.")
        self.assertEqual(vc.compare(a, {"agent1": 2, "agent2": 1}), "concurrent", "Comparison is incorrect.")
        
    
if __name__ == "__main__":
    unittest.main()