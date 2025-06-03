import unittest
import pandas as pd
from src.eda import descriptive_stats

class TestEDA(unittest.TestCase):
    def test_descriptive_stats(self):
        df = pd.DataFrame({
            'headline': ['Test headline', 'Another headline'],
            'publisher': ['Publisher1', 'Publisher2'],
            'stock': ['AAPL', 'MSFT']
        })
        stats = descriptive_stats(df)
        self.assertIn('headline_length_stats', stats)
        self.assertEqual(stats['articles_per_publisher']['Publisher1'], 1)

if __name__ == '__main__':
    unittest.main()