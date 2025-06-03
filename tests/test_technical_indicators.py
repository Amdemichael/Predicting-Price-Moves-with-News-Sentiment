import unittest
import pandas as pd
from src.technical_indicators import calculate_indicators

class TestTechnicalIndicators(unittest.TestCase):
    def test_calculate_indicators(self):
        df = pd.DataFrame({'Close': [100, 101, 102, 103, 104] * 10})
        indicators = calculate_indicators(df)
        self.assertIn('SMA', indicators)
        self.assertIn('RSI', indicators)
        self.assertIn('MACD', indicators)

if __name__ == '__main__':
    unittest.main()