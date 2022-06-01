import unittest

import src.rede_neural as rede_neural

class test_rede_neural(unittest.TestCase):

    def setUp(self):
        self.rede_neural = rede_neural.get_instance();

    

if __name__ == '__main__':
    unittest.main()