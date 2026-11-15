import unittest
from twins import TWINS, run

class TwinTests(unittest.TestCase):
    def test_all_five_finite_and_deterministic(self):
        for name in TWINS:
            with self.subTest(name=name):
                self.assertEqual(run(name),run(name))
                self.assertFalse(run(name)["verified"])
    def test_fpso_mass(self):
        self.assertAlmostEqual(run("fpso")["results"]["mass_balance_residual_kg_s"],0)
    def test_furnace_efficiency(self):
        self.assertLess(run("blast_furnace")["results"]["useful_heat_w"],
                        run("blast_furnace")["results"]["fuel_input_w"])
    def test_chemical_conversion(self):
        self.assertTrue(0<run("chemical_plant")["results"]["conversion"]<1)
    def test_rocket_thrust(self):
        low=run("reusable_rocket",{"thrust_n":200000})["results"]["vertical_acceleration_m_s2"]
        high=run("reusable_rocket",{"thrust_n":300000})["results"]["vertical_acceleration_m_s2"]
        self.assertGreater(high,low)
    def test_lunar_night(self):
        self.assertEqual(run("lunar_base",{"illumination_fraction":0})["results"]["solar_power_w"],0)
    def test_bad_efficiency(self):
        with self.assertRaises(ValueError):
            run("blast_furnace",{"thermal_efficiency":1.1})
    def test_unknown_twin(self):
        with self.assertRaises(KeyError):
            run("not_found")
if __name__=="__main__":unittest.main()
