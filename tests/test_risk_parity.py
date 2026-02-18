import unittest


from infrastructure.mt5.risk_parity import (
    covariance_to_correlation,
    ewma_covariance,
    ewma_lambda_from_half_life,
    risk_parity_weights,
)


class TestRiskParity(unittest.TestCase):
    def test_ewma_lambda_from_half_life(self) -> None:
        self.assertAlmostEqual(ewma_lambda_from_half_life(1), 0.5)

    def test_covariance_to_correlation_identity(self) -> None:
        cov = [
            [1.0, 0.0],
            [0.0, 4.0],
        ]
        corr = covariance_to_correlation(cov)
        self.assertAlmostEqual(corr[0][0], 1.0)
        self.assertAlmostEqual(corr[1][1], 1.0)
        self.assertAlmostEqual(corr[0][1], 0.0)
        self.assertAlmostEqual(corr[1][0], 0.0)

    def test_ewma_covariance_ridge_makes_diagonal_positive(self) -> None:
        # Constant returns -> covariance would be 0 without ridge.
        returns = [[0.0, 0.0] for _ in range(10)]
        cov = ewma_covariance(returns, half_life_bars=5, ridge=1e-6)
        self.assertGreater(cov[0][0], 0.0)
        self.assertGreater(cov[1][1], 0.0)

    def test_risk_parity_weights_identity_cov_is_uniform(self) -> None:
        cov = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        w = risk_parity_weights(cov, max_iter=2000, tol=1e-10)
        self.assertAlmostEqual(sum(w), 1.0)
        for wi in w:
            self.assertAlmostEqual(wi, 1.0 / 3.0, places=6)

    def test_risk_parity_weights_diagonal_cov_matches_inverse_vol(self) -> None:
        # For diagonal cov, ERC weights are proportional to 1/sigma.
        # var=[0.04, 0.01] -> sigma=[0.2,0.1] -> 1/sigma=[5,10] -> w=[1/3,2/3]
        cov = [
            [0.04, 0.0],
            [0.0, 0.01],
        ]
        w = risk_parity_weights(cov, max_iter=5000, tol=1e-10)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertAlmostEqual(w[0], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(w[1], 2.0 / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
