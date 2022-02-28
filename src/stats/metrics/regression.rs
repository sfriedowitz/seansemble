use super::PVA;

#[derive(Clone, Copy, Debug)]
pub enum RegressionMetric {
    MSE,
    MAE,
    R2,
}

impl RegressionMetric {
    pub fn get_score(&self, pva: &PVA<f64>) -> f64 {
        match self {
            Self::MSE => Self::calculate_error(pva, |x| x.powf(2.0)),
            Self::MAE => Self::calculate_error(pva, |x| x.abs()),
            Self::R2 => Self::calculate_r2(pva),
        }
    }

    fn calculate_error(pva: &PVA<f64>, norm: fn(f64) -> f64) -> f64 {
        let mut err = 0.0;
        for (pred, actual) in pva {
            err += norm(pred - actual);
        }
        err / (pva.len() as f64)
    }

    fn calculate_r2(pva: &PVA<f64>) -> f64 {
        let mean = pva.actual.iter().sum::<f64>() / (pva.len() as f64);

        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for (pred, actual) in pva {
            ss_tot += (actual - mean).powf(2.0);
            ss_res += (actual - pred).powf(2.0);
        }

        1.0 - ss_res / ss_tot
    }
}
