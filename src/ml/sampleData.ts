/**
 * High-fidelity synthetic Financial Loan dataset (financial_loan.xlsx representation)
 * Models credit risk and loan default with realistic correlated financial metrics.
 */

export interface LoanRecord {
  loan_id: string;
  loan_amount: number;
  term_months: number;
  int_rate: number;
  installment: number;
  annual_inc: number | null;
  dti: number | null;
  delinq_2yrs: number;
  home_ownership: string;
  purpose: string;
  loan_status: string; // 'Good Loan' (paid) or 'Charged Off' (default)
}

export function generateSampleFinancialLoanData(): LoanRecord[] {
  const records: LoanRecord[] = [];
  const homeOwnershipOptions = ['MORTGAGE', 'RENT', 'OWN'];
  const purposeOptions = [
    'debt_consolidation',
    'credit_card',
    'home_improvement',
    'small_business',
    'major_purchase',
  ];

  // Seeded deterministic generation for reproducible results
  let seed = 12345;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  for (let i = 1; i <= 240; i++) {
    const loan_id = `LN-${10000 + i}`;
    const term_months = rand() > 0.35 ? 36 : 60;
    const loan_amount = Math.round((5000 + rand() * 30000) / 500) * 500;
    const int_rate = Number((7.0 + rand() * 18.0).toFixed(2));

    // Approximate monthly installment
    const r = int_rate / 100 / 12;
    const n = term_months;
    const installment = Number(((loan_amount * (r * Math.pow(1 + r, n))) / (Math.pow(1 + r, n) - 1)).toFixed(2));

    // Annual income: median around 65k
    let annual_inc: number | null = Math.round((32000 + Math.pow(rand(), 1.5) * 120000) / 1000) * 1000;
    // Introduce ~3% missing values for preprocessing test
    if (i % 31 === 0) {
      annual_inc = null;
    }

    // Debt to Income (dti)
    let dti: number | null = Number((5.0 + rand() * 28.0).toFixed(2));
    if (i % 23 === 0) {
      dti = null;
    }

    // Delinquencies in past 2 years (mostly 0, some 1 or 2)
    const delinq_rand = rand();
    const delinq_2yrs = delinq_rand > 0.88 ? (delinq_rand > 0.96 ? 2 : 1) : 0;

    const home_ownership = homeOwnershipOptions[Math.floor(rand() * homeOwnershipOptions.length)];
    const purpose = purposeOptions[Math.floor(rand() * purposeOptions.length)];

    // Realistic loan default risk formula (Charged Off = 1, Good Loan = 0)
    let riskScore = 0;
    if (int_rate > 15) riskScore += 2.0;
    if (int_rate > 19) riskScore += 1.5;
    if (dti && dti > 22) riskScore += 1.8;
    if (annual_inc && annual_inc < 45000) riskScore += 1.5;
    if (delinq_2yrs > 0) riskScore += 2.2;
    if (home_ownership === 'RENT') riskScore += 0.8;
    if (purpose === 'small_business') riskScore += 1.2;
    if (term_months === 60) riskScore += 0.9;
    riskScore += (rand() - 0.5) * 3.5;

    // Default probability threshold
    const isDefault = riskScore > 4.2;
    const loan_status = isDefault ? 'Charged Off' : 'Good Loan';

    records.push({
      loan_id,
      loan_amount,
      term_months,
      int_rate,
      installment,
      annual_inc,
      dti,
      delinq_2yrs,
      home_ownership,
      purpose,
      loan_status,
    });
  }

  return records;
}
