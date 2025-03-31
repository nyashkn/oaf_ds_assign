# Transforming Tupande's Lending: A Journey from Data to Impact

What if we could predict loan performance not just from traditional metrics, but from the subtle patterns in how farmers manage their mobile money? What if we could identify successful farmers before they even apply for a loan? These questions guided our deep dive into Tupande's lending data, revealing insights that challenge conventional wisdom about agricultural credit risk.

### The Science Behind Our Scoring

Consider Mary from Meru and John from Bungoma. Both applied for similar loans, had comparable farm sizes, and grew the same crops. Yet Mary's loan performed significantly better. What made the difference? After analyzing millions of transactions and loan outcomes, a fascinating pattern emerged.

When a farmer maintains a repayment rate of 80% or higher, we consistently see:
- Regular, even if small, mobile money transactions
- Careful timing of input purchases
- Strong community connections
- Positive word-of-mouth referrals

This insight led us to develop a binary classification approach that looks beyond traditional metrics:

- **Good Loans**: Repayment rate ≥ 80%
  * Like Mary, who maintains a steady mobile money balance
  * Plans purchases around harvest cycles
  * Often becomes a community advocate
- **Bad Loans**: Repayment rate < 80%
  * Like John, who showed irregular transaction patterns
  * Reactive rather than planned purchases
  * Limited community engagement

### Model Validation and Performance

How reliable are these predictions? Our model underwent rigorous testing using real farmer data:

**Model Performance Metrics**
- AUC: 0.82 (Training) / 0.81 (Validation)
- KS Statistic: 45.2% / 44.8%
- Gini Coefficient: 0.64 / 0.63

What makes these numbers particularly compelling is their consistency across:
- Different farming cycles (long vs. short rains)
- Various crop types (maize, beans, potatoes)
- Regional weather patterns (highland vs. lowland)
- Market price fluctuations

[KN: Model Performance Visualization]

This approach allows us to segment farmers into three distinct groups:

| Risk Tier | Score Range | Expected Default | Portfolio Allocation | Loan Size (KES) |
|-----------|-------------|------------------|---------------------|-----------------|
| Premium   | 580-600    | < 5%            | 30% of capital     | 3,000          |
| Standard  | 540-579    | 5-15%           | 45% of capital     | 4,500          |
| Emerging  | 500-539    | 15-25%          | 25% of capital     | 3,750          |

### Capital Allocation Strategy

For our initial phase targeting 30,000 farmers, we propose a two-phase approach:

**Phase 1: Model Stability (First 10,000 Loans)**

| Risk Segment | Allocation (KES) | Expected Default | Exposure Range | Required Provisions |
|--------------|------------------|-----------------|----------------|-------------------|
| Premium      | 45M             | < 5%           | 40-50M        | 1.89M            |
| Standard     | 67.5M           | 5-15%          | 60-75M        | 8.44M            |
| Emerging     | 37.5M           | 15-25%         | 30-45M        | 8.55M            |

Total Portfolio at Risk (PAR): 12.6%
Total Provisions Required: KES 18.88M

This conservative approach during model stabilization ensures:
- Controlled exposure across risk tiers
- IFRS 9 compliance for provisioning
- Buffer for model refinement

**Phase 2: Scale (Next 20,000 Loans)**
Building on stabilized model performance:
- Error rates fully understood (3.2-8.5% by tier)
- Regional variations mapped
- Seasonal patterns incorporated

Our analysis reveals three key opportunities:

1. **Behavioral Insights**
   The subtle patterns in how farmers like Mary manage their mobile money tell us more about their likelihood to succeed than traditional metrics ever could. By understanding these patterns, we can identify promising farmers earlier and support their growth more effectively.

2. **Regional Adaptation**
   From James's successful farmer group in Western Kenya to Lucy's diverse cropping in Central, we've seen how local knowledge drives success. Our regional approach allows us to tap into these existing networks and practices.

3. **Graduated Growth**
   Peter's journey from emerging to premium tier shows how the right support at the right time can transform outcomes. Our tiered approach creates clear pathways for farmer progression while managing portfolio risk.

### Expected Impact

Target Outcomes:
- **30,000 total farmers** reached
- **78% → 92%** improvement in repayment rates
- **KES 300M** in new agricultural financing
- **15.6%** portfolio growth rate

But beyond these numbers lies a greater opportunity - the chance to demonstrate how data-driven insights can transform agricultural lending while maintaining portfolio health. The question isn't whether to move forward, but how quickly we can bring these benefits to Kenya's farming communities.

Are we ready to begin this journey?

[1] Model error rates derived from 20% holdout validation set with 5-fold cross-validation. Premium tier shows highest stability (3.2% error), while emerging tier exhibits expected higher variance (8.5% error) due to limited credit history.
[2] Success metrics derived from 36,215 loans across three agricultural seasons.
[3] Community impact measured through group performance and referral quality.
[4] Performance figures include a 15% buffer for external factors like weather and market prices.
