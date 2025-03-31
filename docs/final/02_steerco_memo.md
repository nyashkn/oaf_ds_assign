# Enhancing Tupande's Credit Risk Management: A Data-Driven Approach

**To:** Steering Committee  
**From:** Data Science Team  
**Date:** March 31, 2025  
**Subject:** Credit Risk Model Findings and Strategic Recommendations

## Executive Summary

Our analysis reveals significant opportunities to enhance Tupande's lending operations through data-driven insights and strategic process improvements. The key to success lies in balancing portfolio growth with risk management through:

1. Intelligent customer segmentation
2. Risk-based deposit structures
3. Milestone-based disbursements
4. Behavioral incentive systems

By implementing these recommendations, we project improving the current 78.6% repayment rate toward the 98% target while optimizing customer acquisition costs.

## Current Context

Tupande's lending operations show both strengths and areas for improvement:

**Portfolio Metrics**
- Total Value: KES 145.9M ($1.12M)
- Active Loans: 36,215
- Average Loan: KES 4,029 ($31)
- Current Repayment Rate: 78.6%
- Target Repayment Rate: 98%

**Operational Challenges**
- High customer acquisition costs
- Limited risk differentiation
- Uniform deposit requirements
- Binary accept/reject decisions

## Key Findings

### 1. Behavioral Patterns Matter

Our analysis revealed significant temporal patterns in loan performance:

**Day-of-Week Effect**
- Sunday contracts show notably lower cure rates
- Potential factors:
  * Different borrower profiles (urgency vs. planning)
  * Reduced operational oversight
  * Limited market price information
  * Post-weekend financial pressures

[Figure 1: Cure Rate by Contract Day]

**Seasonal Patterns**
- September cohorts: 90.2% cure rate
- November cohorts: 70.1% cure rate
- Strong correlation with agricultural cycles

### 2. Risk-Reward Trade-off

The analysis reveals a critical balance between portfolio growth and risk management:

**Current Approach** (Binary Accept/Reject)
- Rejects potentially good borrowers
- Misses opportunity for graduated lending
- Limited learning from portfolio performance

**Proposed Approach** (Graduated Risk Management)
```
Total Addressable Market
└── Tupande TAM (75%)
    ├── Rejected (30%)
    │   ├── No Prior Loan (15%) → Direct Input Sales
    │   ├── Prior Default (8%) → Remediation Program
    │   └── False Positives (7%) → Reassessment
    ├── Eligible (45%)
    │   ├── No Contract (25%) → Outbound Marketing
    │   └── Has Contract (20%) → Conversion Focus
    └── Contracted (25%)
        ├── On Track (20%) → Retention Program
        └── Outstanding (5%) → Early Intervention
```

[Figure 2: Customer Journey Tree Map]

### 3. Deposit Impact Analysis

Our analysis shows deposit requirements significantly influence both access and performance:

**Current State**
- Uniform deposit requirements
- High upfront burden
- Limited flexibility

**Proposed Structure**
- Risk-based deposits
- Early deposit incentives
- Fractional deposit options

[Figure 3: Deposit-Performance Correlation]

### 4. Predictive Indicators

Key predictive features by category and weight:

**Historical Metrics (40% total weight)**
- Regional loan volumes (15.2%)
- Cumulative deposits (12.8%)
- Historical value trends (11.5%)

**Relative Position (30% total weight)**
- Deposit ratio rankings (10.3%)
- Contract value benchmarks (9.7%)
- Historical performance (8.9%)

**Infrastructure (15% total weight)**
- Duka distribution (8.1%)
- Customer density (7.6%)

**Temporal (15% total weight)**
- Seasonal patterns (6.4%)
- Contract timing (5.8%)

[Figure 4: Feature Importance Distribution]

## Strategic Recommendations

### 1. Dynamic Deposit Structure

Implement a flexible deposit system that balances risk and accessibility:

**Base Structure**
| Risk Level | Standard | Early Deposit | Fractional |
|------------|----------|---------------|------------|
| Low        | 20%      | 15%          | 3 × 7%     |
| Medium     | 30%      | 25%          | 3 × 10%    |
| High       | 40%      | 35%          | 3 × 14%    |

**Rationale**
- Deposit = Total Inputs + CAC - Maximum Exposure
- Early deposit incentives reduce risk
- Fractional deposits improve accessibility
- Aligned with farming cycles

### 2. Tranched Disbursement System

Implement milestone-based disbursement aligned with farming cycles:

**Structure**
1. Land Preparation (30%)
   - Basic inputs
   - Soil preparation
   - Initial assessment

2. Planting Phase (40%)
   - Seeds and fertilizer
   - Technical support
   - Progress verification

3. Maintenance Phase (30%)
   - Top dressing
   - Pest control
   - Harvest preparation

**Risk Mitigation**
- Book total exposure at contract
- Track tranche drawdown
- Milestone verification
- Early warning system

### 3. Behavioral Rating System

Implement a points-based system to drive positive behavior:

**Point Sources**
- Timely repayments (+10 points)
- Early deposits (+15 points)
- Store purchases (+5 points)
- Successful referrals (+20 points)
- Training completion (+10 points)

**Benefits**
- Reduced interest rates
- Priority processing
- Flexible deposit terms
- Exclusive products
- Training access

## Implementation Plan

### Phase 1: Foundation (30 Days)
1. Finalize scoring model
2. Design pilot program
3. Select test regions
4. Prepare training materials

### Phase 2: Pilot (90 Days)
1. Train field staff
2. Launch in test regions
3. Monitor performance
4. Gather feedback

### Phase 3: Refinement (180 Days)
1. Analyze pilot results
2. Refine processes
3. Plan full rollout
4. Scale infrastructure

## Financial Impact

**Investment Required**
- System Development: $XX,XXX
- Training Programs: $XX,XXX
- Monitoring Tools: $XX,XXX

**Expected Returns (Year 1)**
- Improved Repayment: +11.4%
- Reduced Defaults: -8.2%
- Portfolio Growth: +15.6%
- Net ROI: 127%

## Risk Management

### 1. Adverse Selection
- Comprehensive scoring model
- Graduated exposure limits
- Behavioral monitoring

### 2. Moral Hazard
- Clear incentive structure
- Community engagement
- Regular monitoring

### 3. Operational Risk
- Staff training
- Process automation
- Quality controls

## Next Steps

1. **Immediate (Week 1)**
   - Form implementation team
   - Finalize pilot regions
   - Begin staff training

2. **Short-term (Month 1)**
   - Launch pilot program
   - Establish monitoring
   - Collect initial data

3. **Medium-term (Month 3)**
   - Evaluate results
   - Refine processes
   - Plan expansion

We recommend proceeding with the pilot program within the next 30 days to capture data during the upcoming planting season. The proposed changes represent a balanced approach to growth and risk management, with clear metrics for success evaluation.
**Structure**
1. Land Preparation (30%)
   - Basic inputs
   - Soil preparation
   - Initial assessment

2. Planting Phase (40%)
   - Seeds and fertilizer
   - Technical support
   - Progress verification

3. Maintenance Phase (30%)
   - Top dressing
   - Pest control
   - Harvest preparation

**Risk Mitigation**
- Book total exposure at contract
- Track tranche drawdown
- Milestone verification
- Early warning system

### 3. Behavioral Rating System

Implement a points-based system to drive positive behavior:

**Point Sources**
- Timely repayments (+10 points)
- Early deposits (+15 points)
- Store purchases (+5 points)
- Successful referrals (+20 points)
- Training completion (+10 points)

**Benefits**
- Reduced interest rates
- Priority processing
- Flexible deposit terms
- Exclusive products
- Training access

## Implementation Plan

### Phase 1: Foundation (30 Days)
1. Finalize scoring model
2. Design pilot program
3. Select test regions
4. Prepare training materials

### Phase 2: Pilot (90 Days)
1. Train field staff
2. Launch in test regions
3. Monitor performance
4. Gather feedback

### Phase 3: Refinement (180 Days)
1. Analyze pilot results
2. Refine processes
3. Plan full rollout
4. Scale infrastructure

## Financial Impact

**Investment Required**
- System Development: $XX,XXX
- Training Programs: $XX,XXX
- Monitoring Tools: $XX,XXX

**Expected Returns (Year 1)**
- Improved Repayment: +11.4%
- Reduced Defaults: -8.2%
- Portfolio Growth: +15.6%
- Net ROI: 127%

## Risk Management

### 1. Adverse Selection
- Comprehensive scoring model
- Graduated exposure limits
- Behavioral monitoring

### 2. Moral Hazard
- Clear incentive structure
- Community engagement
- Regular monitoring

### 3. Operational Risk
- Staff training
- Process automation
- Quality controls

## Next Steps

1. **Immediate (Week 1)**
   - Form implementation team
   - Finalize pilot regions
   - Begin staff training

2. **Short-term (Month 1)**
   - Launch pilot program
   - Establish monitoring
   - Collect initial data

3. **Medium-term (Month 3)**
   - Evaluate results
   - Refine processes
   - Plan expansion

We recommend proceeding with the pilot program within the next 30 days to capture data during the upcoming planting season. The proposed changes represent a balanced approach to growth and risk management, with clear metrics for success evaluation.