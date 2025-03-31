# Voice AI Features for Loan Repayment Prediction

Note: Features need to be normalized based on field agent's historical call patterns. We should create a categorical variable agent_group_id by clustering agents based on their typical conversation style (% time talking, chitchat level, etc.).

## 1. Call Metadata Features

'call_duration' - Length of each interaction in minutes. Extracted directly from call logs. [POTENTIALLY_STRONG]

'call_frequency' - Number of calls per month with the farmer. Calculated from call timestamps. [POTENTIALLY_STRONG]

'intercall_interval' - Time difference between consecutive calls. Measures consistency of communication. [POTENTIALLY_STRONG]

'time_of_day' - When calls typically occur (morning/afternoon/evening). [POTENTIALLY_WEAK]

'call_initiation_pattern' - Whether calls are typically initiated by farmer or agent. [POTENTIALLY_STRONG]

'missed_call_ratio' - Proportion of attempted calls that were missed/unanswered. [POTENTIALLY_STRONG]

## 2. Voice-Based Features

'vocal_stress_level' - Normalized score (0-1) measuring stress markers in voice during financial discussions. [POTENTIALLY_STRONG]

'emotional_congruence' - Alignment between spoken content and emotional tone. [POTENTIALLY_WEAK]

'conversation_engagement' - Active participation metrics through response latency and interaction patterns. [POTENTIALLY_STRONG]

'voice_confidence_score' - Measures confidence levels in voice during financial commitments. [POTENTIALLY_WEAK]

## 3. Conversation Content Features

'topic_breadth_score' - Number and variety of unique topics discussed across calls. [POTENTIALLY_STRONG]

'questions_asked_ratio' - Number of questions asked by caller vs. total conversation time. [POTENTIALLY_STRONG]

'question_answer_completion' - Proportion of questions that received complete answers. [POTENTIALLY_STRONG]

'financial_topic_ratio' - Time spent discussing financial matters vs. other topics. [POTENTIALLY_STRONG]

'topic_transition_score' - How conversations flow between subjects. [POTENTIALLY_WEAK]

'interruption_frequency' - Instances of talking over during financial discussions. [POTENTIALLY_WEAK]

'hesitation_markers' - Frequency of filled pauses during financial topics. [POTENTIALLY_STRONG]

'proactive_planning_mentions' - References to future financial planning without prompting. [POTENTIALLY_STRONG]

## 4. Implementation Options Comparison

| Component | OpenAI Stack | Google Cloud Stack | Hugging Face Stack |
|-----------|--------------|-------------------|-------------------|
| **Core Models** | - Whisper Large v3<br>- GPT-4 for analysis | - Speech-to-Text<br>- Cloud Natural Language | - Distil-Whisper<br>- BERT/RoBERTa |
| **Features** | - 99+ languages<br>- Built-in diarization<br>- Highest accuracy | - 125+ languages<br>- Speaker labeling<br>- Real-time processing | - Open source<br>- Customizable<br>- Lower latency |
| **Processing** | - Cloud-based<br>- Managed scaling | - Serverless<br>- Auto-scaling | - Self-hosted option<br>- Custom deployment |
| **Base Costs** | - $0.006/min audio<br>- $0.03/1K tokens | - $0.004/15 sec<br>- $0.001/1K chars | - Free (self-hosted)<br>- $0.0001-0.0012/sec (API) |
| **Scale Costs** | For 36K farmers scenario:<br>(10% × 4 calls + 30% × 2 calls + 20% × 1 call) × 5 min avg<br>= 216,000 minutes total<br>= $1,296 audio + $972 analysis<br>= **$2,268 total** | Same scenario:<br>= 216,000 minutes<br>= $3,456 audio + $324 analysis<br>= **$3,780 total** | Same scenario:<br>- Hosted: $1,620 total<br>- Self-hosted: $800/mo infrastructure<br>= **$1,620 or $800/mo** |
| **Advantages** | - Highest accuracy<br>- Simple integration<br>- Regular updates | - Enterprise-grade<br>- Integrated suite<br>- Good documentation | - Full control<br>- One-time costs<br>- Customizable |
| **Challenges** | - Higher costs<br>- Less flexible<br>- Black box models | - Complex pricing<br>- Regional restrictions<br>- Setup overhead | - Requires maintenance<br>- More development<br>- Limited support |
| **Setup Time** | 1-2 weeks | 2-3 weeks | 4-6 weeks |

## 5. Cost-Benefit Analysis

Current Portfolio Statistics (Training Sample of ~37k):
- Total Portfolio Value: KES 145,898,665 ($1,122,297)
- Total Loans: 36,215
- Average Loan: KES 4,029 ($31)
- Current Repayment Rate: 78.6%
- Target Repayment Rate: 98%

Implementation Cost Impact:
- Annual Voice AI Cost (OpenAI): $2,268
- Cost per Loan: $0.063 (0.2% of average loan value)
- Cost per Portfolio Value: 0.20%
