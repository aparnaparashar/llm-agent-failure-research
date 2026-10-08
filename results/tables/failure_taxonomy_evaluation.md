# Comprehensive Failure Taxonomy Evaluation (16 Cognitive Types + 5 Temporal Dynamics)

Evaluates detection performance and the primary telemetry feature drivers for each failure mode across the unified taxonomy.

| Category           | Module     | Failure_Mode               |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall | Top_Telemetry_Drivers                                                                | Primary_Feature         |
|:-------------------|:-----------|:---------------------------|----------:|---------:|-----------:|------------:|---------:|:-------------------------------------------------------------------------------------|:------------------------|
| Cognitive Failure  | Reflection | hallucination              |     0.933 |    0.896 |      0.825 |       0.853 |    0.8   | lexical_diversity (35%), token_expansion_ratio (25%), latest_message_len (20%)       | lexical_diversity       |
| Cognitive Failure  | Reflection | causal_misattribution      |     0.929 |    0.893 |      0.822 |       0.849 |    0.8   | repeat_tool_ratio (30%), step_ratio (25%), tool_calls_count (25%)                    | repeat_tool_ratio       |
| Cognitive Failure  | Reflection | outcome_misinterpretation  |     0.933 |    0.897 |      0.826 |       0.853 |    0.8   | observation_error_flag (35%), lexical_diversity (25%), step_ratio (20%)              | observation_error_flag  |
| Cognitive Failure  | Reflection | progress_misassessment     |     0.938 |    0.901 |      0.83  |       0.858 |    0.804 | step_ratio (35%), cum_latency (25%), token_expansion_ratio (20%)                     | step_ratio              |
| Cognitive Failure  | Action     | parameter_error            |     0.969 |    0.931 |      0.857 |       0.886 |    0.83  | tool_error_rate (35%), observation_error_flag (30%), consecutive_tool_errors (20%)   | tool_error_rate         |
| Cognitive Failure  | Action     | format_error               |     0.969 |    0.931 |      0.857 |       0.886 |    0.83  | observation_error_flag (35%), step_latency (25%), token_expansion_ratio (20%)        | observation_error_flag  |
| Cognitive Failure  | Action     | planning_action_disconnect |     0.978 |    0.94  |      0.866 |       0.895 |    0.839 | lexical_diversity (30%), repeat_tool_ratio (25%), token_expansion_ratio (25%)        | lexical_diversity       |
| Cognitive Failure  | Planning   | constraint_ignorance       |     0.954 |    0.916 |      0.844 |       0.872 |    0.818 | observation_error_flag (30%), step_ratio (25%), tool_error_count (25%)               | observation_error_flag  |
| Cognitive Failure  | Planning   | impossible_action          |     0.948 |    0.91  |      0.838 |       0.866 |    0.812 | consecutive_tool_errors (35%), tool_error_rate (25%), cum_latency (20%)              | consecutive_tool_errors |
| Cognitive Failure  | Planning   | inefficient_planning       |     0.953 |    0.915 |      0.843 |       0.871 |    0.817 | repeat_tool_ratio (35%), step_ratio (25%), tool_calls_count (25%)                    | repeat_tool_ratio       |
| Cognitive Failure  | Memory     | incomplete_summary         |     0.943 |    0.905 |      0.834 |       0.862 |    0.807 | token_expansion_ratio (35%), avg_message_len (25%), latest_message_len (25%)         | token_expansion_ratio   |
| Cognitive Failure  | Memory     | false_memory_hallucination |     0.943 |    0.905 |      0.834 |       0.862 |    0.807 | lexical_diversity (35%), token_expansion_ratio (25%), latest_message_len (25%)       | lexical_diversity       |
| Cognitive Failure  | Memory     | retrieval_failure          |     0.946 |    0.909 |      0.837 |       0.865 |    0.811 | repeat_tool_ratio (35%), tool_calls_count (25%), step_ratio (20%)                    | repeat_tool_ratio       |
| Cognitive Failure  | System     | environment_error          |     0.976 |    0.937 |      0.862 |       0.891 |    0.834 | tool_error_count (35%), step_latency (30%), observation_error_flag (20%)             | tool_error_count        |
| Cognitive Failure  | System     | tool_execution_error       |     0.976 |    0.938 |      0.863 |       0.892 |    0.835 | tool_error_rate (35%), tool_error_count (30%), observation_error_flag (20%)          | tool_error_rate         |
| Cognitive Failure  | System     | step_limit_exhaustion      |     0.982 |    0.943 |      0.869 |       0.898 |    0.841 | step_ratio (45%), tool_calls_count (25%), cum_latency (20%)                          | step_ratio              |
| Temporal Mechanism | Temporal   | goal_drift                 |     0.935 |    0.898 |      0.826 |       0.854 |    0.8   | lexical_diversity (35%), token_expansion_ratio (25%), avg_message_len (20%)          | lexical_diversity       |
| Temporal Mechanism | Temporal   | looping                    |     0.977 |    0.938 |      0.864 |       0.893 |    0.837 | repeat_tool_ratio (40%), consecutive_tool_errors (25%), repetition_ngram_score (20%) | repeat_tool_ratio       |
| Temporal Mechanism | Temporal   | tool_cascade               |     0.976 |    0.937 |      0.862 |       0.891 |    0.835 | consecutive_tool_errors (35%), tool_error_rate (30%), step_latency (20%)             | consecutive_tool_errors |
| Temporal Mechanism | Temporal   | grounding_loss             |     0.928 |    0.891 |      0.82  |       0.848 |    0.8   | lexical_diversity (35%), observation_error_flag (25%), token_expansion_ratio (20%)   | lexical_diversity       |
| Temporal Mechanism | Temporal   | context_corruption         |     0.967 |    0.93  |      0.857 |       0.885 |    0.83  | token_expansion_ratio (35%), avg_message_len (25%), latest_message_len (25%)         | token_expansion_ratio   |

### Global Telemetry Feature Importance (XGBoost)

| Feature Name | Importance Weight |
|---|---|
| **avg_message_len** | 0.1657 |
| **tool_error_rate** | 0.1612 |
| **lexical_diversity** | 0.1486 |
| **repeat_tool_ratio** | 0.1254 |
| **observation_error_flag** | 0.1128 |
| **tool_calls_count** | 0.0874 |
| **latest_message_len** | 0.0800 |
| **step_ratio** | 0.0518 |
| **tool_error_count** | 0.0221 |
| **consecutive_tool_errors** | 0.0176 |
| **token_expansion_ratio** | 0.0121 |
| **cum_latency** | 0.0078 |
| **step_latency** | 0.0075 |
| **repetition_ngram_score** | 0.0000 |
