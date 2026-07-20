# 📓 Preliminary Results & Research Log — Reflex Diffing in Phi-4-mini
This is a living progress log of this project and my findings so far.

## Summary
I implemented the generation, and grading code and got started on the sentence level annotation.
Since I used a wrong parameter, I circled back to reproducing the baseline.

## Log

July 20th:
    **Baseline** 
    - Question: How can I quickly and cheaply generate the rollouts necessary for this project?
        - Decision: Implement the generation using modal serverless H100 GPUs for this batch workload.
        - -> quick and easy generation (< 5min) for free within free-tier limits
    - Decision: Use the same generation parameters as in the technical report for consistency.    
    - Implemented grading using math-verify, as in the Phi-4-mini reasoning technical report.
    - Graded rollouts -> consistent with paper results for the reasoning model 
    - Risk: Suspected that  
        - tracked parsing failures and regraded -> not a problem
    - Read paper again, saw parameter mismatch (accidentally used higher temperature)
        - -> running again with correct parameters.


    

