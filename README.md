# Work3
## ISSUE1: duplicate function name inside one contract

Reason: Utilize contract_name-function_name as the key

Solution: utilize contract_name-function_name-ast_id as the key for each sample

## ISSUE2: slither do not support yul to slitherIR

solution: NA, to remove these samples

## ISSUE3: gasleft in the loop condition

for(int i; gasUse < gasleft(); i++)

solution: remove the condition part, the condition is not key for Dos

