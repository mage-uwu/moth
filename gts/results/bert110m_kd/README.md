# Abandoned: phase 3 as distillation from bert-base-uncased

`pod/bert_distill_job.sh`: resumed from phase 2 (step 108,829, validation loss 2.713) with
`--teacher google-bert/bert-base-uncased --distill-alpha 0.75 --distill-temp 2 --mask-prob 0.25 --lr 5e-4`,
on Wikipedia shards 20-39, one A100 (134K tokens/s with the teacher, against 188K without). Stopped after 17,171 of
41,347 planned steps (70 minutes) and not continued: phase 3 is plain pretraining from phase 2 instead.

- The teacher on our validation batches: loss 2.351, masked accuracy 59.0%, so only 0.36 ahead of the student.
- Validation rose to 2.96 in the re-warm and was at 2.926 at step 126,000 (37% of the leg). Phase 2 at the same point
  of its schedule was 0.08 above its start; this leg was 0.21 above its start.

With so little headroom, the soft targets carried mostly the teacher's uncertainty, and the teacher cost 29% of the
throughput. `kd_curve.log` holds the evaluations and the training loss split into KL and cross-entropy.
