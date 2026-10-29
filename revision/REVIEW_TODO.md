# npj unconventional computing revision: what each reviewer point needs

status: [ ] todo, [~] code ready / running, [x] done. paper text (section D) is edited last,
after the runs are in. full code audit: ~/Desktop/clifford_vae_fixes.txt section 7.

## editor
- [ ] results reported accurately: every table/figure regenerated from results_rev/ (revision/aggregate.py)
- [ ] overstated conclusions rewritten from the new numbers (abstract, intro bullet 3 "+29%", discussion,
      "matching or exceeding all existing methods", main.tex:475 role-filler claim)
- [ ] limitations fully explained (new limitations paragraph)
- [ ] point-by-point response letter

## reviewer 1
- [~] major 1, disentanglement / compositionality: shapes2d held-out shape-colour pairs
      (blue circle, red square, yellow triangle, green star never in training).
      N, N_L2, S, C x beta_max in {1, 4} x 10 seeds, d=128. report recon MAE seen vs unseen, gap,
      attribute accuracy of reconstructions. text: we do not claim disentanglement; HRR codes are distributed.
      also fix fig 2 traversal (sweeps dropped index 0 -> use dims 1,2).
- [~] major 2, role-filler better than unitary?: re-run with exact schlegel 3.3 protocol + references at the
      model's real item-memory dimension (2d for clifford). audit: the old "win" is entirely the 2d dimension.
      answer: matches unitary at equal dimension. bundles are l2-normalised, not unitary; we unbind with the
      role's exact inverse (role unitary), so the result is filler + crosstalk, cleaned up by nearest neighbour.
- [ ] minor: "(S^1)^n" -> (S^1)^{d-1} everywhere
- [ ] minor: sec 2.1 "enables effective bundling" -> "binding"
- [ ] minor: eq (9) duplicates eq (8) -> need the submitted pdf to see which pair

## reviewer 2
- [~] limitation 1, simple datasets: add stl-10 (64x64, d in {128, 256}); rest goes in limitations
- [~] limitation 2 / revision 2, compute cost: cost table from run logs (params, s/epoch, epochs, peak mem,
      encode latency)
- [ ] revision 1, overview figure of the clifford-vae pipeline
- [ ] limitation 3 / revision 3, downstream neuro-symbolic task: ON HOLD, ask michael.
      slot-query code exists (train.py --slot_query), off by default.

## also in the revision (from the audit / fixes file)
- [~] all vsa tests -> schlegel 3.1-3.3 exactly (bundle capacity too, no threshold)
- [~] k-NN: one function, mean codes (clifford: unitary vector), full test set, euclidean N/N_L2, cosine S/V/C
- [ ] fig 1 surface-area caption + argument (torus of radius-1/sqrt(d) circles also vanishes; argue via
      factorised kl / per-circle kappa)
- [ ] fig 6 caption panel order; eq 14 h_0 = h_m = 1 (match code); state item-memory dimension in methods
- [ ] methods: val split, per-circle kappa, kappa cap 50*dim, N_L2 kl definition, adamw wd 0.01, grad clip 1,
      fid protocol (10k prior samples vs full test set), data in [-1, 1], s/v on S^d in R^{d+1} for mnist
- [ ] mnist table 2 changes completely (gaussian kl was /d; kappa was pinned at 10)
- [ ] vmf mnist: queued LAST, only if time/budget allows
