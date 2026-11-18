# Release 2: EXPLORATORY analyses (not preregistered; no multiplicity correction)

## B. B-run fragility: clean B already answers A (top-1), by family and corrupt step (E3, all candidates in f_none)
        task      family   n  B_says_A  B_says_B  in_f_pilot
   add_chain  pilot_calc  51     0.000     0.980       0.882
   add_chain pilot_story 164     0.006     0.970       0.640
   add_chain     rs_plus 213     0.000     0.972       0.704
   add_chain    rs_story 203     0.039     0.911       0.739
   add_chain rs_symbolic 157     0.000     0.994       0.955
entity_track      ot_bin 305     0.377     0.446       0.489
entity_track   ot_drawer 225     0.316     0.636       0.667
entity_track   ot_placed 318     0.733     0.053       0.472
entity_track       pilot 249     0.538     0.426       0.602

        task  corrupt_step   n  B_says_A  B_says_B
   add_chain             0 272     0.018     0.967
   add_chain             1 260     0.008     0.942
   add_chain             2 256     0.008     0.973
entity_track             0 440     0.466     0.477
entity_track             1 657     0.530     0.292

