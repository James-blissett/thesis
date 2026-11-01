# Failure descriptions: check by eye (task_resampling_handoff.md §5)

20 failed original LIBERO-10 rollouts from the unseen tasks (headline setup), drawn with seed 20261103 by `failure_types.py`. The draw was spread over the action-based types (10 froze, 0 never grasped because no rollout is of that type, 10 other) and, within a type, over tasks. For each rollout I looked at the agentview frames at policy steps 0, cut/3, 2·cut/3 and cut − 1, rotated upright (strips in `failure_frames/<rollout>.jpg`), and wrote one sentence on what went wrong before reading anything else about the rollout. The rule column is the action-based type. "Run" is the longest stretch of repeated actions before the cut, with the step it starts.

**Rule for "matches":** a "froze" label matches when the frames show the arm stuck: it does not move, or it hovers or presses without progress for a large part of the rollout. An "other" label matches when the arm keeps moving and grasping but gets the task wrong.

| # | rollout | task | type (rule) | run (start) | what went wrong (by eye) | matches |
|---|---|---|---|---|---|---|
| 1 | task05_trial27 | 5: book into the back compartment of the caddy | froze | 80 (t = 12) | By one third of the way the book and mug are off the table (out of view), and the arm holds still, raised above the caddy, until the cut. | yes |
| 2 | task01_trial20 | 1: cream cheese and butter into the basket | froze | 39 (t = 189) | It takes the butter over to the basket, then hovers there and never goes back for the cream cheese. | yes |
| 3 | task02_trial02 | 2: stove on, moka pot on it | froze | 100 (t = 128) | The arm never leaves its starting pose; stove and moka pot are untouched for the whole rollout. | yes (the arm is still from the start, well before the rule's freeze start) |
| 4 | task09_trial25 | 9: yellow-and-white mug into the microwave, close it | froze | 37 (t = 207) | It reaches the mugs but never picks up the yellow-and-white mug, and ends pressed against the white mug. | yes |
| 5 | task07_trial15 | 7: alphabet soup and cream cheese into the basket | froze | 23 (t = 193) | It goes to the alphabet soup can and hovers over it with the gripper open without ever lifting it. | yes |
| 6 | task03_trial08 | 3: black bowl into the bottom drawer, close it | froze | 56 (t = 169) | It reaches the bowl again and again but never lifts it, and ends hovering over it. | yes |
| 7 | task04_trial23 | 4: white mug on left plate, yellow-and-white mug on right plate | froze | 23 (t = 178) | It puts the white mug on the right plate instead of the left one, then stays near it; the yellow-and-white mug is never moved. | no (the error is a wrong placement; the stall only comes after it) |
| 8 | task05_trial03 | 5: book into the back compartment of the caddy | froze | 22 (t = 131) | It picks up the book but holds it beside the caddy and never puts it in. | yes |
| 9 | task01_trial03 | 1: cream cheese and butter into the basket | froze | 38 (t = 190) | It picks up the butter, carries it to the right instead of to the basket, and stops there over the other items. | yes |
| 10 | task02_trial19 | 2: stove on, moka pot on it | froze | 59 (t = 101) | It moves to the stove and stays in place above it; the stove is never turned on and the pot never moved. | yes |
| 11 | task04_trial08 | 4: white mug on left plate, yellow-and-white mug on right plate | other | 4 (t = 112) | It puts the white mug on the right plate instead of the left one, then picks up the yellow-and-white mug, so both mugs are headed for the wrong plates. | yes |
| 12 | task05_trial26 | 5: book into the back compartment of the caddy | other | 2 (t = 22) | It lifts the book and puts it into the caddy, but not into the back compartment. | yes |
| 13 | task07_trial10 | 7: alphabet soup and cream cheese into the basket | other | 6 (t = 66) | The alphabet soup can ends tipped on the basket's rim, not in it, and the arm moves on to the cream cheese without finishing. | yes |
| 14 | task02_trial17 | 2: stove on, moka pot on it | other | 2 (t = 49) | It turns the stove on, then reaches past the moka pot towards the pan and never puts the pot on the stove. | yes |
| 15 | task01_trial06 | 1: cream cheese and butter into the basket | other | 2 (t = 67) | It picks up the butter, then ends reaching to the far right of the table; both items are not in the basket. | yes |
| 16 | task03_trial07 | 3: black bowl into the bottom drawer, close it | other | 3 (t = 160) | It picks up the bowl but swings it away from the drawer, and the bowl ends in the gripper, not in the drawer. | yes |
| 17 | task06_trial05 | 6: white mug on the plate, chocolate pudding to the right of the plate | other | 2 (t = 138) | It puts the white mug on the plate, then picks up the pudding but has not set it down to the right of the plate by the cut. | yes |
| 18 | task09_trial15 | 9: yellow-and-white mug into the microwave, close it | other | 3 (t = 90) | It puts the mug in the microwave but does not close the door. | yes |
| 19 | task04_trial02 | 4: white mug on left plate, yellow-and-white mug on right plate | other | 2 (t = 30) | It puts the white mug on the right plate instead of the left one, then reaches for the yellow-and-white mug. | yes |
| 20 | task05_trial24 | 5: book into the back compartment of the caddy | other | 5 (t = 1) | It lifts the book over the caddy but is still holding it above the back compartment at the cut. | yes |

**Result:** the action-based type matches 19 of the 20 (froze 9 of 10, other 10 of 10). Most "froze" labels are right, so the 20-step threshold is unchanged.

**What the check shows beyond the count:**
- In 7 of the 10 "froze" rollouts the stall is the **end state of a failed attempt** (hovering over an object it could not lift, or holding an object it could not place), and the rule's freeze start falls in the last 20–60 steps before the cut. Only 3 freeze early (#1, #3, #10).
- The rule can date a freeze too late. In #3 the arm is visibly still from the start, but the run of exactly repeated actions only starts at t = 128. Before that, 87% of steps repeat the previous action, but small changes break the runs, and the gripper moves 8 mm in total over those 128 steps.
- "Other" failures are mostly **wrong-goal or unfinished manipulation** (wrong plate, wrong compartment, object on the rim, door left open, second object not finished). Nothing in the logged actions marks them as abnormal.
