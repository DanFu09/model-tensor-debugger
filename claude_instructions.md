# Claude Instructions

I have two ML models that I am trying to debug. They should have the same output.

I have augmented each model with logging, printing out input and output before and after attention, and before and after MLP for each layer.

I would like to get a web app where I can upload all these traces, match them to each other, and visualize the differences.

Here is an example of some of the files for one model:
3_post_attn_pre_resid.pth
3_post_ln_pre_attn.pth
3_post_mlp.pth
3_pre_mlp.pth

And the other:
0_post_attn_pre_resid.pth
0_post_ln_pre_attn.pth
0_post_mlp.pth
0_pre_mlp.pth

These are PyTorch tensors saved.

Shapes may be slightly different, but I want help examining these as well.

Help me build the app. I want to drag and drop a .zip or .tar.gz into the web app.

It should be completely local.

Make a runplan and execute it to help me do this task.